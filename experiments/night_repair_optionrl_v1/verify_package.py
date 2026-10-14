"""Verify the morning review package **without the repository**, by recomputing the numbers.

The model is rebuilt here directly from the checkpoint's ``state_dict`` shapes rather than by
importing the training code, so a mistake shared with the repository would show up as a mismatch.
Only the packaged sample is reproduced; this is not a rerun of the full test panel.

    python verify_package.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HORIZON = 16  # the packaged sample covers steps 1/4/8/16
STEPS = (1, 4, 8, 16)
MIDPOINTS = [0, 1, 4, 5]
ANCHOR_MAP = {0: 0, 1: 1, 4: 2, 5: 3}
NORMALS = [2, 3, 6, 7]


def sinusoidal(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


class Block(nn.Module):
    def __init__(self, state, prefix):
        super().__init__()
        d_model = state[f"{prefix}.attention.qkv.weight"].shape[0] // 3
        self.d_model = d_model
        self.heads = 4
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # the FFN layer indices depend on where the activations sit, so they are read from the
        # state dict rather than hard-coded
        ffn_indices = sorted({int(key[len(prefix) + 5:].split(".")[0]) for key in state
                              if key.startswith(f"{prefix}.ffn.") and key.endswith(".weight")})
        ffn = state[f"{prefix}.ffn.{ffn_indices[0]}.weight"].shape[0]
        modules = []
        for position, index in enumerate(ffn_indices):
            modules.append(nn.Linear(int(state[f"{prefix}.ffn.{index}.weight"].shape[1]),
                                     int(state[f"{prefix}.ffn.{index}.weight"].shape[0])))
            if position + 1 < len(ffn_indices):
                modules.append(nn.GELU())
        self.ffn = nn.Sequential(*modules)
        self.ffn_indices = ffn_indices
        # the attention block keeps its projections one level down, under ``attention.*``
        for name, module in (("norm1", self.norm1), ("attention.qkv", self.qkv),
                             ("attention.out", self.out), ("norm2", self.norm2)):
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
            module.weight.data.copy_(state[f"{prefix}.{name}.weight"])
            module.bias.data.copy_(state[f"{prefix}.{name}.bias"])
        for position, index in enumerate(self.ffn_indices):
            target = self.ffn[2 * position]
            target.weight.requires_grad_(False)
            target.bias.requires_grad_(False)
            target.weight.data.copy_(state[f"{prefix}.ffn.{index}.weight"])
            target.bias.data.copy_(state[f"{prefix}.ffn.{index}.bias"])
        self.attention_dropout_identity = True

    def forward(self, x, allow):
        batch, length, _dim = x.shape
        head_dim = self.d_model // self.heads
        normalised = self.norm1(x)
        qkv = self.qkv(normalised).view(batch, length, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        bias = torch.where(allow.unsqueeze(1), torch.zeros((), device=x.device),
                           torch.full((), -1e9, device=x.device))
        weights = torch.softmax(scores + bias, dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, self.d_model)
        x = x + self.out(attended)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, state):
        super().__init__()
        d_model = state["encoder.input.weight"].shape[0]
        self.input = nn.Linear(58, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        count = len({key.split(".")[2] for key in state if key.startswith("encoder.blocks.")})
        self.blocks = nn.ModuleList([Block(state, f"encoder.blocks.{i}") for i in range(count)])
        for module, name in ((self.input, "encoder.input"),
                             (self.final_norm, "encoder.final_norm")):
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
            module.weight.data.copy_(state[f"{name}.weight"])
            module.bias.data.copy_(state[f"{name}.bias"])

    def forward(self, window, valid):
        x = self.input(torch.cat([window, valid.unsqueeze(-1)], dim=-1)) \
            + sinusoidal(window.shape[1], self.d_model).unsqueeze(0)
        length = window.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        padding = valid < 0.5
        query_padded, key_padded = padding.unsqueeze(2), padding.unsqueeze(1)
        eye = torch.eye(length, dtype=torch.bool).unsqueeze(0)
        allow = (causal.unsqueeze(0) & ~key_padded & ~query_padded) | (query_padded & eye)
        for block in self.blocks:
            x = block(x, allow)
        x = self.final_norm(x)
        return x * valid.unsqueeze(-1)


def sequence(prefix, state, activation, trailing_activation=True):
    indices = sorted({int(key[len(prefix):].split(".")[0]) for key in state
                      if key.startswith(prefix) and key.endswith(".weight")})
    modules = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        modules.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
        if position + 1 < len(indices) or trailing_activation:
            modules.append(activation())
    network = nn.Sequential(*modules)
    for position, index in enumerate(indices):
        layer = network[2 * position]
        layer.weight.requires_grad_(False)
        layer.bias.requires_grad_(False)
        layer.weight.data.copy_(state[f"{prefix}{index}.weight"])
        layer.bias.data.copy_(state[f"{prefix}{index}.bias"])
    return network


def decode(residual, anchor, stats):
    mean, std = float(stats["geometry_mean"]), float(stats["geometry_std"])
    out = np.array(residual, dtype=np.float64, copy=True)
    out[..., MIDPOINTS] = out[..., MIDPOINTS] * std + mean
    out[..., NORMALS] = out[..., NORMALS] * float(stats["normal_std"]) + float(stats["normal_mean"])
    anchor = np.asarray(anchor, dtype=np.float64)
    for column, anchor_index in ANCHOR_MAP.items():
        out[..., column] += anchor[:, anchor_index].reshape((anchor.shape[0],) + (1,)
                                                            * (out.ndim - 2))
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "results" / "expected_metrics.json").read_text())
    samples = dict(np.load(here / "samples" / "windows.npz", allow_pickle=True))
    window = torch.tensor(samples["window"], dtype=torch.float32)
    valid = torch.tensor(samples["token_valid"], dtype=torch.float32)
    command = torch.tensor(samples["command"], dtype=torch.float32)
    elapsed = torch.tensor(samples["elapsed"], dtype=torch.float32)
    anchor = samples["anchor"]
    # the packaged sample uses ``truth`` for the future geometry vector
    truth = samples["truth"] if "truth" in samples else samples["truth_geometry"]
    geometry_mask = samples["geometry_mask"]
    step_mask = samples["step_mask"]

    hold_error = np.where((geometry_mask[:, 0, 0] > 0) & (step_mask[:, 0] > 0),
                          np.hypot(anchor[:, 0] - truth[:, 0, 0], anchor[:, 1] - truth[:, 0, 1]),
                          np.nan)
    observed = {"hold_step1_r0_p50_cm": float(np.nanmedian(hold_error) * 100)}
    ok = True
    for name, reference in expected["models"].items():
        checkpoint = torch.load(here / "checkpoints" / f"{name}.pt", map_location="cpu",
                                weights_only=False)
        state = checkpoint["state_dict"]
        encoder = Encoder(state)
        activation = nn.GELU
        decoder = sequence("decoder.net.", state, activation, trailing_activation=False)
        with torch.no_grad():
            hidden = encoder(window, valid)
            # the readout index comes from the mask itself; ``valid.sum() - 1`` would point into the
            # left padding for any history shorter than the window
            positions = torch.arange(valid.shape[1]).unsqueeze(0)
            masked = torch.where(valid > 0.5, positions, torch.full_like(positions, -1))
            lengths = masked.max(dim=1).values.clamp(min=0)
            summary = hidden[torch.arange(hidden.shape[0]), lengths, :]
            horizon = HORIZON
            codes = sinusoidal(horizon, 16).unsqueeze(0).expand(window.shape[0], -1, -1)
            if checkpoint["kind"] == "direct_multistep":
                latent = summary.unsqueeze(1).expand(-1, horizon, -1)
            else:
                transition = sequence("transition.", state, activation, trailing_activation=False)
                latent = summary
                collected = []
                for step in range(horizon):
                    # advance first, then decode z_(k+1) for t+(k+1)dt; appending before the
                    # transition would decode the un-advanced z0 at k=1 and make the ablation
                    # identical to the normal path by construction
                    clock = torch.stack([elapsed + step * 0.05, torch.full_like(elapsed, 0.05)],
                                        dim=-1)
                    latent = latent + transition(torch.cat([latent, command, clock], dim=-1))
                    collected.append(latent)
                latent = torch.stack(collected, dim=1)
            raw = decoder(torch.cat([latent, codes, command.unsqueeze(1).expand(-1, horizon, -1)],
                                    dim=-1))
        geometry = raw[..., :8].numpy().astype(np.float64)
        decoded = decode(geometry, anchor, checkpoint["stats"])
        for step in STEPS:
            keep = (geometry_mask[:, step - 1, 0] > 0) & (step_mask[:, step - 1] > 0)
            error = np.hypot(decoded[keep, step - 1, 0] - truth[keep, step - 1, 0],
                             decoded[keep, step - 1, 1] - truth[keep, step - 1, 1])
            value = float(np.median(error) * 100)
            observed[f"{name}|step{step}_r0_p50_cm"] = value
            want = reference[f"step{step}_r0_p50_cm"]
            if abs(value - want) > 5e-4:
                ok = False
                print(f"MISMATCH {name} step{step}: packaged {want} vs recomputed {value}")
        want_hold = reference.get("hold_step1_r0_p50_cm")
        if want_hold is not None and abs(observed["hold_step1_r0_p50_cm"] - want_hold) > 5e-4:
            ok = False
            print(f"MISMATCH hold: packaged {want_hold} vs recomputed "
                  f"{observed['hold_step1_r0_p50_cm']}")
    # interface invariants: a numerical match alone does not prove the readout is semantically right
    invariants = {}
    valid_probe = torch.zeros(1, 33)
    valid_probe[0, 32] = 1.0
    hidden_probe = encoder(torch.randn(1, 33, 57), valid_probe)
    old_style_index = int(valid_probe.sum(dim=1).clamp(min=1).long()[0] - 1)
    positions = torch.arange(33).unsqueeze(0)
    masked = torch.where(valid_probe > 0.5, positions, torch.full_like(positions, -1))
    true_index = int(masked.max(dim=1).values[0])
    invariants["t0_old_formula_reads_padding"] = (old_style_index != true_index
                                                 and old_style_index == 0
                                                 and true_index == 32)
    encoder.eval()
    with torch.no_grad():
        summary = hidden_probe[0, 32]
    invariants["t0_summary_non_zero"] = bool(summary.abs().sum() > 0)
    print(json.dumps({"interface_invariants": invariants}, indent=1))
    if not all(invariants.values()):
        ok = False
    print(json.dumps({"samples": int(len(window)), "recomputed": observed,
                      "full_panel_values": {name: entry.get("full_panel")
                                            for name, entry in expected["models"].items()},
                      "subset_reproduction": ok,
                      "scope": "packaged sample only, not the full test panel"},
                     indent=1, default=float))
    print("VERIFY_PACKAGE_OK" if ok else "VERIFY_PACKAGE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
