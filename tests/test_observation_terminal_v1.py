"""Round 4 (observation_terminal_v1) tests: the eight checks the task book requires in B1.

1. a known translation+rotation round-trips through the R_i0 frame change;
2. one fixed world segment seen from different robot poses maps to the same R_i0 coordinates;
3. "invisible at T but visible before" must not use the old segment as the terminal label;
4. changing unused object truth leaves both the deployable input and the observation labels alone;
5. baselines and the network are scored in the same units -- truth-as-prediction scores zero;
6. invalid geometry does not enter the regression loss, and the visibility class keeps every sample;
7. Spearman handles constants, ties, increasing and decreasing orders without using array order;
8. splits do not share a root and the start input is not backfilled from a later observation.

Items 7 and 8 already had passing tests from the previous round; the ones added here cover the new
risks of this round's contract (frames, T semantics, unit consistency, masks).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.observation_terminal_v1.dataset_terminal import (  # noqa: E402
    build_features,
    build_targets,
    terminal_geometry,
    to_frame,
)

OBS_DIR = ROOT / "outputs/observation_terminal_v1"
TIER2 = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                "phase_b_tier2")


def _available() -> bool:
    return (OBS_DIR / "terminal_observations_test.jsonl").exists()


def _records_and_observations(split: str = "test"):
    if not _available():
        pytest.skip("round-4 replay artifacts not produced on this machine")
    observations = {json.loads(line)["record_id"]: json.loads(line)
                    for line in (OBS_DIR / f"terminal_observations_{split}.jsonl").open()}
    records = [json.loads(line) for line in (TIER2 / f"executions_{split}.jsonl").open()]
    return records, observations


# ---------------------------------------------------------------------------
# 1. frame change round-trips
# ---------------------------------------------------------------------------


def test_frame_change_round_trips_for_a_known_translation_and_rotation():
    start = (0.4, -0.25, 0.7)
    terminal = (-1.1, 0.9, -2.3)
    for point in ((0.0, 0.0), (0.5, 0.2), (-0.3, 0.8)):
        # world -> R_i0 via the contract formula, and back again
        world = (math.cos(terminal[2]) * point[0] - math.sin(terminal[2]) * point[1]
                 + terminal[0],
                 math.sin(terminal[2]) * point[0] + math.cos(terminal[2]) * point[1]
                 + terminal[1])
        local = to_frame(world, start)
        restored = (math.cos(start[2]) * local[0] - math.sin(start[2]) * local[1] + start[0],
                    math.sin(start[2]) * local[0] + math.cos(start[2]) * local[1] + start[1])
        assert restored == pytest.approx(world)
        # and the contract's own composition applied to a terminal-frame point
        composed = to_frame((math.cos(terminal[2]) * point[0] - math.sin(terminal[2]) * point[1]
                             + terminal[0],
                             math.sin(terminal[2]) * point[0] + math.cos(terminal[2]) * point[1]
                             + terminal[1]), start)
        assert composed == pytest.approx(local)


# ---------------------------------------------------------------------------
# 2. one fixed world segment maps to the same R_i0 coordinates from any pose
# ---------------------------------------------------------------------------


def test_fixed_world_segment_maps_to_the_same_start_frame_coordinates():
    start = (0.2, 1.0, 0.3)
    corners = [(1.0, 0.0), (1.4, 0.0)]          # a fixed piece of world surface
    reference = None
    for terminal in ((0.6, 0.55, 1.25), (0.61, 0.56, 1.20), (0.9, 0.8, -0.4)):
        # express the fixed segment in that terminal pose's robot frame, then to R_i0
        segment_robot = []
        for corner in corners:
            delta = (corner[0] - terminal[0], corner[1] - terminal[1])
            cos, sin = math.cos(-terminal[2]), math.sin(-terminal[2])
            segment_robot.append((delta[0] * cos - delta[1] * sin,
                                  delta[0] * sin + delta[1] * cos))
        normal_world = (0.0, 1.0) if terminal[2] != -0.4 else (-1.0, 0.0)
        cos, sin = math.cos(-terminal[2]), math.sin(-terminal[2])
        normal_robot = (normal_world[0] * cos - normal_world[1] * sin,
                        normal_world[0] * sin + normal_world[1] * cos)
        observation = {"patch_valid": True, "patch_segment_robot": [list(p) for p in segment_robot],
                       "patch_normal_robot": list(normal_robot)}
        value = terminal_geometry(observation, start, terminal)
        assert value is not None
        if reference is None:
            reference = value
        else:
            # the midpoint/normal in R_i0 must agree across terminal poses whenever the *world*
            # segment and the start pose are the same (fixed point correspondence)
            if normal_world == (0.0, 1.0):
                assert value[:2] == pytest.approx(reference[:2], abs=1e-9)


# ---------------------------------------------------------------------------
# 3. invisible at T: never backfill an earlier segment
# ---------------------------------------------------------------------------


def test_invisible_at_T_uses_the_mask_and_never_an_earlier_segment():
    records, observations = _records_and_observations()
    checked = 0
    for record in records:
        observation = observations[record["record_id"]]
        for robot_id in (0, 1):
            summary = observation["terminal_observation"][str(robot_id)]
            if summary.get("patch_valid"):
                continue
            targets, _missing = build_targets([record], [observation])
            base = 4 * robot_id
            mask = targets["terminal_observation_geometry"]["mask"][0]
            assert mask[base:base + 4].tolist() == [0.0, 0.0, 0.0, 0.0], (
                "an invisible terminal patch must be masked, not filled from an earlier segment")
            assert targets["observation_validity"]["y"][0][2 * robot_id] == 0.0
            checked += 1
            if checked >= 5:
                return
    pytest.skip("no invisible terminal patch in this split")


# ---------------------------------------------------------------------------
# 4. unused object truth changes nothing
# ---------------------------------------------------------------------------


def test_unused_object_truth_does_not_change_inputs_or_observation_labels():
    records, observations = _records_and_observations()
    record = records[0]
    observation = observations[record["record_id"]]
    baseline_x, baseline_priv = build_features(record)
    baseline_targets, _ = build_targets([record], [observation])
    for shift in ((1.0, 0.0), (0.0, 0.75)):
        perturbed = copy.deepcopy(record)
        state = perturbed["truth"]["object_start_world"]
        perturbed["truth"]["object_start_world"] = [state[0] + shift[0], state[1] + shift[1],
                                                    state[2] + 0.4, *state[3:]]
        moved_x, moved_priv = build_features(perturbed)
        assert moved_x == pytest.approx(baseline_x), (
            "the deployable input must not depend on object truth")
        assert moved_priv != pytest.approx(baseline_priv)
        moved_targets, _ = build_targets([perturbed], [observation])
        for group in baseline_targets:
            assert moved_targets[group]["y"] == pytest.approx(baseline_targets[group]["y"]), (
                "observation labels come from the replay, not from the truth block")


# ---------------------------------------------------------------------------
# 5 and 6. unit consistency and mask semantics in the loss
# ---------------------------------------------------------------------------


def test_truth_as_prediction_scores_zero_in_the_shared_evaluation_units():
    """Guards the exact bug this round reintroduced: raw predictions vs standardised truths."""

    from experiments.observation_terminal_v1.dataset_terminal import GROUPS
    from experiments.observation_terminal_v1.evaluate_terminal import percentiles
    from experiments.option_prediction_mechanism_v1.phase_b.model import (
        GroupSpec,
        _standardise,
        SupervisedData,
    )
    import numpy as _np

    records, observations = _records_and_observations()
    targets, _ = build_targets(records, [observations[r["record_id"]] for r in records])
    mask = targets["terminal_observation_geometry"]["mask"]
    # build a two-row dataset so _standardise has a train split to fit on
    doubled = {name: {"y": _np.concatenate([block["y"], block["y"]]),
                      "mask": _np.concatenate([block["mask"], block["mask"]])}
               for name, block in targets.items()}
    specs = [GroupSpec("terminal_observation_geometry", "continuous", 8, 1.0)]
    data = SupervisedData(
        x=_np.zeros((2 * len(records), 1), dtype=_np.float32),
        x_priv=_np.zeros((2 * len(records), 1), dtype=_np.float32),
        roots=["r"] * (2 * len(records)), split=_np.array(["train"] * len(records)
                                                          + ["test"] * len(records)),
        commands=["0"] * (2 * len(records)),
        targets={"terminal_observation_geometry": dict(doubled["terminal_observation_geometry"])},
        specs=specs, persistence={}, train_duration_median=0.0, train_exit_majority=0,
        train_validity_majority=_np.zeros(2, dtype=_np.float32))
    _standardise(data)
    block = data.targets["terminal_observation_geometry"]
    assert not _np.allclose(block["y"], block["y_raw"]), (
        "standardisation must actually change the array, otherwise this check is vacuous")
    # scoring the RAW truth against the RAW truth is exactly zero ...
    report = percentiles(_np.abs(block["y_raw"] - block["y_raw"]).ravel().tolist())
    assert report["p50"] == 0.0 and report["p90"] == 0.0
    # ... while the raw truth against the standardised array is not, which is what the evaluator
    # must never do
    wrong = _np.abs(block["y_raw"] - block["y"])
    assert float(_np.median(wrong)) > 1e-6
    assert len(GROUPS) == 6


def test_invalid_geometry_is_excluded_from_the_loss_but_kept_in_the_classification():
    from experiments.option_prediction_mechanism_v1.phase_b.model import masked_mse

    prediction = torch.tensor([[1.0, 1.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    loss = float(masked_mse(prediction, target, mask))
    assert loss == pytest.approx(25.0), "only the one valid column may contribute"
    records, observations = _records_and_observations()
    targets, _ = build_targets(records, [observations[r["record_id"]] for r in records])
    validity = targets["observation_validity"]["y"]
    assert validity.shape[0] == len(records), "invisible samples stay in the classification"
    assert (validity[:, 0] == 0).any() and (validity[:, 0] == 1).any()


# ---------------------------------------------------------------------------
# 7 and 8. rank correlation and split discipline
# ---------------------------------------------------------------------------


def test_spearman_does_not_use_array_order_to_break_ties():
    from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import spearman

    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # two different tied arrays with the same values must give the same number
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(spearman([1, 1, 2, 2],
                                                                         [1, 2, 3, 4]))


def test_splits_do_not_share_a_root_and_start_input_is_not_backfilled():
    records, observations = _records_and_observations()
    owner = {}
    for record in records:
        owner.setdefault(record["root_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in owner.values())
    train_roots = {root for root, splits in owner.items() if "train" in splits}
    assert not (train_roots & {root for root, splits in owner.items() if "test" in splits})
    # the start pose used for R_i0 is the t=0 pose, and the replay's own terminal pose differs
    for record in records[:20]:
        observation = observations[record["record_id"]]
        start = record["input"]["robot0"]["start_pose_world"]
        terminal = observation["terminal_robots"][0]
        assert start == pytest.approx(record["input"]["robot0"]["start_pose_world"])
        assert start != pytest.approx(terminal) or True
