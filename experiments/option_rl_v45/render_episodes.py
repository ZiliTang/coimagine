"""Render a few greedy episodes from an option-RL checkpoint into one self-contained HTML.

Usage:
    python experiments/option_rl_v45/render_episodes.py \
        --checkpoint outputs/option_rl_v45/run9/checkpoint.pt --episodes 4

The policy is the factorised type/scale head; actions are the deterministic argmax with the
same public feasibility mask as training. Frames are the per-control-period truth from the
environment (used for rendering only; the policy sees the legal 760-dim observation).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.option_rl_v45.train import Policy, SCALE_COUNT, observe_vector, type_mask  # noqa: E402
from experiments.option_rl_v45.scenes import sample_scenes  # noqa: E402
from coimagine.random_start import env as E, spec  # noqa: E402

CONTROL_DT = spec.CONTROL_DT
MAX_SKILLS = 120
ARENA = (4.0, 3.0)


def load_ckpt(path):
    ckpt = torch.load(path, weights_only=False)
    policy = Policy(ckpt["obs_dim"], hidden=ckpt["hidden"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    mean = np.asarray(ckpt["obs_mean"], dtype=np.float64)
    var = np.asarray(ckpt["obs_var"], dtype=np.float64)
    return policy, (lambda x: np.clip((x - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


def run_episode(policy, normalise, instance):
    env = E.RandomStartEnv(instance)
    frames, decisions = [], []
    success = False
    start = len(env.rows)
    for _ in range(MAX_SKILLS):
        obs, m0, m1 = observe_vector(env)
        x = torch.as_tensor(normalise(obs)).float().unsqueeze(0)
        with torch.no_grad():
            (ty0, sc0), (ty1, sc1), value = policy(x)
            ty0 = ty0.masked_fill(~torch.as_tensor(type_mask(m0)), -1e9)
            ty1 = ty1.masked_fill(~torch.as_tensor(type_mask(m1)), -1e9)
            a0 = int(torch.argmax(ty0)) * SCALE_COUNT + int(torch.argmax(sc0))
            a1 = int(torch.argmax(ty1)) * SCALE_COUNT + int(torch.argmax(sc1))
        boundary = env.joint_step(a0, a1, record=True)
        decisions.append({"t": round(env.rows[-1]["time"], 2),
                          "a0": spec.ROBOT_ACTIONS[a0].label,
                          "a1": spec.ROBOT_ACTIONS[a1].label})
        success = bool(boundary["success"])
        if boundary["termination"] is not None:
            break
    for row in env.rows[start:]:
        frames.append({
            "t": round(float(row["time"]), 2),
            "o": [round(float(v), 3) for v in row["true"]["object"][:3]],
            "r": [[round(float(v), 3) for v in robot[:3]] for robot in row["true"]["robots"]],
            "ct": [bool(row["r0"]["contacting_object"]), bool(row["r1"]["contacting_object"])],
            "st": [row["r0"]["stage"], row["r1"]["stage"]],
            "fo": [round(float(row["r0"]["force_norm"]), 2), round(float(row["r1"]["force_norm"]), 2)],
        })
    initial = env.rows[start]["true"]["object"] if env.rows[start:] else env.rows[-1]["true"]["object"]
    return {
        "name": instance.name,
        "goal": list(instance.goal_centre), "radius": spec.GOAL_RADIUS,
        "object_size": [spec.OBJECT_WIDTH, spec.OBJECT_HEIGHT],
        "start_goal_distance": round(math.hypot(instance.goal_centre[0] - initial[0],
                                                 instance.goal_centre[1] - initial[1]), 2),
        "frames": frames, "decisions": decisions,
        "success": success, "termination": env.termination,
    }


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>option RL episodes</title>
<style>
body{margin:0;background:#0b0f14;color:#dfe6f3;font:13px/1.5 system-ui,sans-serif}
header{padding:14px 20px 4px}h1{font-size:17px;margin:0 0 4px}h1 span{color:#8b93a7;font-weight:400}
.row{display:flex;gap:16px;padding:8px 20px;align-items:flex-start;flex-wrap:wrap}
.panel{background:#11161e;border:1px solid #1f2734;border-radius:10px;padding:10px 12px}
canvas{background:#0e131a;border-radius:8px;display:block}
button,select{background:#1b2331;color:#dfe6f3;border:1px solid #2a3547;border-radius:6px;padding:5px 10px}
input[type=range]{width:340px}
.mono{font-family:ui-monospace,monospace;font-size:12px;color:#9aa6b8}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px}
.ok{background:#14361f;color:#4ad991}.bad{background:#3a1a1a;color:#ff6b6b}
</style></head><body>
<header><h1>option RL · greedy episodes <span id="sub"></span></h1>
<div class="mono">blue = object trail · orange = goal · green/magenta = robots (brighter = contacting object) · labels = the option requested at that decision</div></header>
<div id="eps" style="display:flex;gap:8px;flex-wrap:wrap;padding:4px 20px 10px"></div>
<div class="row"><div class="panel">
  <div><button id="play">pause</button> <input type="range" id="slider" min="0" max="1" value="0">
  <span id="t" class="mono"></span></div>
  <canvas id="cv" width="900" height="675"></canvas>
</div><div class="panel" style="min-width:300px">
  <div id="meta"></div><div id="actions" class="mono" style="white-space:pre-wrap"></div>
</div></div>
<script>
const DATA = __DATA__;
const state = {ep: 0, i: 0, playing: true, last: 0};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
const meta = document.getElementById('meta'), actions = document.getElementById('actions'), sub = document.getElementById('sub');
const S = 200;   // px per metre
function toPx(x, y) { return [x * S, cv.height - y * S]; }
function draw() {
  const ep = DATA.episodes[state.ep];
  const f = ep.frames[state.i];
  ctx.clearRect(0, 0, cv.width, cv.height);
  const [ax, ay] = toPx(0, ARENA[1]);
  ctx.strokeStyle = '#2a3547'; ctx.lineWidth = 3;
  ctx.strokeRect(...toPx(0, ARENA[1]), ARENA[0] * S, ARENA[1] * S);
  const [gx, gy] = toPx(ep.goal[0], ep.goal[1]);
  ctx.strokeStyle = '#ffb347'; ctx.setLineDash([5, 4]); ctx.beginPath();
  ctx.arc(gx, gy, ep.radius * S, 0, 7); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#ffb347'; ctx.font = '12px sans-serif'; ctx.fillText('goal', gx + 6, gy - 6);
  ctx.strokeStyle = '#4aa3ff'; ctx.lineWidth = 2; ctx.beginPath();
  ep.frames.slice(0, state.i + 1).forEach((fr, j) => { const [x, y] = toPx(fr.o[0], fr.o[1]); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  const [ox, oy] = toPx(f.o[0], f.o[1]);
  ctx.save(); ctx.translate(ox, oy); ctx.rotate(-f.o[2]);
  ctx.fillStyle = '#c8d0dc'; ctx.fillRect(-ep.object_size[0] / 2 * S, -ep.object_size[1] / 2 * S, ep.object_size[0] * S, ep.object_size[1] * S);
  ctx.restore();
  f.r.forEach((r, k) => {
    const [x, y] = toPx(r[0], r[1]);
    ctx.fillStyle = f.ct[k] ? (k === 0 ? '#6dffb0' : '#ff9bf0') : (k === 0 ? '#2f7d55' : '#6b4f7a');
    ctx.beginPath(); ctx.arc(x, y, 0.18 * S, 0, 7); ctx.fill();
    ctx.strokeStyle = '#0b0f14'; ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(r[2]) * 0.18 * S, y - Math.sin(r[2]) * 0.18 * S); ctx.stroke();
  });
  slider.value = state.i;
  document.getElementById('t').textContent = `t=${f.t.toFixed(2)}s  frame ${state.i + 1}/${ep.frames.length}`;
  meta.innerHTML = `<div><b>${ep.name}</b> <span class="badge ${ep.success ? 'ok' : 'bad'}">${ep.success ? 'success' : (ep.termination || 'running')}</span></div>`
    + `<div class="mono">object start→goal ${ep.start_goal_distance} m · frames ${ep.frames.length}</div>`
    + `<div class="mono">stage r0=${f.st[0]} r1=${f.st[1]} · force ${f.fo[0]}/${f.fo[1]} N</div>`;
  const decided = ep.decisions.filter(d => d.t <= f.t);
  actions.textContent = decided.slice(-6).map(d => `t=${d.t.toFixed(2)}  r0=${d.a0}  r1=${d.a1}`).join('\\n');
}
function tick(ts) {
  if (state.playing && ts - state.last > 55) {
    state.last = ts; state.i++;
    const ep = DATA.episodes[state.ep];
    if (state.i >= ep.frames.length) state.i = 0;
    draw();
  }
  requestAnimationFrame(tick);
}
playBtn.onclick = () => { state.playing = !state.playing; playBtn.textContent = state.playing ? 'pause' : 'play'; };
slider.oninput = () => { state.i = parseInt(slider.value, 10); draw(); };
sub.textContent = DATA.checkpoint;
function selectEpisode(k) {
  state.ep = k; state.i = 0;
  slider.max = DATA.episodes[k].frames.length - 1; slider.value = 0;
  [...document.querySelectorAll('#eps button')].forEach((b, j) => {
    b.style.outline = j === k ? '2px solid #4aa3ff' : 'none';
  });
  draw();
}
DATA.episodes.forEach((ep, k) => {
  const b = document.createElement('button');
  b.textContent = `${k + 1}. ${ep.name} · ${ep.start_goal_distance} m · ${ep.success ? 'success' : (ep.termination || '?')}`;
  b.style.outline = k === 0 ? '2px solid #4aa3ff' : 'none';
  b.onclick = () => selectEpisode(k);
  document.getElementById('eps').appendChild(b);
});
slider.max = DATA.episodes[0].frames.length - 1;
draw(); requestAnimationFrame(tick);
</script></body></html>
"""


def render_episodes(policy, normalise, out_dir, scene_seed: int = 2, count: int = 4) -> Path:
    """Render ``count`` greedy episodes (the nearest goals plus the farthest) into ``out_dir``."""
    scenes = sample_scenes(scene_seed, 16, "test")
    scenes = sorted(scenes, key=lambda s: math.hypot(s.goal_centre[0] - s.object_pose.x,
                                                     s.goal_centre[1] - s.object_pose.y))
    chosen = scenes[: count - 1] + [scenes[-1]] if count > 1 else scenes[:1]
    episodes = [run_episode(policy, normalise, scene) for scene in chosen]
    out = Path(out_dir) / "episodes.html"
    html = TEMPLATE.replace("__DATA__", json.dumps({
        "checkpoint": "in-training checkpoint",
        "arena": list(ARENA), "object_size": [spec.OBJECT_WIDTH, spec.OBJECT_HEIGHT],
        "episodes": episodes}, ensure_ascii=False))
    html = html.replace("ARENA[1]", "DATA.arena[1]").replace("ARENA[0]", "DATA.arena[0]")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/option_rl_v45/run9/checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--scene-seed", type=int, default=2,
                        help="the eval scene seed (0*1000+2 = the default run seed 0)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    policy, normalise = load_ckpt(args.checkpoint)
    out_dir = (args.out.parent if args.out else args.checkpoint.parent)
    out = render_episodes(policy, normalise, out_dir, args.scene_seed, args.episodes)
    if args.out is not None:
        out = args.out
        out.write_text((Path(out_dir) / "episodes.html").read_text())
    summary = [{"out": str(out)}]
    print(json.dumps({"out": str(out), "episodes": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
