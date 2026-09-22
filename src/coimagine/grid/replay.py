"""Episode replays for Stage 2: scan, belief, messages and the committed action.

Each replay step is one decision boundary and shows three panels:

* the simulator ground truth with the true ray scans (evaluation only);
* robot A's information state with the team's posterior over the object pose;
* robot B's information state with the same posterior and the contact-physics
  marginals that drive the decision.

The step figures are also assembled into a GIF so the belief evolution is visible.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .dynamics import execute  # noqa: E402
from .planner import (  # noqa: E402
    PHYSICS,
    ContactContext,
    GridCodec,
    HypKey,
    PBelief,
    PlannerSpec,
    _filtered_belief,
    choose_action,
    contact_candidates,
    initial_contact_belief,
)
from .scenes import mechanism_scenes  # noqa: E402
from .sensing import OccupancyScan, information_state, ray_cast  # noqa: E402
from .state import Cell, Scene  # noqa: E402
from .viz import (  # noqa: E402
    draw_belief_heatmap,
    draw_ground_truth_panel,
    draw_information_panel,
    save_figure,
    shared_legend,
    use_paper_style,
)


@dataclass(frozen=True)
class TraceStep:
    t: int
    scene: Scene
    scans: Mapping[int, OccupancyScan]
    action: str
    y_public: str
    feedback: Mapping[int, str]
    messages: Optional[Tuple[int, int]]
    origin_marginal: Mapping[Cell, float]
    p_constrained: float
    p_resistance2: float
    cost: float


def origin_marginal(belief: PBelief) -> Dict[Cell, float]:
    out: Dict[Cell, float] = {}
    for (origin, _robots, _hyp, _h_a, _h_b), weight in belief.items():
        out[origin] = out.get(origin, 0.0) + weight
    return out


def _physics_marginals(belief: PBelief) -> Tuple[float, float]:
    total = sum(belief.values()) or 1.0
    constrained = 0.0
    heavy = 0.0
    for (_origin, _robots, hyp, _h_a, _h_b), weight in belief.items():
        physics = PHYSICS[hyp[0]]
        if physics.constraint == "constrained" and not hyp[1]:
            constrained += weight
        if physics.resistance >= 2:
            heavy += weight
    return constrained / total, heavy / total


def run_trace(
    scene: Scene,
    true_hyp: HypKey,
    spec: PlannerSpec,
    *,
    horizon: int = 3,
    radius: int = 6,
    codec: Optional[GridCodec] = None,
) -> List[TraceStep]:
    """Run one closed-loop episode and record every boundary."""

    ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
    belief = initial_contact_belief(ctx)
    state = ctx
    hyp = true_hyp
    h_a: Tuple[str, ...] = ()
    h_b: Tuple[str, ...] = ()
    skills: Tuple = ()
    steps: List[TraceStep] = []

    for t in range(horizon):
        candidates = contact_candidates(state)
        scans = {
            robot.robot_id: ray_cast(state.scene(hyp), robot.robot_id, radius)
            for robot in state.robots
        }
        constrained, heavy = _physics_marginals(belief)
        chosen, _value = choose_action(
            spec, ctx, belief, candidates, skills, true_hyp=hyp
        )
        action_label = chosen.candidate_id if chosen is not None else "stop"
        if chosen is None:
            steps.append(
                TraceStep(t, state.scene(hyp), scans, action_label, "stalled", {},
                          None, origin_marginal(belief), constrained, heavy, 0.0)
            )
            break

        successor, result = execute(state.scene(hyp), chosen.joint)
        codec_used = codec or spec.codec
        messages: Optional[Tuple[int, int]] = None
        if codec_used is not None:
            full = skills + (chosen.joint,)
            messages = (
                codec_used.encode(ctx, full, 0, h_a + (result.feedback[0],), candidates),
                codec_used.encode(ctx, full, 1, h_b + (result.feedback[1],), candidates),
            )
        steps.append(
            TraceStep(
                t=t,
                scene=state.scene(hyp),
                scans=scans,
                action=action_label,
                y_public=result.y_public,
                feedback=dict(result.feedback),
                messages=messages,
                origin_marginal=origin_marginal(belief),
                p_constrained=constrained,
                p_resistance2=heavy,
                cost=result.cost,
            )
        )

        belief = _filtered_belief(
            spec, state, belief, chosen.joint, result, h_a, h_b, candidates, skills
        )
        state = ContactContext(
            static=ctx.static, origin=successor.obj.origin, robots=successor.robots
        )
        hyp = (hyp[0], successor.released)
        h_a = h_a + (result.feedback[0],)
        h_b = h_b + (result.feedback[1],)
        skills = skills + (chosen.joint,)
        if successor.is_goal():
            break

    return steps


def _draw_step(step: TraceStep, spec: PlannerSpec, figsize=(9.8, 3.6)):
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    draw_ground_truth_panel(axes[0], step.scene, step.scans, letter="a")

    infos = {
        robot.robot_id: information_state(step.scene, step.scans[robot.robot_id])
        for robot in step.scene.robots
    }
    for ax, robot_id, letter in zip(axes[1:], (0, 1), ("b", "c")):
        draw_information_panel(ax, infos[robot_id], step.scans.get(robot_id), letter=letter)
        draw_belief_heatmap(ax, step.origin_marginal, step.scene.static)

    title = (
        f"{spec.label}  |  t = {step.t}  |  action = {step.action}  |  "
        f"public = {step.y_public}  |  feedback = {dict(step.feedback)}"
    )
    if step.messages is not None:
        title += f"  |  messages = {step.messages}"
    fig.suptitle(title, fontsize=9)
    axes[2].text(
        0.02,
        0.02,
        f"P(constrained) = {step.p_constrained:.2f}\nP(r>=2) = {step.p_resistance2:.2f}",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", linewidth=0.6),
        zorder=30,
    )
    fig.legend(
        handles=shared_legend(), loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(wspace=0.08)
    fig.tight_layout()
    return fig


def render_trace(
    scene: Scene,
    true_hyp: HypKey,
    spec: PlannerSpec,
    *,
    out_dir: str = "outputs/grid/replays",
    stem: str = "replay",
    horizon: int = 3,
    gif: bool = True,
) -> Dict[str, object]:
    """Render the step figures, a contact sheet and optionally a GIF."""

    steps = run_trace(scene, true_hyp, spec, horizon=horizon)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_paths: List[str] = []
    for step in steps:
        fig = _draw_step(step, spec)
        for path in save_figure(fig, str(out), f"{stem}_t{step.t}", formats=("png",)):
            frame_paths.append(path)

    paths: Dict[str, object] = {"frames": frame_paths, "n_steps": len(steps)}
    if gif and frame_paths:
        from PIL import Image

        frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
        gif_path = out / f"{stem}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=700,
            loop=0,
        )
        paths["gif"] = str(gif_path)
    return paths


def render_suite_replays(output_dir: str = "outputs/grid/replays") -> Dict[str, object]:
    """One replay per mechanism scene, using the message-using planner."""

    from .planner import planner_spec

    out: Dict[str, object] = {}
    for scene in mechanism_scenes():
        hypothesis = (PHYSICS.index(scene.physics), False)
        out[scene.static.name] = render_trace(
            scene,
            hypothesis,
            planner_spec("codec", bits=1, depth=2),
            out_dir=output_dir,
            stem=f"replay_{scene.static.name}",
        )
    return out
