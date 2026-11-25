"""Recover and verify the progress PPO and parent decision chains on the same panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import torch

from tools import cap_abc_event_ppo as event


RUN = event.long.OUT.parent / "cap_abc_ppo_events_20261124_01"
ROOT = Path(__file__).resolve().parents[1]
PANEL_SEED = 662344521
FIGURES = ROOT / "docs/research/key_findings/2026-11-24_abc_ppo_event_comparison"


def data_path(policy):
    return RUN / f"{policy}_routes.json"


def replay_room(key):
    chain = []
    candidate = None
    choice = None
    pending = None
    raw_candidates = event.candidate_rows
    raw_scores = event.base.scores
    raw_review = event.PushReview
    raw_execute = event.s.tick_action

    def candidates(frame, history, obs, proposals, continuation=None):
        nonlocal candidate, choice
        panel = raw_candidates(frame, history, obs, proposals, continuation)
        candidate = (panel, proposals)
        choice = None
        return panel

    def scores(panel, weights):
        nonlocal choice
        assert candidate is not None and candidate[0] is panel
        logits = raw_scores(panel, weights)
        choice = int(logits.argmax())
        return logits

    def record(kind, tick, action):
        nonlocal candidate, choice
        if choice is None:
            assert kind == "boundary" and candidate is None
            selected, matching = "shared", list(event.base.EXPERTS)
        elif choice == event.CONTINUE:
            assert kind != "boundary" and action is None
            selected, matching = "continue", []
        else:
            proposals = candidate[1]
            assert action == proposals[choice]
            selected = event.base.EXPERTS[choice]
            matching = [cap for cap, proposal in zip(event.base.EXPERTS, proposals) if proposal == action]
        chain.append(dict(tick=tick, kind=kind, choice=selected, matching=matching, command=action))
        candidate = choice = None

    def review(mode, decide, **settings):
        def capture(world, obs, action, left, reason):
            assert candidate is None and choice is None
            replacement = decide(world, obs, action, left, reason)
            if choice is not None:
                record(reason, obs["tick"], replacement)
            else:
                assert replacement is None and candidate is None
            return replacement

        return raw_review(mode, capture, **settings)

    def execute(world, action, controller, update, counts, *, review=None):
        nonlocal pending
        if pending is None:
            record("boundary", event.s.observe(world)["tick"], action)
        else:
            assert action == pending and candidate is None
        pending = raw_execute(world, action, controller, update, counts, review=review)
        return pending

    with (patch.object(event, "candidate_rows", candidates),
          patch.object(event.base, "scores", scores),
          patch.object(event, "PushReview", review),
          patch.object(event.s, "tick_action", execute)):
        result = event.episode(key, WEIGHTS, MODE, time_ticks=PLAN["time_ticks"],
                               progress_m=PLAN["progress_m"])
    if result != EXPECTED[key]:
        changed = [field for field in result if result[field] != EXPECTED[key].get(field)]
        raise ValueError(f"{key}: replay disagreed with frozen evaluation in {changed}")
    reviews = [row for row in chain if row["kind"] != "boundary"]
    assert len(reviews) == result["reviews"].get("eligible", 0)
    assert sum(row["choice"] != "continue" for row in reviews) == result["reviews"].get("switched", 0)
    assert all(first["tick"] < second["tick"] for first, second in zip(chain, chain[1:]))
    return dict(room=key, success=result["success"], ticks=result["ticks"],
                history=result["history"], events=chain)


def collect(workers, policy):
    global PLAN, EXPECTED, WEIGHTS, MODE
    PLAN = event.base.read(RUN / "protocol.json")
    for source, digest in PLAN["sources"].items():
        if event.long.ppo.sha(ROOT / source) != digest:
            raise ValueError(f"frozen source changed: {source}")
    setup = event.base.setup()
    assert all(setup[key] == PLAN[key] for key in ("profile", "executor"))
    assert setup["expert_sha"] == PLAN["experts"]
    assert PLAN["contact_ticks"] == 3
    cases = event.base.get_rooms(PANEL_SEED)
    keys = PLAN["confirmation"]
    assert len(keys) == len(cases) == 200 and set(keys) == set(cases)
    event.base.ROOMS.update(cases)
    evaluation = event.base.read(RUN / "evaluation" / f"{policy}.json")
    finished = event.base.read(RUN / "finished.json")
    EXPECTED = evaluation["rooms"]
    assert set(EXPECTED) == set(keys)
    MODE = "original" if policy == "parent" else "progress"
    stage = 0 if policy == "parent" else finished["rounds"]
    actor = event.actor_path(RUN, MODE, stage)
    assert event.long.ppo.sha(actor) == evaluation["policy_sha"]
    if policy == "progress":
        assert evaluation["policy_sha"] == finished["actors"]["progress"]
    WEIGHTS = event.load_weights(actor)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        episodes = []
        for i, trace in enumerate(pool.map(replay_room, keys), 1):
            episodes.append(dict(number=i, **trace))
            if i % 20 == 0:
                print(f"verified {i}/{len(keys)} decision chains", flush=True)
    return dict(policy_sha=evaluation["policy_sha"], panel_seed=PANEL_SEED, trace_parity=len(episodes),
                successes=sum(row["success"] for row in episodes), episodes=episodes)


def render(data, target, policy, width):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    episodes = data["episodes"]
    n = len(episodes)
    assert data["trace_parity"] == n == 200
    background, ink, muted = "#10141d", "#eef3f9", "#adbbcd"
    cell_colors = {"A": "#6ce4ff", "B": "#ffd479", "C": "#d7aaff",
                   "continue": "#68778d", "shared": "#c5d1dc"}
    status_colors = ("#ed7c84", "#64d69b")
    symbols = {value: i for i, value in enumerate(cell_colors, 1)}
    matrix = np.zeros((n, width), dtype=np.uint8)
    for index, row in enumerate(episodes):
        assert row["number"] == index + 1
        matrix[index, :len(row["events"])] = [symbols[e["choice"]] for e in row["events"]]
    outcomes = np.array([[row["success"]] for row in episodes], dtype=np.uint8)

    for path in font_manager.findSystemFonts():
        if Path(path).name.startswith(("NotoSansSC", "wqy")):
            font_manager.fontManager.addfont(path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = [name for name in ("Noto Sans SC", "WenQuanYi Micro Hei") if name in available]
    if not chinese_fonts:
        raise ValueError("install Noto Sans SC or WenQuanYi Micro Hei to render the Chinese labels")
    with plt.rc_context({"font.family": [*chinese_fonts, "DejaVu Sans"], "font.size": 10,
                         "text.color": ink, "axes.labelcolor": ink, "savefig.facecolor": background}):
        fig = plt.figure(figsize=(15.3, 12), facecolor=background)
        grid = fig.add_gridspec(1, 2, left=.075, right=.976, bottom=.10, top=.874,
                               wspace=.012, width_ratios=[.014, .986])
        outcome_ax = fig.add_subplot(grid[0])
        route_ax = fig.add_subplot(grid[1], sharey=outcome_ax)
        outcome_ax.imshow(outcomes, vmin=0, vmax=1, cmap=ListedColormap(status_colors),
                          extent=(0, 1, n+.5, .5), aspect="auto", interpolation="nearest")
        route_ax.imshow(matrix, vmin=0, vmax=len(cell_colors),
                        cmap=ListedColormap(["#1b2533", *cell_colors.values()]),
                        extent=(.5, width+.5, n+.5, .5), aspect="auto", interpolation="nearest")
        route_ax.set_facecolor("#1b2533")
        outcome_ax.set_title("成败", fontsize=9, color=muted, pad=6)
        outcome_ax.set_ylabel("局序号", labelpad=12)
        outcome_ax.set_xticks([])
        outcome_ax.set_yticks([1, *range(20, n+1, 20)])
        route_ax.set_xlabel("决策事件序号", labelpad=8)
        step = 20 if width <= 240 else 40
        route_ax.set_xticks([1, *range(step, width+1, step)])
        route_ax.tick_params(axis="y", left=False, labelleft=False)
        for ax in (outcome_ax, route_ax):
            ax.tick_params(colors=muted, length=0, pad=6)
            for y in range(20, n, 20):
                ax.axhline(y+.5, color="#edf2f8", alpha=.13, lw=.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#364357")

        title, detail = ("共同父模型", "20,000 局检查点 · 原命令边界") if policy == "parent" else (
            "进度触发 PPO", "追加 40,000 局的冻结模型")
        fig.text(.075, .955, f"{title}｜终点 200 局决策链", fontsize=24, weight="bold", va="top")
        fig.text(.075, .913, f"{detail}    成功 {data['successes']}/200    "
                 f"逐局核对 {data['trace_parity']}/200", fontsize=11, color=muted, va="top")
        meanings = {"A": "CAP A", "B": "CAP B", "C": "CAP C", "continue": "继续", "shared": "三者同案"}
        handles = [Patch(color=color, label=meanings[key]) for key, color in cell_colors.items()]
        handles.extend([Patch(color=status_colors[1], label="成功"), Patch(color=status_colors[0], label="失败")])
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.072, .028),
                   ncol=len(handles), frameon=False, labelcolor=ink, fontsize=10.5)
        fig.text(.976, .039, "深色空白为该局已结束", ha="right", fontsize=9.5, color=muted)
        pending = target.with_suffix(".pending.png")
        fig.savefig(pending, dpi=180)
        plt.close(fig)
    pending.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--policy", choices=("progress", "parent"), default="progress")
    parser.add_argument("--render-only", action="store_true", help="use the verified traces already on disk")
    args = parser.parse_args()
    path = data_path(args.policy)
    if not args.render_only:
        torch.set_num_threads(1)
        result = collect(args.workers, args.policy)
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        pending.replace(path)
        print(path)
    result = event.base.read(path)
    other_path = data_path("progress" if args.policy == "parent" else "parent")
    sets = [result]
    if other_path.is_file():
        other = event.base.read(other_path)
        assert [row["room"] for row in other["episodes"]] == [row["room"] for row in result["episodes"]]
        sets.append(other)
    width = max(len(row["events"]) for values in sets for row in values["episodes"])
    target = FIGURES / f"{args.policy}_routes.png"
    render(result, target, args.policy, width)
    print(target)
