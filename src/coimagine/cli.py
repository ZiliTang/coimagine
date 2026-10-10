"""Command-line entry point for the 2D protocol core.

    python -m coimagine.cli check
    python -m coimagine.cli micro --output outputs/micro
    python -m coimagine.cli micro --horizons 1,2,3 --bits 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from .diagnostics import DiagnosticConfig, run_micro_diagnostics, write_report
from .toy import all_instances
from .toy.model import validate_model


def _parse_int_list(text: str) -> tuple:
    return tuple(int(part) for part in text.split(",") if part.strip())


def _cmd_check(_args: argparse.Namespace) -> int:
    models = all_instances(horizon=3)
    for model in models:
        validate_model(model)
        observation_rows = len(model.observations)
        actions = ", ".join(model.actions)
        print(
            f"{model.name:<18} x={len(model.x_values)} theta={len(model.theta_values)} "
            f"actions=[{actions}] H={model.horizon} observation_rows={observation_rows}"
        )
        print(f"    {model.description}")
    print(f"\n{len(models)} instances validated.")
    return 0


def _cmd_micro(args: argparse.Namespace) -> int:
    config = DiagnosticConfig(
        horizons=_parse_int_list(args.horizons),
        bits=_parse_int_list(args.bits),
        limited_search_depth=args.limited_search_depth,
        output_dir=args.output,
    )
    reports, summary = run_micro_diagnostics(config)
    paths = write_report(reports, summary, args.output)
    print(f"git revision: {summary['git_revision']}")
    print(f"instances: {summary['instances']}")
    print(f"wall clock: {summary['wall_clock_seconds']:.2f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.print_table:
        from .diagnostics import format_gap_markdown

        print()
        print(format_gap_markdown(reports))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .diagnostics import run_sweeps

    paths = run_sweeps(args.output)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_gallery(args: argparse.Namespace) -> int:
    from .grid import default_scenes, render_gallery, render_noise_comparison

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    paths = render_gallery(
        args.output, scenes=default_scenes(), radius=args.radius, formats=formats
    )
    if args.noise_comparison:
        scenes = {scene.static.name: scene for scene in default_scenes()}
        scene = scenes.get(args.noise_scene, scenes[sorted(scenes)[0]])
        paths += render_noise_comparison(
            scene, args.output, radius=args.radius, formats=formats
        )
    for path in paths:
        print(f"  {path}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .diagnostics import format_gate_markdown, run_grid_gate, write_gate_report

    reports, summary = run_grid_gate()
    paths = write_gate_report(reports, summary, args.output)
    print(f"git revision: {summary['git_revision']}")
    print(f"wall clock: {summary['wall_clock_seconds']:.2f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.print_table:
        print()
        print(format_gate_markdown(reports))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coimagine-2d",
        description="CoImagine 2D protocol core: micro oracles and diagnostics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate the micro model tables")
    check.set_defaults(func=_cmd_check)

    micro = sub.add_parser("micro", help="run the micro oracle diagnostics")
    micro.add_argument(
        "--horizons", default="1,3", help="comma-separated horizons (default 1,3)"
    )
    micro.add_argument(
        "--bits", default="0,1,2", help="comma-separated message budgets (default 0,1,2)"
    )
    micro.add_argument(
        "--limited-search-depth", type=int, default=1, help="depth for the limited search"
    )
    micro.add_argument("--output", default="outputs/micro", help="output directory")
    micro.add_argument(
        "--print-table", action="store_true", help="print the gap table to stdout"
    )
    micro.set_defaults(func=_cmd_micro)

    sweep = sub.add_parser("sweep", help="render the budget / cost / noise sweep figures")
    sweep.add_argument(
        "--output", default="outputs/micro/figures", help="figure output directory"
    )
    sweep.set_defaults(func=_cmd_sweep)

    gallery = sub.add_parser(
        "gallery", help="render the Stage 2 scene gallery (ground truth vs each robot)"
    )
    gallery.add_argument(
        "--output", default="outputs/grid/scenes", help="figure output directory"
    )
    gallery.add_argument("--radius", type=int, default=6, help="lidar radius in cells")
    gallery.add_argument(
        "--formats", default="png,pdf", help="comma-separated output formats"
    )
    gallery.add_argument(
        "--noise-comparison",
        action="store_true",
        help="also render a noiseless-versus-noisy information-state figure",
    )
    gallery.add_argument(
        "--noise-scene", default="pillar_occlusion", help="scene for the noise comparison"
    )
    gallery.set_defaults(func=_cmd_gallery)

    gate = sub.add_parser(
        "gate",
        help="run the Stage 2 design gate (C1 room / C2 value) before the grid suite",
    )
    gate.add_argument(
        "--output", default="outputs/grid/gate", help="gate report output directory"
    )
    gate.add_argument("--print-table", action="store_true", help="print the gate tables")
    gate.set_defaults(func=_cmd_gate)

    learn = sub.add_parser(
        "learn",
        help="run the learning probe: can a model discover the contact modes from data?",
    )
    learn.add_argument("--output", default="outputs/grid/learning", help="report directory")
    learn.add_argument("--scenes", type=int, default=400, help="number of scenes to generate")
    learn.add_argument("--epochs", type=int, default=400, help="training epochs")
    learn.add_argument("--seed", type=int, default=0, help="random seed")
    learn.add_argument("--print-table", action="store_true", help="print the probe tables")
    learn.set_defaults(func=_cmd_learn)

    diagnose = sub.add_parser(
        "diagnose",
        help="round-20 diagnosis: unified metrics, exact-model swap, planner audit",
    )
    diagnose.add_argument(
        "--output", default="outputs/grid/diagnosis", help="report directory"
    )
    diagnose.add_argument(
        "--quick", action="store_true", help="small scene counts, for a smoke run"
    )
    diagnose.add_argument(
        "--seeds", default="0,1,2,3,4", help="comma-separated training seeds"
    )
    diagnose.add_argument("--print-table", action="store_true", help="print the summary")
    diagnose.set_defaults(func=_cmd_diagnose)
    return parser


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnostics.diagnosis import format_diagnosis, run_diagnosis

    result = run_diagnosis(
        args.output,
        quick=args.quick,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
    )
    print(f"git revision: {result['meta']['git_revision']}")
    print(f"wall clock: {result['meta']['wall_clock_seconds']:.1f}s")
    print(f"  json: {args.output}/diagnosis.json")
    print(f"  md:   {args.output}/diagnosis.md")
    if args.print_table:
        print()
        print(format_diagnosis(result))
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    from .diagnostics.learning_probe import (
        format_probe_markdown,
        run_learning_probe,
        write_probe_report,
    )

    result = run_learning_probe(n_scenes=args.scenes, epochs=args.epochs, seed=args.seed)
    paths = write_probe_report(result, args.output)
    print(f"git revision: {result.meta['git_revision']}")
    print(f"wall clock: {result.wall_clock_seconds:.2f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.print_table:
        print()
        print(format_probe_markdown(result))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
