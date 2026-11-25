"""Render the open recovery-round instructions and authorized developer evidence."""
import argparse
from pathlib import Path

from experiments.planar_tasks.authoring_v2.random_room_author import GUIDES


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = ROOT / "experiments/planar_tasks/authoring_v2/recovery_round.md"


def make_prompt(parent, timeline, controls, fork_tools):
    sections = [INSTRUCTIONS.read_text()]
    for title, path in [
        ("本轮母策略源码", parent),
        ("开发前缀与可干预时间线", timeline),
        ("相似的自行恢复开发对照", controls),
        ("本轮实际可用的分叉接口、程序接口和预算", fork_tools),
        *((name, ROOT / name) for name in GUIDES),
    ]:
        content = path.read_text()
        if not content.strip():
            raise ValueError(f"empty input for {title}")
        sections.append(f"## {title}\n\n{content}")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "timeline", "controls", "fork-tools"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(make_prompt(args.parent, args.timeline, args.controls, args.fork_tools))


if __name__ == "__main__":
    main()
