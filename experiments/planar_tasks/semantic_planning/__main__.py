import argparse
import json
from pathlib import Path

from .artifact import create_revision, verify_revision


def main():
    parser = argparse.ArgumentParser(description='Compile/verify history-semantic knowledge; no LLM or physics calls.')
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('destination', type=Path)
    build.add_argument('--source', type=Path, default=Path(__file__).with_name('reference.py'))
    build.add_argument('--parent', type=Path)
    build.add_argument('--evidence', type=Path, action='append', default=[])
    build.add_argument('--reason', default='initial hand-written interface seed')
    verify = commands.add_parser('verify')
    verify.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        result = create_revision(args.source, args.destination, parent=args.parent,
                                 evidence=args.evidence, reason=args.reason)
    else:
        result = verify_revision(args.destination)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
