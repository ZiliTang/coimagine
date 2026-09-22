#!/usr/bin/env python
"""Run the micro oracle diagnostics and render the sweep figures.

Equivalent to::

    python -m coimagine.cli micro --output outputs/micro --print-table
    python -m coimagine.cli sweep --output outputs/micro/figures
"""

from __future__ import annotations

import sys

from coimagine.cli import main


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = "outputs/micro"
    status = main(["micro", "--output", output, "--print-table"])
    if status:
        return status
    return main(["sweep", "--output", f"{output}/figures"])


if __name__ == "__main__":
    raise SystemExit(run())
