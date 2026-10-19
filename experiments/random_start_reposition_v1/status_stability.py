"""Compact status of the bounded stability runs; reads saved state only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else lt.STABILITY_ROOT
    rows = []
    for (stage, arm), name in lt.STABILITY_RUN_DIRS.items():
        run_dir = root / name
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        monitor_path = run_dir / "monitor.json"
        monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
        update = monitor.get("update", 0)
        rows.append(dict(stage=stage, arm=arm, dir=name,
                         status=manifest.get("status"),
                         update=f"{update}/{manifest['config']['updates']}",
                         hours=round(monitor.get("elapsed_hours", 0.0), 3),
                         best=monitor.get("best"),
                         latest=monitor.get("latest_evaluation")))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
