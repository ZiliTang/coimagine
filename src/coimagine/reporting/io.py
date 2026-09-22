from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping


def _row(item: object) -> Mapping[str, object]:
    if hasattr(item, "to_dict"):
        return getattr(item, "to_dict")()
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, Mapping):
        return item
    raise TypeError(f"Unsupported record type: {type(item)!r}")


def write_csv(path: Path, items: Iterable[object]) -> None:
    rows = [dict(_row(item)) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, items: Iterable[object]) -> None:
    rows = [dict(_row(item)) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
