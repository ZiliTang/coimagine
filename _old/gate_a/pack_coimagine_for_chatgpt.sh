#!/usr/bin/env bash
set -euo pipefail

# Package only the code subset normally needed for ChatGPT-assisted editing.
# Place this script in the repository root beside src/, configs/, and scripts/.
#
# Usage:
#   bash pack_coimagine_for_chatgpt.sh
#   bash pack_coimagine_for_chatgpt.sh /tmp/coimagine_subset.zip

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="$(dirname "${ROOT}")/coimagine_code_subset_${TIMESTAMP}.zip"
OUTPUT="${1:-${DEFAULT_OUTPUT}}"

for required in src configs scripts; do
  if [[ ! -d "${ROOT}/${required}" ]]; then
    echo "error: missing ${ROOT}/${required}" >&2
    exit 2
  fi
done

python3 - "${ROOT}" "${OUTPUT}" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).expanduser().resolve()
output.parent.mkdir(parents=True, exist_ok=True)

include_roots = [root / "src", root / "configs", root / "scripts"]
root_files = [root / "pack_coimagine_for_chatgpt.sh"]
excluded_dirs = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "venv",
    "outputs",
    "output",
    "build",
    "dist",
}
excluded_suffixes = {".pyc", ".pyo", ".so", ".egg-info"}


def excluded(path: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in excluded_dirs or part.endswith(".egg-info") for part in relative.parts):
        return True
    return path.suffix in excluded_suffixes

files: list[Path] = []
for include_root in include_roots:
    for path in include_root.rglob("*"):
        if path.is_file() and not excluded(path) and path.resolve() != output:
            files.append(path)
for path in root_files:
    if path.is_file() and path.resolve() != output:
        files.append(path)

files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())

print(f"created: {output}")
print(f"files:   {len(files)}")
print("included roots: src/, configs/, scripts/, pack_coimagine_for_chatgpt.sh")
PY
