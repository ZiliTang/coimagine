"""Build the checkpoint index and the file hash manifest for the delivered artefacts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def run_index(run_dir):
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return None
    entries = []
    for weight in sorted(checkpoints.glob("*.pt")):
        sidecar = read_json(weight.with_suffix(".json"))
        entries.append(dict(file=str(weight.relative_to(run_dir)),
                            bytes=weight.stat().st_size, sha256=sha256_file(weight),
                            sidecar_stage=(sidecar or {}).get("stage"),
                            sidecar_update=(sidecar or {}).get("update"),
                            critic_mode=(sidecar or {}).get("critic_mode"),
                            env_version=(sidecar or {}).get("env_version"),
                            split_version=(sidecar or {}).get("split_version")))
    manifest = read_json(run_dir / "manifest.json", {})
    index = dict(run_dir=str(run_dir), arm=manifest.get("config", {}).get("arm"),
                 critic_mode=manifest.get("config", {}).get("critic_mode"),
                 stage=manifest.get("config", {}).get("stage"),
                 status=manifest.get("status"), deadline_unix=manifest.get("deadline_unix"),
                 warm_start=manifest.get("config", {}).get("warm_start"),
                 warm_start_sha256=manifest.get("config", {}).get("warm_start_sha256"),
                 panel_contract=manifest.get("config", {}).get("panel_contract"),
                 checkpoints=entries)
    lt.atomic_json(checkpoints / "INDEX.json", index)
    return index


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else lt.STABILITY_ROOT
    indexes = {}
    for (stage, arm), name in lt.STABILITY_RUN_DIRS.items():
        run_dir = root / name
        if not (run_dir / "manifest.json").exists():
            continue
        index = run_index(run_dir)
        indexes[name] = {key: index[key] for key in
                         ("arm", "critic_mode", "stage", "status", "deadline_unix",
                          "warm_start_sha256")}
        indexes[name]["checkpoints"] = len(index["checkpoints"])
    manifest = read_json(root / "manifest.json", {})
    manifest["runs"] = indexes
    manifest["comparison_present"] = (root / "comparison" / "comparison.json").exists()
    manifest["final_test_present"] = (root / "comparison" / "final_test.json").exists()
    manifest["report_present"] = (root / "REPORT.md").exists()
    lt.atomic_json(root / "manifest.json", manifest)
    files = []
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.endswith(".tmp")
                or path.name.endswith(".tmp.png")
                or path.name in ("FILES.sha256.json", "FILES.sha256.txt")):
            continue
        relative = str(path.relative_to(root))
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path)))
    lt.atomic_json(root / "FILES.sha256.json", dict(root=str(root), files=files))
    (root / "FILES.sha256.txt").write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in files))
    print(json.dumps(dict(files=len(files), runs=indexes), indent=1))


if __name__ == "__main__":
    main()
