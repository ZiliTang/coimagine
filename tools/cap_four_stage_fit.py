"""Fit a versioned one-step model from completed offline collections only."""
import argparse
from collections import Counter
from pathlib import Path
import time

from tools.cap_symbolic_general import ROOT, read, save, sha
from experiments.planar_tasks.authoring_v2.four_stage.prediction import JointModel, compact_row
from experiments.planar_tasks.authoring_v2.four_stage.specialization import specialized_schema
from experiments.planar_tasks.relational_search.contract import digest


BASE = ROOT / "outputs/semantic_rl/cap_four_stage_v1"


def load_rows(collection, split, host,mode_profile="contact_v1"):
    if not (collection / "complete.json").exists():
        raise ValueError("collection is not complete")
    if read(collection / "complete.json")["errors"]:
        raise ValueError("resolve collection errors before fitting")
    rows, identities = [], set()
    for path in sorted((collection / split).glob("*/*/branches/*.json.gz")):
        row = read(path)
        if row["status"] != "complete":
            continue
        context_hash=digest(row["public_context"])
        identity = (row["room"], context_hash, row["action"])
        # Both curricula share initial boundaries. Repeated files are not new data.
        if identity in identities:
            continue
        identities.add(identity)
        row["artifact"] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                               public_sha256=row["public_sha256"],context_sha256=context_hash)
        rows.append(compact_row(row, host,mode_profile))
    return rows


def fit(name, collections, policy, version=0, parent=None, normalization="fixed",
        calibration_scope="action",mode_profile="contact_v1"):
    out = BASE / "models" / name
    if out.exists():
        raise FileExistsError(out)
    source = Path(policy).read_text()
    host = specialized_schema(source)
    training, calibration = [], []
    started = time.perf_counter()
    for collection in collections:
        root = BASE / "data" / collection
        training.extend(load_rows(root, "fit", host,mode_profile))
        calibration.extend(load_rows(root, "calibration", host,mode_profile))
    # Across increments, require each physical branch to occur once.
    def deduplicate(rows):
        unique={}
        for row in rows:
            key=(row["room"],row["source"]["context_sha256"],row["action"])
            if key in unique:
                comparable=lambda r:{k:v for k,v in r.items() if k!="source"}
                if comparable(row)!=comparable(unique[key]):
                    raise ValueError("same public context has different repeated labels; audit before deduplication")
            else:
                unique[key]=row
        return list(unique.values())
    training,calibration=deduplicate(training),deduplicate(calibration)
    model = JointModel.fit(training, calibration, version=version, parent=parent, host=host,
                           normalization=normalization,calibration_scope=calibration_scope,mode_profile=mode_profile)
    save(out / "model.json.gz", model.packet)
    save(out / "calibration_inputs.json.gz", calibration)
    summary = dict(version=version, parent=parent, model_sha256=model.sha256,
                   policy_sha256=sha(Path(policy)), metrics=len(host["metrics"]), normalization=normalization,
                   fit_rows=len(training), calibration_rows=len(calibration),
                   fit_rooms=len(model.packet["training_rooms"]),
                   calibration_rooms=len(model.packet["calibration_rooms"]),
                   actions=dict(Counter(r["action"] for r in training)),
                   calibration=model.packet["calibration"],
                   wall_seconds=time.perf_counter()-started,
                   scope="development effect model; no confirmation results")
    save(out / "summary.json", summary)
    hashes = {}
    for path in [Path(policy), Path(__file__), *sorted((ROOT / "experiments/planar_tasks/authoring_v2/four_stage").glob("*.py"))]:
        path = path.resolve()
        dest = out / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = sha(path)
    save(out / "manifest.json", dict(sources=hashes, model_file_sha256=sha(out / "model.json.gz"),
         collections={c: sha(BASE / "data" / c / "sources.json") for c in collections}))
    print({k: v for k, v in summary.items() if k not in ("actions", "calibration")}, flush=True)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--collections", nargs="+", required=True)
    p.add_argument("--policy", default="experiments/planar_tasks/authoring_v2/four_stage/author_round1.py")
    p.add_argument("--version", type=int, default=0)
    p.add_argument("--parent")
    p.add_argument("--normalization", choices=["fixed", "fit_residual_q90"], default="fixed")
    p.add_argument("--calibration-scope", choices=["action", "action_mode"], default="action")
    p.add_argument("--mode-profile",choices=["contact_v1","contact_and_public_stage_v2"],default="contact_v1")
    args = p.parse_args()
    fit(args.name, args.collections, args.policy, args.version, args.parent, args.normalization,
        args.calibration_scope,args.mode_profile)
