"""Checked, frozen author-room development probe with separate physical replay."""
import argparse

from tools.cap_four_stage_campaign import BASE
from tools.cap_symbolic_general import read, save, sha
from experiments.planar_tasks.authoring_v2.four_stage import campaign
from experiments.planar_tasks.authoring_v2.four_stage.prediction import JointModel
from experiments.planar_tasks.authoring_v2.four_stage.execution import run, audit
from experiments.planar_tasks.authoring_v2.four_stage.online import OnlineRollout


def probe(name, author, model, room_index=0, online=False):
    package = BASE / "authors" / author
    receipt = campaign.verify_frozen(package)
    room, fixture = list(read(BASE / "rooms.json")["rooms"]["author"].items())[room_index]
    out = BASE / "author_probes" / name
    if out.exists():
        raise FileExistsError(out)
    model_path = BASE / "models" / model / "model.json.gz"
    save(out / "reservation.json", dict(room=room, model_file_sha256=sha(model_path),
        freeze=receipt, ticks=1800, mode="enforce", online=online,
        online_limits=dict(branches=60,ticks=3000,seconds=15),
        scope="second author development probe; no confirmation"))
    result = run(fixture, (package / "policy.py").read_text(),
                 model=JointModel(read(model_path)), mode="enforce",
                 rollout_factory=OnlineRollout if online else None,
                 heartbeat=lambda v: save(out / "heartbeat.json", v))
    save(out / "result.json.gz", result)
    print(result["summary"], flush=True)
    physical = audit(result)
    save(out / "audit.json.gz", physical)
    print(physical["summary"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--author", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--room-index", type=int, default=0)
    p.add_argument("--online", action="store_true")
    a = p.parse_args()
    probe(a.name, a.author, a.model, a.room_index, a.online)
