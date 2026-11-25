"""Author a third CAP on paired, public failure prefixes from two frozen CAPs."""
import argparse
import ast
import time
from collections import Counter, deque
from pathlib import Path

from tools import cap_authored_handoff as pair
from tools import cap_authored_handoff_graph as graph

OUT = pair.OUT.parent / "cap_authored_third_20261123_01"
WORK = pair.WORK.parent / "cap_handoff_c_1123"
CAM = OUT / "campaign"
DOC = pair.DOC.parent / "2026-11-23_authored_third"
SEED = 662324591
REVISIONS = (1, 2, 3)
ANCHORS = ("A", "B")
EXECUTOR = pair.g.prep.prior.EXECUTOR
s = pair.s
read, save = pair.read, pair.save


def load_pair():
    member = read(pair.OUT / "member.json")
    assert pair.base.host.RoomCampaign(pair.CAM).verify_frozen() == member["seal"]
    sources = {"A": (pair.WORK / "basis.py").read_text(), "B": Path(member["path"]).read_text()}
    assert {k: pair.base.digest(src) for k, src in sources.items()} == {"A": pair.A, "B": member["sha"]}
    sources = {k: pair.perception.source_for_profile(src) for k, src in sources.items()}
    return sources


def use(sources):
    global MEMBERS
    MEMBERS = {k: s.old.dq.load_native(src)[0] for k, src in sources.items()}


def episode(fixture, start, *, family=None, switch=False, expected=None, scan=False):
    names = (start, "C") if switch else (start,)
    members = {name: s.phase.FrozenCAP(MEMBERS[name]) for name in names}
    counts, hist, evidence = Counter(), deque(maxlen=5), []
    hit = None
    n = 0
    with s.phase.world(fixture) as (world, _):
        def update(history):
            for member in members.values():
                member.update(history)

        update(world.history)
        while not s.ended(world):
            obs = s.observe(world)
            public = pair.sample(obs, hist)
            if hit is None and family and pair.eligible(public, n, family):
                hit = dict(n=n, tick=public["tick"], h=world.history.sha256)
                if expected is not None:
                    assert hit == expected
            who = "C" if switch and hit is not None else start
            action = members[who].propose(obs)
            if scan or (hit and n < hit["n"] + 10):
                evidence.append(dict(public, n=n, action=action, h=world.history.sha256,
                                     execution=[r["execution"] for r in obs["robots"]]))
            s.tick_action(world, action, EXECUTOR, update, counts)
            n += 1
        result = s.output(world, counts)
    if expected is not None:
        assert hit == expected
    return dict(result=result, hit=hit, evidence=evidence)


def scan_one(job):
    key, fixture = job
    dest = OUT / "screen" / (key + ".json")
    if not dest.exists():
        save(dest, {arm: episode(fixture, arm, scan=True) for arm in ANCHORS})


def select(rows):
    def cases(family):
        hits = {}
        for key, episodes in rows.items():
            hits[key] = {}
            for arm, row in episodes.items():
                trace = row["evidence"]
                point = next((x for x in trace if pair.eligible(x, x["n"], family)), None)
                if point:
                    repeat = sum(x["flags"][pair.FAMILIES.index(family)] for x in trace[-8:])
                    hits[key][arm] = (point, repeat)
        return hits

    missing = {key for key, episodes in rows.items() if all(not x["result"]["success"] for x in episodes.values())}
    plans, rates = [], {}
    for family in pair.FAMILIES:
        hits = cases(family)
        selected = {}
        available = {"failure": Counter(), "preserve": Counter()}
        for label, n_cases in (("failure", 8), ("preserve", 4)):
            for arm in ANCHORS:
                eligible = [k for k in hits if arm in hits[k] and
                            ((k in missing) if label == "failure" else rows[k][arm]["result"]["success"])]
                available[label][arm] = len(eligible)
                ranked = sorted(eligible, key=lambda k: (-int(hits[k][arm][1] >= 2) if label == "failure" else 0,
                                                          pair.key(f"C:{family}:{label}:{arm}:{k}")))
                for k in [v for v in ranked if v not in selected][:n_cases]:
                    selected[k] = (arm, hits[k][arm][0], label)
        persistent = sum(k in missing and any(n >= 2 for _, n in origins.values()) for k, origins in hits.items())
        total = sum(bool(x) for x in hits.values())
        rates[family] = dict(available, persistent_joint_failures=persistent, eligible_rooms=total, selected=len(selected))
        if len(selected) == 24:
            plans.append((persistent / total, persistent, family, selected))
    if not plans:
        raise ValueError("no eligible fault family supplies 24 distinct balanced author cases")
    _, _, family, selected = max(plans, key=lambda x: x[:3])
    return family, selected, rates, len(missing)


def prepare():
    sources = load_pair()
    use(sources)
    fixtures = pair.dev()
    pair.pool(scan_one, list(fixtures.items()), 24)
    rows = {k: read(OUT / "screen" / (k + ".json")) for k in fixtures}
    family, selected, rates, failed = select(rows)
    refs, packet = {}, []
    for key, (arm, point, label) in selected.items():
        ref = {k: point[k] for k in ("n", "tick", "h")}
        refs[key] = dict(ref, arm=arm, label=label)
        item = rows[key][arm]
        trace = item["evidence"]
        n = point["n"]
        packet.append(dict(room=key, parent=arm, label=label, take_over=ref,
                           outcome={u: rows[key][u]["result"]["success"] for u in ANCHORS},
                           prefix_transition=trace[max(0, n-5):n+13], last_steps=trace[-4:]))
    plan = dict(family=family, references=refs, rates=rates, profile=pair.perception.DEFAULT,
                executor=EXECUTOR, parent_sha={k: pair.base.digest(v) for k, v in sources.items()},
                dev_content={k: s.phase.input_hash(f) for k, f in fixtures.items()},
                jointly_failed=failed, revisions=list(REVISIONS), author_minutes=12,
                confirmation_seed=SEED)
    confirmations = confirm_rooms()
    known = s.collect_identities() | set(plan["dev_content"].values())
    for identifier in (pair.OUT/"confirmation_ids.json",
                       OUT.parent/"cap_authored_router_20261123_01/confirmation_ids.json"):
        known.update(read(identifier).values())
    prior = read(OUT.parent / "cap_ungated_ppo_20261123_01" / "protocol.json")
    known.update(prior["confirm"] + prior["diagnostic"])
    assert len(confirmations) == 200 and not known.intersection(confirmations)
    plan["held_content"] = sorted(confirmations)
    save(OUT / "protocol.json", plan)
    campaign = pair.base.host.RoomCampaign.create(CAM, {k: fixtures[k] for k in refs})
    with campaign.locked():
        campaign.state["model"].update(name="gpt-6.2-sol", reasoning_effort="high")
        campaign.state["limits"].update(attempts=72, resets=72, execution_ticks=129600, feature_batches=21672)
        campaign.state["author_protocol"].update(wall_seconds=720, max_library_policies=1,
            max_revisions=3, probe_design="24 distinct development rooms, one frozen A or B prefix each",
            executor=EXECUTOR, perception=pair.perception.DEFAULT)
        campaign._save()
    save(CAM / "knowledge/transfer_execution_protocol.json",
         {k: campaign.state[k] for k in ("stage1", "limits", "author_protocol")})
    save(CAM / "knowledge/third_member_protocol.json", {k: plan[k] for k in ("family", "profile", "executor", "parent_sha", "revisions", "author_minutes")})
    WORK.mkdir(parents=True, exist_ok=True)
    for arm, text in sources.items():
        (WORK / f"expert_{arm}.py").write_text(text)
    save(WORK / "cases.json", dict(family=family, distribution=rates, failed_rooms=failed, cases=packet))
    print("Prepared", family, "with", failed, "joint failures of", len(rows), flush=True)


def confirm_rooms():
    return {s.phase.input_hash(case["fixture"]): case["fixture"]
            for case in s.phase.cap.rooms.generate(SEED)["cases"]}


def start():
    assert (WORK / "cases.json").is_file()
    c = pair.base.host.RoomCampaign(CAM)
    with c.locked():
        assert not c.state["submissions"]
        c.state["deadline"] = time.time() + 720
        c._save()


def probe_one(job):
    key, fixture, reference, family, revision = job
    arm = reference["arm"]
    expected = {k: reference[k] for k in ("n", "tick", "h")}
    save(OUT / "probes" / str(revision) / (key + ".json"),
         episode(fixture, arm, family=family, switch=True, expected=expected))


def probe(revision):
    if revision not in REVISIONS:
        raise ValueError("use revision 1, 2 or 3")
    text = (WORK / f"v{revision}.py").read_text()
    session = pair.base.host.RoomSession(CAM)
    c = session.c
    with c.locked():
        c._check()
        if any(x["summary"] == f"C revision {revision}" for x in c.state["submissions"]):
            raise ValueError("revision already submitted")
    plan = read(OUT / "protocol.json")
    sources = load_pair()
    assert plan["parent_sha"] == {k: pair.base.digest(v) for k, v in sources.items()}
    cfg = pair.base.stream_config(text)
    names = next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "NAMES")
    labels = dict(candidate=dict.fromkeys(names, "A"), state=dict.fromkeys(cfg["state_names"], "A"))
    sid = session.tool("submit", dict(source=text, summary=f"C revision {revision}", groups=labels))["id"]
    refs = plan["references"]
    jobs = [(k, read(CAM / "tasks" / k / "fixture.json"), v, plan["family"], revision) for k, v in refs.items()]
    with c.locked():
        c._charge(attempts=len(jobs), resets=len(jobs), execution_ticks=1800*len(jobs), feature_batches=301*len(jobs))
        c._save()
    use(dict(sources, C=text))
    pair.pool(probe_one, jobs, 12)
    rows = {k: read(OUT / "probes" / str(revision) / (k + ".json")) for k in refs}
    ticks = sum(v["result"]["ticks"] for v in rows.values())
    decisions = sum(v["result"]["decisions"] for v in rows.values())
    with c.locked():
        c.state["used"]["execution_ticks"] -= 1800*len(rows) - ticks
        c.state["used"]["feature_batches"] -= 301*len(rows) - decisions
        c._save()
    summary = dict(sid=sid, revision=revision, sha=pair.base.digest(text), ticks=ticks, cases=len(rows),
                   rescued=sum(rows[k]["result"]["success"] for k,v in refs.items() if v["label"] == "failure"),
                   lost=sum(not rows[k]["result"]["success"] for k,v in refs.items() if v["label"] == "preserve"))
    save(OUT / f"probe{revision}.json", summary)
    save(WORK / f"results{revision}.json", dict(summary=summary, rows=rows))
    print(summary, flush=True)


def freeze():
    author = read(WORK / "selection.json")
    summary = read(OUT / f"probe{author['revision']}.json")
    path = CAM / "semantics" / (summary["sid"] + ".py")
    assert pair.base.digest(path.read_text()) == summary["sha"]
    assert path.read_bytes() == (WORK / f"v{summary['revision']}.py").read_bytes()
    c = pair.base.host.RoomCampaign(CAM)
    seal = c.verify_frozen() if read(CAM / "campaign.json")["status"] == "frozen" else c.freeze()
    selection = dict(summary, reason=author["reason"], seal=seal, source_sha256=graph.file_sha(path), path=str(path.relative_to(OUT)))
    save(OUT / "member.json", selection)


def evaluate_one(job):
    key, fixture, family = job
    dest = OUT / "held" / (key + ".json")
    if dest.exists():
        return
    modes = {m: episode(fixture, m, family=family) for m in (*ANCHORS, "C")}
    for arm in ANCHORS:
        reference = modes[arm]["hit"]
        switch = episode(fixture, arm, family=family, switch=True, expected=reference) if reference else modes[arm]
        modes[arm+"C"] = switch
    save(dest, {m: dict(result=value["result"], hit=value["hit"]) for m,value in modes.items()})


def evaluate():
    plan = read(OUT / "protocol.json")
    member = read(OUT / "member.json")
    assert pair.base.host.RoomCampaign(CAM).verify_frozen() == member["seal"]
    path = OUT / member["path"]
    assert graph.file_sha(path) == member["source_sha256"]
    text = path.read_text()
    assert pair.base.digest(text) == member["sha"]
    sources = load_pair()
    assert {k: pair.base.digest(v) for k,v in sources.items()} == plan["parent_sha"]
    use(dict(sources, C=text))
    rooms = confirm_rooms()
    assert sorted(rooms) == plan["held_content"]
    pair.pool(evaluate_one, [(k, f, plan["family"]) for k,f in rooms.items()], 24)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "start", "probe", "freeze", "evaluate"))
    p.add_argument("revision", nargs="?", type=int)
    args = p.parse_args()
    if args.command == "probe":
        probe(args.revision)
    else:
        {"prepare": prepare, "start": start, "freeze": freeze, "evaluate": evaluate}[args.command]()
