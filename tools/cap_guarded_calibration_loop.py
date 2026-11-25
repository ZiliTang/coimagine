"""Freeze three development-only PUSH effect models and exercise the same paired decision panel."""
from collections import Counter
import hashlib
import json
from math import floor, fsum
from pathlib import Path

from experiments.planar_tasks.authoring_v2 import guarded_options, option_effect_model, public_option_particles
from tools import cap_guarded_calibration_forks as forks
from tools.cap_guarded_calibration_metrics import paired_changes, summarize

RULE = Path(guarded_options.__file__).with_name("guarded_face_change_rule.json")
ROOT = forks.OUT
PUBLIC = ("original_action", "features", "public_history")
LOW_RISK, HIGH_RISK = .20, .80


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path, content):
    if path.exists():
        if read(path) != content:
            raise ValueError(f"frozen file differs at {path}; use a new version for changed inputs")
    else:
        forks.write_json(path, content)


def contract_verdicts(outcomes, rule):
    branches = []
    for outcome in outcomes:
        values = dict(forward_mm=floor(outcome["forward_m"] * 1000),
                      static_contact=int(outcome["object_static"]),
                      entry=int(outcome["endpoint_followup_available"]))
        branches.append(dict(observe="boundary", change={k: [v, v] for k, v in values.items()},
                             post={"entry": ["effect", "entry"]}))
    profile = dict(units=dict(ready="flag", forward_mm="mm", static_contact="flag", entry="flag"),
                   cells={"public_anchor": {"ready": [1, 1]}},
                   options={"push_from_contact": dict(requires={"ready": [1, 1]}, quality="empirical",
                                                      outcomes=branches),
                            "change_face_ccw": dict(requires={"entry": [1, 1]})})
    case = guarded_options.compile_policy(rule, profile).check()["cases"][0]
    global_issues = [i["property"] for i in case["issues"] if "branch" not in i]
    if global_issues != ["model_quality"] or len(branches) != len(case["branches"]):
        raise ValueError("unsupported rule or invalid public precondition")
    return [not b["issues"] for b in case["branches"]]


def estimate(model, progress, features, rule):
    p = model.predict(progress, features)
    samples = p["samples"]
    risk = None
    decision = "query"
    if samples:
        passed = contract_verdicts([s["outcome"] for s in samples], rule)
        risk = fsum(s["weight"] for s, ok in zip(samples, passed) if not ok)
        if p["status"] == "local":
            if risk <= LOW_RISK:
                decision = "admit"
            elif risk >= HIGH_RISK:
                decision = "reject"
    return dict(model_decision=decision, support=p["status"], local_rooms=p["local_support_rooms"],
                predicted_contract_risk=risk, predicted_forward_m=p["mean"].get("forward_m"),
                forward_variance=p["variance"].get("forward_m"), forward_interval=p["interval"].get("forward_m"))


def dispatch(model, progress, public, rule, particles, simulate):
    if set(public) != set(PUBLIC):
        raise ValueError("dispatch accepts a public prefix only")
    result = estimate(model, progress, public["features"], rule)
    query = result["model_decision"] == "query"
    result.update(probe_requested=query, calls=0, prefix_ticks=0, candidate_ticks=0,
                  followup_ticks=0, source="model", admitted=result["model_decision"] == "admit")
    if query:
        probe = particles.probe(**public, progress_m=progress, simulate=simulate)
        result.update({k: probe[k] for k in ("calls", "prefix_ticks", "candidate_ticks", "followup_ticks",
                                              "compatible_room_count", "matches")})
        if probe["status"] == "simulated":
            result.update(source="fallback", admitted=all(contract_verdicts(
                [s["outcome"] for s in probe["samples"]], rule)), particle_endpoints=probe["samples"])
        else:
            result.update(source="insufficient", admitted=False)
    return result


def packets(split):
    result = {}
    for room, fixture in sorted(forks.panels()[split].items()):
        path = ROOT / "forks" / split / (room + ".json")
        p = read(path)
        if (p["split"], p["room"], p["collector_sha256"], p["content_sha256"], p["trace_sha256"],
            p["policy_sha256"]) != (split, room, forks.frozen_hash(), forks.archive.content_hash(fixture),
                                    sha(forks.trace_path(split, room)), forks.source_lock()["sha256"]):
            raise ValueError(f"frozen input mismatch in {room}")
        expected = {(a["anchor_id"], a["public_history"], dose) for a in p["anchors"] for dose in forks.PROGRESS}
        measured = {(r["anchor_id"], r["public_history"], r["progress_m"]) for r in p["rows"]}
        if expected != measured or len(measured) != len(p["rows"]) or any(r["room"] != room for r in p["rows"]):
            raise ValueError(f"incomplete paired panel for {room}")
        result[room] = p
    return result


def freeze(dev, rule):
    cumulative, versions = [], {}
    rooms = sorted(dev)
    model = None
    for i in range(3):
        acquired = []
        for room in rooms[4*i:4*i+4]:
            for row in dev[room]["rows"]:
                reason = "initial"
                if model is not None:
                    pending = estimate(model, row["progress_m"], row["features"], rule)["model_decision"] == "query"
                    audit = row["anchor_id"] == "early" and row["progress_m"] == min(forks.PROGRESS)
                    if not pending and not audit:
                        continue
                    reason = "query+audit" if pending and audit else ("query" if pending else "audit")
                acquired.append(dict(room=room, anchor_id=row["anchor_id"], progress_m=row["progress_m"],
                                     reason=reason))
                cumulative.append(row)
        model = option_effect_model.EffectModel.fit(cumulative)
        label = f"v{i}"
        path = ROOT / "models" / (label + ".json")
        seal(path, dict(version=label, parent=None if i == 0 else f"v{i-1}", batch_rooms=rooms[4*i:4*i+4],
                        acquired=acquired, model=model.to_dict(), collector_sha256=forks.frozen_hash(),
                        rule_sha256=sha(RULE), predictor_sha256=sha(Path(option_effect_model.__file__))))
        model = option_effect_model.EffectModel.from_dict(read(path)["model"])
        versions[label] = model
    return versions


def donor_index(dev):
    donors = [dict(room=room, anchor_id=a["anchor_id"], **{k: a[k] for k in PUBLIC})
              for room in sorted(dev)[12:] for a in dev[room]["anchors"]]
    return public_option_particles.PublicOptionParticles(donors)


def replay_one(chosen, diagnostic):
    for room, aid, dose in chosen:
        packet = diagnostic[room]
        a = next(a for a in packet["anchors"] if a["anchor_id"] == aid)
        ref = next(r["outcome"] for r in packet["rows"] if (r["anchor_id"],r["progress_m"]) == (aid,dose))
        obs = forks.evaluate(forks.panels()["diagnostic"][room], forks.record_for("diagnostic", room),
                             a, dose, execute_followup=True)
        if obs != ref:
            raise ValueError("selected physical command or continuation failed to reproduce")
        return dict(room=room, anchor_id=aid, progress_m=dose, reproduced=True, outcome=obs)
    return dict(reproduced=None, reason="no_admitted_candidate")


def run():
    rule = read(RULE)
    dev = packets("dev")
    versions = freeze(dev, rule)
    particles = donor_index(dev)
    diagnostic = packets("diagnostic")
    all_rows, selected, stats = {}, {}, {}
    for name, model in versions.items():
        decisions, chosen = {}, []
        for room, p in diagnostic.items():
            for a in p["anchors"]:
                public = {k:a[k] for k in PUBLIC}
                allowed = []
                for dose in forks.PROGRESS:
                    dec = dispatch(model, dose, public, rule, particles, forks.simulate_development_particle)
                    decisions[(room,a["anchor_id"],dose)] = dec
                    if dec["admitted"]:
                        allowed.append(dose)
                if allowed:
                    chosen.append((room, a["anchor_id"], max(allowed)))
        # The decision functions never receive the held physical endpoints or their suffixes.
        rows = []
        for room, p in diagnostic.items():
            for r in p["rows"]:
                key = room, r["anchor_id"], r["progress_m"]
                o = r["outcome"]
                rows.append(dict(room=room, anchor_id=r["anchor_id"], progress_m=r["progress_m"],
                                 public_history=r["public_history"], **decisions[key],
                                 actual_contract_ok=contract_verdicts([o], rule)[0],
                                 actual_forward_m=o["forward_m"], followup_completed=o["followup_completed"]))
        all_rows[name], selected[name] = rows, chosen
        seal(ROOT / "decisions" / (name + ".json"), rows)
        stats[name] = summarize(rows)
        stats[name].update(model_support=dict(Counter(r["support"] for r in rows)),
                           query_match_counts=dict(sorted(Counter(str(r["compatible_room_count"]) for r in rows
                                                                  if r["probe_requested"]).items())),
                           probe_prefix_ticks=sum(r["prefix_ticks"] for r in rows),
                           probe_push_ticks=sum(r["candidate_ticks"] for r in rows),
                           train_count=len(model.to_dict()["rows"]),
                           rooms_with_admission=len({room for room, _, _ in chosen}),
                           selected_boundaries=len(chosen),
                           selected_followup_completed=sum(r["followup_completed"] for r in rows
                               if (r["room"],r["anchor_id"],r["progress_m"]) in set(chosen)))
    def inventory(data):
        measured = [r["outcome"] for p in data.values() for r in p["rows"]]
        return dict(rooms=len(data), eligible=sum(bool(p["anchors"]) for p in data.values()),
                    boundaries=sum(len(p["anchors"]) for p in data.values()), candidates=len(measured),
                    entry=sum(o["endpoint_followup_available"] for o in measured),
                    completed=sum(o["followup_completed"] for o in measured),
                    forward_20mm=sum(o["forward_m"]>=.020 for o in measured),
                    contract_ok=sum(contract_verdicts(measured,rule)),
                    endpoint_static=sum(o["object_static"] for o in measured),
                    followup_reasons=dict(Counter(o["followup_reason"] for o in measured)),
                    candidate_prefix_ticks=sum(o["prefix_ticks"] for o in measured),
                    push_ticks=sum(o["push_ticks"] for o in measured),followup_ticks=sum(o["followup_ticks"] for o in measured))
    result = dict(versions=stats, paired_v2_minus_v0=paired_changes(all_rows),
                  development=inventory(dev), diagnostic=inventory(diagnostic),
                  selected_replay_v2=replay_one(selected["v2"], diagnostic))
    hashes = {p.name:sha(p) for p in (Path(__file__),RULE,Path(forks.__file__),Path(guarded_options.__file__),
              Path(option_effect_model.__file__),Path(public_option_particles.__file__),
              Path(__file__).with_name("cap_guarded_calibration_metrics.py"))}
    seal(ROOT / "manifest.json", dict(sources=hashes, rule_risk_thresholds=[LOW_RISK,HIGH_RISK],
         donors=sorted(dev)[12:], development=sorted(dev), diagnostic=sorted(diagnostic),
         room_files={p["room"]:sha(ROOT/"forks"/p["split"]/(p["room"]+".json")) for p in [*dev.values(), *diagnostic.values()]},
         model_files={n:sha(ROOT/"models"/(n+".json")) for n in versions},
         decisions={n:sha(ROOT/"decisions"/(n+".json")) for n in versions}))
    seal(ROOT / "metrics.json",result)
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({k: [r["admitted"],r["useful"],r["calls"],r["probes_requested"]]
                      for k,r in output["versions"].items()}))
