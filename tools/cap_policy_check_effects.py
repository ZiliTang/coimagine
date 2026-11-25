"""Join frozen policy-check packets to measured old-P commands on the identical public prefix."""
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.authoring_v2.policy_checks import check_boundary


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs/semantic_rl"
CHECK = EXPERIMENTS / "cap_policy_checker_v1"
STUDY = EXPERIMENTS / "cap_safe_nominal_v1"
MIN_GAIN = .02


def read(path):
    return json.loads(path.read_bytes())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_tick(rows, tick):
    matches = [r for r in rows if r["tick"] == tick]
    if len(matches) != 1:
        raise ValueError(f"expected one boundary at {tick}, got {len(matches)}")
    return matches[0]


def check_anchor(filename, source, stratum):
    saved = read(filename)
    meta = saved["metadata"]
    room, tick = meta["room"], meta["tick"]
    full = read(source)
    step = at_tick(full["steps"], tick)
    boundary = at_tick(full["boundaries"], tick)
    if (step["pending"] or boundary["action"] != step["action"]
            or boundary["public_history"] != meta["history_sha256"]
            or digest(saved["observation"]) != meta["observation_sha256"]
            or saved["observation"]["tick"] != tick):
        raise ValueError(f"prefix, observation, or action mismatch at {room}/{tick}")
    obs = saved["observation"]
    goal = obs["goal"]
    old = step["action"]
    subject = [old["a0"], old["a1"]]
    if goal["target_id"] != old["target_id"] or goal["xy"] != step["goal"]:
        raise ValueError(f"evaluated target differs at {room}/{tick}")
    packet, menu = saved["packet"], saved["menu"]
    public_menu = [p for p in obs["entity_legal_pairs"].get(goal["target_id"], []) if min(p) < 32]
    if {tuple(p) for p in public_menu} != {tuple(p) for p in menu}:
        raise ValueError(f"public PUSH menu differs at {room}/{tick}")
    candidate = saved["action"]
    selected = [candidate["a0"], candidate["a1"]]
    if candidate["target_id"] != goal["target_id"] or min(selected) >= 32:
        selected = None
    if selected != saved["selected"]:
        raise ValueError(f"recorded proposal binding differs at {room}/{tick}")
    rep = check_boundary(public_menu, packet, selected=selected)
    review = next(r for r in packet["reviews"] if r["pair"] == subject)
    end = step["end_tick"]
    forward = math.dist(step["truth"]["object"][:2], step["goal"]) - math.dist(step["end_truth"]["object"][:2], step["goal"])
    phase_endpoints = [pose for phase in step.get("phase_truth", [step]) for pose in (phase["truth"], phase["end_truth"])]
    if not phase_endpoints or step["truth"] != phase_endpoints[0] or step["end_truth"] != phase_endpoints[-1]:
        raise ValueError(f"incomplete phase endpoints at {room}/{tick}")
    controls = [seg for call in step["segments"] for seg in call["segments"]]
    if (not controls or controls[0]["start_tick"] != tick or controls[-1]["end_tick"] != end
            or any(a["end_tick"] != b["start_tick"] for a,b in zip(controls, controls[1:]))):
        raise ValueError(f"incomplete actual controls at {room}/{tick}")

    # Public contact sensors are also logged during controller reviews.
    # Only strictly after-issue measurements count as evidence of post-command contact.
    post = [(r["tick"], r["contact"]) for r in full["reviews"] if tick < r["tick"] <= end]
    post.extend((r["tick"], r["contact"]) for r in full["steps"] if r["tick"] == end)
    contacts = []
    for sample_tick, flags in post:
        actual = next(c for c in controls if c["start_tick"] < sample_tick <= c["end_tick"])
        if any(touch and code < 32 for touch, code in zip(flags, actual["pair"])):
            contacts.append(sample_tick)
    return dict(
        room=room, tick=tick, rank=meta["rank"], stratum=stratum, end_tick=end,
        old_a0=old["a0"], old_a1=old["a1"], old_requested_ticks=old["ticks"],
        old_executed_ticks=end-tick, old_verdict=review["verdict"], old_review_reason=review.get("reason"),
        old_visited=subject in packet["proposed"], old_selected_pair=(saved["selected"] == subject),
        exact_candidate_command_matches=(saved["action"] == old),
        candidate_a0=saved["action"]["a0"], candidate_a1=saved["action"]["a1"],
        candidate_requested_ticks=saved["action"]["ticks"], old_goal_gain_m=forward,
        old_goal_gain_ge_2cm=forward >= MIN_GAIN,
        object_static_contact_sampled=any(pose["object_static"] for pose in phase_endpoints),
        static_sampling="phase_endpoints" if "phase_truth" in step else "command_endpoints",
        pusher_touch_at_issue=any(flag and code < 32 for flag, code in zip(step["contact"], subject)),
        pusher_touch_sampled_after_issue=bool(contacts), post_issue_contact_sample_ticks=len(set(t for t,_ in post)),
        legal_push_pairs=len(public_menu), passing_pairs=rep["counts"]["passed"],
        omitted_passing_pairs=len(rep["omitted_passing"]),
        omitted_passing_pairs_unmeasured=sum(pair != subject for pair in rep["omitted_passing"]),
        history_sha256=meta["history_sha256"], observation_sha256=meta["observation_sha256"],
        old_raw_sha256=checksum(source), packet_sha256=checksum(filename))


def dev_fork(anchor):
    fork = next(p for p in read(STUDY/"forks/risk/pairs.json") if p["room"] == anchor["room"])
    if (fork["anchor"]["tick"] != anchor["tick"] or fork["anchor"]["history"] != anchor["history_sha256"]
            or fork["anchor"]["observation"] != anchor["observation_sha256"]):
        raise ValueError("development fork did not replay matching prefix")
    trial = read(STUDY/"forks/risk/suffix"/(anchor["room"] + ".json"))
    window = trial["frames"][anchor["tick"]+1:anchor["tick"]+121]
    if len(window) != 120:
        raise ValueError("incomplete development window")
    report = {which: {k:fork[which][k] for k in
                    ("window_ticks","goal_advance_m","object_static_ticks","pusher_target_contact_ticks","terminal_success")}
              for which in ("base","safe")}
    report.update(room=anchor["room"], tick=anchor["tick"],
                  safe_window_any_robot_target_touch_ticks=sum(bool(r[2]) for r in window),
                  safe_window_translation_push_ticks=sum(bool(r[4]) for r in window))
    return report


def main():
    prereg = read(CHECK/"selection.json")
    prefixes = read(CHECK/"prefixes.json")["ticks"]
    dev = prereg["development_control"]["room"]
    strata = {room:k for k, rooms in prereg["strata"].items() for room in rooms}
    if set(strata) | {dev} != set(prefixes):
        raise ValueError("registered rooms do not match the prefix index")
    if checksum(STUDY/"room_records.json") != prereg["room_records_sha256"]:
        raise ValueError("registered source rooms changed")
    expected = {f"{room}_t{t:04d}.json" for room,ticks in prefixes.items() for t in ticks}
    files = sorted((CHECK/"packets").glob("*.json"))
    if {p.name for p in files} != expected:
        raise ValueError("incomplete anchor packet set")
    rows = []
    for path in files:
        room, _ = path.stem.rsplit("_t", 1)
        split = "dev" if room == dev else "confirm"
        rows.append(check_anchor(path, STUDY/f"runs/{split}/P/full"/(room+".json"), strata.get(room, "development")))
    confirm = [r for r in rows if r["room"] != dev]
    groups = {}
    for verdict in sorted({r["old_verdict"] for r in confirm}):
        group = [r for r in confirm if r["old_verdict"] == verdict]
        groups[verdict] = dict(n=len(group),ge_2cm=sum(r["old_goal_gain_ge_2cm"] for r in group),
                   sampled_static_contact=sum(r["object_static_contact_sampled"] for r in group),
                   pusher_contact_sampled_after_issue=sum(r["pusher_touch_sampled_after_issue"] for r in group),
                   exactly_same_command=sum(r["exact_candidate_command_matches"] for r in group))
    report = dict(
        panel=prereg["panel"], rooms=len({r["room"] for r in confirm}), anchors=len(confirm),
        by_old_pair_gate_verdict=groups,
        evaluated_action_instances=sum(r["legal_push_pairs"] for r in confirm),
        unmeasured_action_instances=sum(r["legal_push_pairs"]-1 for r in confirm),
        omitted_passing_unmeasured=sum(r["omitted_passing_pairs_unmeasured"] for r in confirm),
        development_control=dev_fork(next(r for r in rows if r["room"] == dev)),
        resolution="Matched old-P command start/end target centre truth. Static contact sampled at physics phase endpoints. Pusher contact sampled after issue using public command boundaries/option callbacks and the actual control active at the sample. Absence of a sampled contact does not imply continuous absence.",
        scope="Reviews refer to a joint actuator pair at the public prefix. Durations, subsequent decisions and actual control phases may differ. Measured movement under the historical command does not validate a candidate time window, prove safety or attribute the displacement to an individual robot. Other candidate pairs at held anchors were not executed by this check. The separate development counterfactual has tick-resolved physics.",
        csv="physical_checks.csv", selection_sha256=checksum(CHECK/"selection.json"))
    with (CHECK/"physical_checks.csv.tmp").open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (CHECK/"physical_checks.csv.tmp").replace(CHECK/"physical_checks.csv")
    temp=CHECK/"physical_checks.json.tmp"
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    temp.replace(CHECK/"physical_checks.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
