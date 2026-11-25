"""Candidate features computed from public decision-boundary observations."""

from math import cos, hypot, pi


feature_names = (
    "Continue productive contact drive",
    "Continue active contact preparation",
    "Change repeated stalled intent",
    "Paired push opposition",
)

_PREPARING = frozenset({"standoff", "final", "approach", "align", "orient", "touch", "ready"})
_WAITING = frozenset({"find", "view_insufficient", "wait", "wait_peer_contact", "peer_wait",
                      "perception_wait", "blocked", "move_blocked"})
_CONTACT_SKILLS = frozenset({"push", "rotate", "reposition"})


def _unit(value):
    return max(0.0, min(1.0, value))


def _intent(action):
    if 0 <= action <= 31:
        return "push", action // 4
    if 32 <= action <= 37:
        return "rotate", (action - 32) // 3
    if action in (38, 39, 40):
        return "reposition", 0
    if action in (41, 42, 43):
        return "reposition", 1
    if action == 44:
        return "brake", 0
    if action == 46:
        return "yield", 0
    if 48 <= action <= 79:
        return "move", (action - 48) // 4
    return "unknown", 0


def _similarity(first, second):
    family, direction = first
    other_family, other_direction = second
    if family != other_family or family == "unknown":
        return 0.0
    if family in ("push", "move"):
        return max(0.0, cos((direction - other_direction) * pi / 4))
    return float(direction == other_direction)


def _joint_intent(action, target_id):
    """Require target association for object-specific commands."""
    intents = (_intent(action["a0"]), _intent(action["a1"]))
    if target_id is not None and action["target_id"] == target_id:
        return intents
    return tuple(("unknown", 0) if intent[0] in _CONTACT_SKILLS else intent
                 for intent in intents)


def _stage(frame, robot):
    stages = frame.get("stages") or ()
    return stages[robot] if robot < len(stages) else ""


def _opposition(candidate, intents):
    if any(family != "push" for family, _ in intents):
        return 0.0
    _, first_direction = intents[0]
    _, second_direction = intents[1]
    first_speed = candidate["a0"] % 4 + 1
    second_speed = candidate["a1"] % 4 + 1
    balance = 2 * min(first_speed, second_speed) / (first_speed + second_speed)
    return min(0.0, cos((first_direction - second_direction) * pi / 4)) * balance


def features(frame, previous, candidate, target_id):
    """Return signed or unit-range measurements for one proposed joint command."""
    new = _joint_intent(candidate, target_id)
    opposition = _opposition(candidate, new)
    if not previous:
        return 0.0, 0.0, 0.0, opposition

    history = list(previous[-4:]) + [frame]
    latest = _joint_intent(history[-2]["action"], target_id)
    retained = [_similarity(intent, prior) for intent, prior in zip(new, latest)]
    drive = [0.0, 0.0]
    preparation = [0.0, 0.0]
    stall_ticks = [0.0, 0.0]
    useful = [0.0, 0.0]

    for before, after in zip(history, history[1:]):
        action = before["action"]
        span = after["tick"] - before["tick"]
        if not 0 < span <= action["ticks"]:
            continue
        old = _joint_intent(action, target_id)
        age = frame["tick"] - after["tick"]
        recency = 60 / (60 + age)
        visible = before["seen"] and after["seen"]
        gap_rate = (before["gap"] - after["gap"]) * 40 / span if visible else 0.0
        closing = _unit(gap_rate / 0.16)
        flat_gap = 1 - _unit(gap_rate / 0.07)

        for robot in range(2):
            family, _ = old[robot]
            if family not in _CONTACT_SKILLS and family != "move":
                continue
            repeated = _similarity(old[robot], latest[robot])
            if repeated <= 0:
                continue
            x0, y0 = before["xy"][robot]
            x1, y1 = after["xy"][robot]
            travel = hypot(x1 - x0, y1 - y0)
            movement = _unit(travel * 40 / (span * 0.16))
            if visible:
                along, cross = before["rel"][2 * robot:2 * robot + 2]
                new_along, new_cross = after["rel"][2 * robot:2 * robot + 2]
                approach = _unit((hypot(along, cross) - hypot(new_along, new_cross))
                                 * 40 / (span * 0.16))
            else:
                approach = 0.0

            stage = _stage(after, robot)
            previous_stage = _stage(before, robot)
            current_stage = _stage(frame, robot)
            touch = bool(after["touch"][robot])
            acquired = touch and not before["touch"][robot]
            contact_drive = 0.0
            if family in ("push", "rotate") and touch:
                if stage == "push":
                    contact_drive = 1.0
                elif not stage and before["touch"][robot]:
                    contact_drive = 0.35
            if visible:
                drive[robot] = max(drive[robot], recency * repeated * contact_drive * closing)

            preparing = 0.0
            if family in _CONTACT_SKILLS and stage in _PREPARING:
                advanced = previous_stage in _PREPARING and previous_stage != stage
                preparing = max(0.75 * movement, 0.5 * advanced, float(acquired))
            elif family == "move" and stage in ("move", ""):
                preparing = approach * movement
            elif family in _CONTACT_SKILLS and not stage:
                preparing = 0.45 * max(approach * movement, float(acquired))
            useful[robot] = max(useful[robot], recency * repeated * preparing)
            if current_stage in _PREPARING or current_stage in ("move", "", "wait_peer_contact"):
                preparation[robot] = max(preparation[robot], recency * repeated * preparing)

            if visible:
                stationary_wait = (stage in _WAITING or stage in _PREPARING or stage == "move")
                stationary_wait *= (stage == previous_stage) * (1 - movement)
                stalled = max(contact_drive, stationary_wait) * flat_gap
                stall_ticks[robot] += span * recency * repeated * stalled

    preserved_drive = max(
        drive[i] * retained[i] * (0.7 + 0.3 * _similarity(new[1 - i], latest[1 - i]))
        for i in range(2)
    )
    preserved_preparation = max(preparation[i] * retained[i] for i in range(2))
    # A change in the affected role matters more than an unrelated peer change.
    # Confidence ramps up after roughly one full command window of evidence.
    active_alternative = any(family not in ("brake", "unknown") for family, _ in new)
    change = 0.0
    for i in range(2):
        if new[i][0] == "unknown":
            continue
        confidence = _unit((stall_ticks[i] - 30) / 55) * (1 - max(drive[i], useful[i]))
        contrast = (1 - retained[i]) * active_alternative - retained[i]
        change += 0.5 * confidence * contrast

    return preserved_drive, preserved_preparation, change, opposition
