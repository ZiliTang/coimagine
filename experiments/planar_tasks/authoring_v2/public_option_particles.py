"""Try one progress-limited PUSH in two development rooms with compatible public prefixes.

The collector selects contact-ready pushes on the public task object in each room.
Room-local target names and the original command duration do not describe the new
candidate. Robot identities, direction, speed, peer command, and the nine public
features do. Equality of these coarse observations does not establish equality
of the observation histories or of the hidden states.
"""

from decimal import Decimal, ROUND_FLOOR
from math import isfinite

from .option_effect_model import MEASUREMENTS, PUBLIC_FLAGS, PUBLIC_SCALES, _features


SCHEMA = "public-push-particle-cell-v1"
_DONOR_FIELDS = {"room", "anchor_id", "public_history", "features", "original_action"}
_ACTIONS = {"target_id", "a0", "a1", "ticks"}
_ENDPOINT_FLAGS = ("object_static", "endpoint_followup_available")
_ENDPOINT_DETAILS = ("endpoint_terminal", "push_reason", "progress_reached")


def _cell(features):
    values = _features(features)
    # Decimal boundaries make an input such as 0.30 fall in cell 3 at width 0.10.
    # Floor, including at negative values, defines half-open cells [k*w, (k+1)*w).
    bins = tuple(
        int((Decimal(str(value)) / Decimal(str(width))).to_integral_value(rounding=ROUND_FLOOR))
        for value, width in zip(values, PUBLIC_SCALES.values())
    )
    return bins + values[-len(PUBLIC_FLAGS):]


def _joint_codes(action):
    if set(action) != _ACTIONS or not isinstance(action["target_id"], str) or not action["target_id"]:
        raise ValueError("expected a target-bearing joint command")
    if any(type(action[key]) is not int for key in ("a0", "a1", "ticks")):
        raise ValueError("expected integer robot codes and command duration")
    if min(action["a0"], action["a1"]) < 0 or not 1 <= action["ticks"] <= 40:
        raise ValueError("invalid robot code or command duration")
    return action["a0"], action["a1"]


def _sample(match, result):
    if "followup_completed" in result or result["followup_ticks"] != 0:
        raise ValueError("particle simulation must stop at the candidate endpoint")
    prefix_ticks, candidate_ticks = result["prefix_ticks"], result["push_ticks"]
    if type(prefix_ticks) is not int or prefix_ticks < 0:
        raise ValueError("expected measured prefix rebuild ticks")
    if type(candidate_ticks) is not int or not 1 <= candidate_ticks <= 40:
        raise ValueError("expected measured candidate ticks within the 40-tick cap")
    outcome = {}
    for field in MEASUREMENTS:
        value = result[field]
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"expected finite {field}")
        outcome[field] = value
    for field in _ENDPOINT_FLAGS:
        value = result[field]
        if type(value) not in (bool, int) or value not in (0, 1):
            raise ValueError(f"expected observed binary {field}")
        outcome[field] = bool(value)
    outcome.update({key: result[key] for key in _ENDPOINT_DETAILS if key in result})
    return dict(**match, weight=0.5, prefix_ticks=prefix_ticks, candidate_ticks=candidate_ticks,
                followup_ticks=0, outcome=outcome)


class PublicOptionParticles:
    """An index over the public anchor descriptions of a fixed four-room donor pool.

    Rooms without eligible anchors contribute no rows. Supply the pool separately
    from the rooms to be queried. No evaluation-room identifier enters probe().
    """

    def __init__(self, donors):
        index = {}
        anchors = set()
        rooms = set()
        for donor in donors:
            if set(donor) != _DONOR_FIELDS:
                raise ValueError("expected public donor anchor fields only")
            room, anchor = donor["room"], donor["anchor_id"]
            if not isinstance(room, str) or not room or anchor not in ("early", "late"):
                raise ValueError("invalid development room or anchor identity")
            if (room, anchor) in anchors:
                raise ValueError("duplicate room and anchor")
            history = donor["public_history"]
            if not isinstance(history, str) or not history:
                raise ValueError("expected public history digest")
            rooms.add(room)
            anchors.add((room, anchor))
            key = (_joint_codes(donor["original_action"]), _cell(donor["features"]))
            index.setdefault(key, []).append((room, anchor, history))
        if len(rooms) > 4:
            raise ValueError("the frozen donor pool has only four rooms")
        self._index = {key: sorted(rows) for key, rows in index.items()}

    def probe(self, *, original_action, features, progress_m, public_history, simulate):
        """Call simulate(room, anchor_id, progress_m) once per compatible donor room.

        Choose by room ID then early/late, irrespective of the exact history
        digest. With fewer than two distinct rooms the entire request abstains.
        """
        if progress_m not in (0.10, 0.20):
            raise ValueError("requested PUSH progress must be 0.10 m or 0.20 m")
        if not isinstance(public_history, str) or not public_history:
            raise ValueError("expected public history digest")
        codes, cell = _joint_codes(original_action), _cell(features)
        seen = set()
        matches = []
        for room, anchor, history in self._index.get((codes, cell), ()):
            if room in seen:
                continue
            seen.add(room)
            if len(matches) < 2:
                matches.append(dict(room=room, anchor_id=anchor,
                                    exact_public_history_match=history == public_history))
        samples = []
        if len(matches) == 2:
            for match in matches:
                result = simulate(match["room"], match["anchor_id"], progress_m)
                samples.append(_sample(match, result))
        return dict(schema=SCHEMA, status="simulated" if samples else "insufficient",
                    compatibility="public_cell_and_joint_codes", cell=list(cell), joint_codes=list(codes),
                    compatible_room_count=len(seen), matches=matches, samples=samples,
                    calls=len(samples), prefix_ticks=sum(s["prefix_ticks"] for s in samples),
                    candidate_ticks=sum(s["candidate_ticks"] for s in samples), followup_ticks=0)
