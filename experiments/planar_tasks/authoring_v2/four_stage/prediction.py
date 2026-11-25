"""Room-balanced one-step joint predictions from raw public observations.

No learned scoring or privileged feature channel. Empirical calibration is
versioned separately from the author program and is evaluated on admissions.
"""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from ..symbolic_general.core import digest
from .frontend import schema


MOTION_NAMES = [f"r{i}_{j}" for i in (0, 1) for j in range(4)] + ["target_boundary"]


def flatten_motion(motion):
    return {**{f"r{i}_{j}": float(motion[f"r{i}"][j]) for i in (0, 1) for j in range(4)},
            "target_boundary": float(motion["target_boundary"])}


def feature(context):
    o, z = context["observation"], context["symbols"]
    # These leading context coordinates also impose a maximum distance, so
    # hundreds of similar raw-ray entries cannot dilute a large pose mismatch.
    values = [z["gx"] / .75, z["gy"] / .75,
              z["r0_rx"] / .30, z["r0_ry"] / .30,
              z["r1_rx"] / .30, z["r1_ry"] / .30, z["speed"] / .2]
    for robot in o["robots"]:
        own = robot["own"]
        ref = robot.get("execution", {}).get("reference_velocity") or (0., 0.)
        values += [own["vx"] / .2, own["vy"] / .2, ref[0] / .2, ref[1] / .2,
                   math.cos(own["angle"]), math.sin(own["angle"])]
        scan = robot["scan"]
        if len(scan["distances"]) != 72:
            raise ValueError("fixed 72-ray input required")
        values += [float(d) / 6 for d in scan["distances"]]
        for label in (0, 1, 2, 3):
            values += [float(v == label) for v in scan["category"]]
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite public predictor feature")
    return result


def mode(context):
    z = context["symbols"]
    return tuple(bool(z[k]) for k in ("seen", "contact0", "contact1", "target_stationary"))


def scale(name, spec):
    if name.startswith("missing:"):
        return .1  # area normalized by one square metre
    return {"m": .5, "m/s": .2, "s": 2., "1": .2}[spec["unit"]]


class JointModel:
    def __init__(self, packet):
        self.packet = deepcopy(packet)
        self.host = schema()
        if packet["schema_sha256"] != digest(self.host):
            raise ValueError("predictor semantic schema changed")
        self.groups = defaultdict(list)
        for row in packet["rows"]:
            self.groups[row["action"]].append(row)
        self.matrices = {a: np.asarray([feature(r["public_context"]) for r in rows])
                         for a, rows in self.groups.items()}
        self.modes = {a: [mode(r["public_context"]) for r in rows] for a, rows in self.groups.items()}
        self.sha256 = digest(packet)

    def neighbors(self, context, action):
        rows = self.groups.get(action, [])
        if not rows:
            return []
        delta = self.matrices[action] - feature(context)
        distances = np.maximum(np.sqrt(np.mean(delta * delta, axis=1)),
                               np.max(np.abs(delta[:, :7]), axis=1))
        query_mode = mode(context)
        by_room = {}
        for i, row in enumerate(rows):
            d = float(distances[i])
            if self.modes[action][i] != query_mode or d > self.packet["max_distance"]:
                continue
            if row["room"] not in by_room or d < by_room[row["room"]][0]:
                by_room[row["room"]] = (d, row)
        return sorted(by_room.values(), key=lambda p: (p[0], p[1]["room"]))[:self.packet["neighbors"]]

    def points(self, context, action):
        current = context["symbols"]
        points = []
        for distance, row in self.neighbors(context, action):
            future = {}
            for key, spec in self.host["metrics"].items():
                if spec["unit"] == "bool":
                    future[key] = row["next_semantic"][key]
                else:
                    value = current[key] + row["next_semantic"][key] - row["semantic"][key]
                    lo, hi = spec["domain"]
                    future[key] = max(lo, min(hi, value))
            points.append(dict(values=future, motion=deepcopy(row["process"]),
                               distance=distance, room=row["room"]))
        return points

    @classmethod
    def fit(cls, rows, calibration=(), *, version=0, parent=None, alpha=.10):
        training = sorted({r["room"] for r in rows})
        cal_rooms = sorted({r["room"] for r in calibration})
        if set(training) & set(cal_rooms):
            raise ValueError("fitting and calibration rooms overlap")
        if not 0 < alpha < 1:
            raise ValueError("risk level must be between zero and one")
        fields = ("room", "action", "public_context", "semantic", "next_semantic", "process")
        clean = [{k: deepcopy(r[k]) for k in fields} for r in rows]
        packet = dict(version=version, parent=parent, family="raw-public-joint-room-knn-v1",
                      schema_sha256=digest(schema()), alpha=alpha, max_distance=.6,
                      min_rooms=4, neighbors=8, rows=clean, calibration={},
                      training_rooms=training, calibration_rooms=cal_rooms,
                      guarantee="empirical; verify joint coverage on independent admission subset")
        model = cls(packet)
        per_room = defaultdict(dict)
        boolean_misses = defaultdict(set)
        for row in calibration:
            points = model.points(row["public_context"], row["action"])
            if len(points) < packet["min_rooms"]:
                continue
            residual = 0.
            for key, spec in model.host["metrics"].items():
                actual = row["next_semantic"][key]
                values = [p["values"][key] for p in points]
                if spec["unit"] == "bool":
                    if actual not in values:
                        boolean_misses[row["action"]].add(key)
                else:
                    residual = max(residual, max(0., min(values)-actual, actual-max(values)) / scale(key, spec))
            actual_motion = flatten_motion(row["process"])
            predicted_motion = [flatten_motion(p["motion"]) for p in points]
            for key in MOTION_NAMES:
                values = [m[key] for m in predicted_motion]
                actual = actual_motion[key]
                residual = max(residual, max(0., min(values)-actual, actual-max(values)) / .01)
            scores = per_room[row["action"]]
            scores[row["room"]] = max(residual, scores.get(row["room"], 0.))
        for action, scores in per_room.items():
            rank = math.ceil((1-alpha) * (len(scores)+1))
            padding = sorted(scores.values())[rank-1] if rank <= len(scores) else None
            packet["calibration"][action] = dict(rooms=len(scores), rank=rank, padding=padding,
                unknown_boolean_fields=sorted(boolean_misses[action]),
                room_scores=scores)
        return cls(packet)

    def predict(self, public_context, action):
        points = self.points(public_context, action)
        cal = self.packet["calibration"].get(action, {})
        calibrated = cal.get("padding") is not None
        pad = cal.get("padding") or 0.
        unresolved = set(cal.get("unknown_boolean_fields", []))
        branches = []
        # Keep joint categorical outcome modes. Numeric intervals enclose all
        # neighbors, and motion bounds cover every neighbor's process; this
        # conservative product does not select an independently favorable mode.
        if points:
            numeric = {}
            for key, spec in self.host["metrics"].items():
                if spec["unit"] == "bool":
                    continue
                values = [p["values"][key] for p in points]
                width = pad * scale(key, spec)
                lo, hi = spec["domain"]
                numeric[key] = [max(lo, min(values)-width), min(hi, max(values)+width)]
            motions = [flatten_motion(p["motion"]) for p in points]
            motion = {}
            for i in (0, 1):
                motion[f"r{i}"] = [
                    min(0., min(m[f"r{i}_0"] for m in motions)-pad*.01),
                    max(0., max(m[f"r{i}_1"] for m in motions)+pad*.01),
                    min(0., min(m[f"r{i}_2"] for m in motions)-pad*.01),
                    max(0., max(m[f"r{i}_3"] for m in motions)+pad*.01)]
            motion["target_boundary"] = max(m["target_boundary"] for m in motions)+pad*.01
            seen = set()
            for p in points:
                flags = {k: None if k in unresolved else v for k, v in p["values"].items()
                         if self.host["metrics"][k]["unit"] == "bool"}
                identity = digest(flags)
                if identity in seen:
                    continue
                seen.add(identity)
                branches.append(dict(updates={**numeric, **flags}, motion=deepcopy(motion),
                                     label="joint-neighbor-mode"))
        return dict(model_id=self.sha256, supported=len(points) >= self.packet["min_rooms"],
                    calibrated=calibrated, branches=branches, support_rooms=len(points),
                    calibration=cal, assurance=self.packet["guarantee"])
