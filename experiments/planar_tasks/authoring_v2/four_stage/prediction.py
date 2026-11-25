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


def mode(context, profile="contact_v1"):
    z = context["symbols"]
    flags = tuple(bool(z[k]) for k in ("seen", "contact0", "contact1", "target_stationary"))
    if profile == "contact_and_public_stage_v2":
        return flags+tuple(r.get("execution",{}).get("stage") or "none"
                           for r in context["observation"]["robots"])
    if profile != "contact_v1":
        raise ValueError("unknown public mode profile")
    return flags


def compact_row(row, host, mode_profile="contact_v1"):
    """Keep public-derived features and labels; raw provenance stays in its artifact."""
    context_mode = row["context_mode"] if "context_mode" in row else list(mode(row["public_context"],mode_profile))
    if len(context_mode) != (6 if mode_profile=="contact_and_public_stage_v2" else 4):
        raise ValueError("recompute public modes from original data before changing profiles")
    return dict(room=row["room"], action=row["action"],
                feature=(row["feature"] if "feature" in row else feature(row["public_context"]).tolist()),
                context_mode=context_mode,
                semantic={k: row["semantic"][k] for k in host["metrics"]},
                next_semantic={k: row["next_semantic"][k] for k in host["metrics"]},
                process=deepcopy(row["process"]), source=row.get("artifact", row.get("source")))


def scale(name, spec):
    if name.startswith("missing:"):
        return .1  # area normalized by one square metre
    return {"m": .5, "m/s": .2, "s": 2., "1": .2}[spec["unit"]]


class JointModel:
    def __init__(self, packet):
        self.packet = deepcopy(packet)
        self.host = deepcopy(packet.get("schema",schema()))
        if packet["schema_sha256"] != digest(self.host):
            raise ValueError("predictor semantic schema changed")
        self.groups = defaultdict(list)
        for row in packet["rows"]:
            self.groups[row["action"]].append(row)
        self.matrices = {a: np.asarray([r["feature"] if "feature" in r else feature(r["public_context"]) for r in rows])
                         for a, rows in self.groups.items()}
        self.modes = {a: [tuple(r["context_mode"]) if "context_mode" in r else mode(r["public_context"]) for r in rows]
                      for a, rows in self.groups.items()}
        self.sha256 = digest(packet)

    def neighbors(self, context, action):
        return self._neighbors(feature(context), mode(context,self.packet.get("mode_profile","contact_v1")), action)

    def _neighbors(self, query_feature, query_mode, action, exclude_room=None):
        rows = self.groups.get(action, [])
        if not rows:
            return []
        delta = self.matrices[action] - query_feature
        critical = list(range(7))
        if self.packet.get("distance_profile") == "pose_and_both_robot_proprioception_v2":
            # Each robot contributes six proprioception entries, then 72
            # ranges and four 72-category indicators. Do not let those rays
            # dilute a different moving robot or a different controller ramp.
            critical += list(range(7,13)) + list(range(373,379))
        distances = np.maximum(np.sqrt(np.mean(delta * delta, axis=1)),
                               np.max(np.abs(delta[:, critical]), axis=1))
        query_mode = tuple(query_mode)
        by_room = {}
        for i, row in enumerate(rows):
            if row["room"] == exclude_room:
                continue
            d = float(distances[i])
            if self.modes[action][i] != query_mode or d > self.packet["max_distance"]:
                continue
            if row["room"] not in by_room or d < by_room[row["room"]][0]:
                by_room[row["room"]] = (d, row)
        return sorted(by_room.values(), key=lambda p: (p[0], p[1]["room"]))[:self.packet["neighbors"]]

    def points(self, context, action):
        return self._points(context["symbols"], self.neighbors(context, action))

    def _points(self, current, neighbors):
        points = []
        for distance, row in neighbors:
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

    def residuals(self, row, points):
        residuals = {}
        for key, spec in self.host["metrics"].items():
            if spec["unit"] == "bool":
                continue
            values = [p["values"][key] for p in points]
            actual = row["next_semantic"][key]
            radius=self.semantic_radius(points,key)
            residuals[key] = max(0., min(values)-radius-actual, actual-max(values)-radius)
        actual = flatten_motion(row["process"])
        predicted = [flatten_motion(p["motion"]) for p in points]
        for key in MOTION_NAMES:
            values = [m[key] for m in predicted]
            # These channels are whole-path bounds, not future point values.
            # A shorter enclosed path is covered evidence, not a miss that
            # should further enlarge an already conservative envelope.
            lower_bound = key.startswith("r") and key.rsplit("_",1)[1] in ("0", "2")
            residuals["motion:"+key] = (max(0., min(values)-actual[key]) if lower_bound else
                                       max(0., actual[key]-max(values)))
        return residuals

    def semantic_radius(self, points, name):
        # A query between recorded boundaries must not inherit an artificially
        # zero-width semantic prediction merely because several rooms share
        # the same nearest prefix. This is an empirical model choice, included
        # in calibration and independently checked on actual admitted endpoints.
        if not self.packet.get("semantic_distance_expansion"):
            return 0.
        return max((p["distance"] for p in points),default=0.)*scale(name,self.host["metrics"][name])

    def error_scale(self, action, name):
        default = .01 if name.startswith("motion:") else scale(name, self.host["metrics"][name])
        return self.packet.get("error_scales", {}).get(action, {}).get(name, default)

    def calibration_key(self, action, context_mode):
        if self.packet.get("calibration_scope", "action") == "action_mode":
            key = action+"|"+"".join("1" if v else "0" for v in context_mode[:4])
            if len(context_mode)>4:
                key += "|"+"|".join(context_mode[4:])
            return key
        return action

    @classmethod
    def fit(cls, rows, calibration=(), *, version=0, parent=None, alpha=.10, host=None,
            normalization="fixed", calibration_scope="action",mode_profile="contact_v1",
            semantic_distance_expansion=False):
        training = sorted({r["room"] for r in rows})
        cal_rooms = sorted({r["room"] for r in calibration})
        if set(training) & set(cal_rooms):
            raise ValueError("fitting and calibration rooms overlap")
        if not 0 < alpha < 1:
            raise ValueError("risk level must be between zero and one")
        if normalization not in ("fixed", "fit_residual_q90"):
            raise ValueError("unknown residual normalization")
        if calibration_scope not in ("action", "action_mode"):
            raise ValueError("unknown calibration partition")
        host=deepcopy(host if host is not None else schema())
        clean = [compact_row(r, host,mode_profile) for r in rows]
        packet = dict(version=version, parent=parent, family="raw-public-joint-room-knn-v2",
                      schema=host,schema_sha256=digest(host), alpha=alpha, max_distance=.6,
                      min_rooms=4, neighbors=8, rows=clean, calibration={},
                      training_rooms=training, calibration_rooms=cal_rooms,
                      normalization=normalization, motion_horizon="command_then_brake40",
                      calibration_scope=calibration_scope,
                      mode_profile=mode_profile,
                      distance_profile="pose_and_both_robot_proprioception_v2",
                      semantic_distance_expansion=bool(semantic_distance_expansion),
                      guarantee="empirical; verify joint coverage on independent admission subset")
        model = cls(packet)
        if normalization == "fit_residual_q90":
            # Estimate channel units using FIT rooms only, leaving the queried
            # room out. One room-max joint calibration score still covers all
            # channels, rather than spending alpha separately on every channel.
            samples = defaultdict(lambda: defaultdict(list))
            for row in clean:
                points = model._points(row["semantic"], model._neighbors(
                    np.asarray(row["feature"]), row["context_mode"], row["action"], row["room"]))
                if len(points) >= packet["min_rooms"]:
                    for key, residual in model.residuals(row, points).items():
                        samples[row["action"]][key].append(residual)
            packet["error_scales"] = {}
            for action, columns in samples.items():
                packet["error_scales"][action] = {
                    key: max(.001 if key.startswith("motion:") else
                             .1*scale(key, host["metrics"][key]), float(np.quantile(values, .90)))
                    for key, values in columns.items()}
            model = cls(packet)
        per_room = defaultdict(dict)
        boolean_misses = defaultdict(set)
        for row in calibration:
            row = compact_row(row, host,mode_profile)
            points = model._points(row["semantic"], model._neighbors(
                np.asarray(row["feature"]), row["context_mode"], row["action"]))
            if len(points) < packet["min_rooms"]:
                continue
            calibration_key = model.calibration_key(row["action"], row["context_mode"])
            residual = max((value/model.error_scale(row["action"], key)
                            for key, value in model.residuals(row, points).items()), default=0.)
            for key, spec in model.host["metrics"].items():
                actual = row["next_semantic"][key]
                values = [p["values"][key] for p in points]
                if spec["unit"] == "bool":
                    if actual not in values:
                        boolean_misses[calibration_key].add(key)
            scores = per_room[calibration_key]
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
        cal = self.packet["calibration"].get(self.calibration_key(action,
            mode(public_context,self.packet.get("mode_profile","contact_v1"))), {})
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
                width = pad * self.error_scale(action, key)+self.semantic_radius(points,key)
                lo, hi = spec["domain"]
                numeric[key] = [max(lo, min(values)-width), min(hi, max(values)+width)]
            motions = [flatten_motion(p["motion"]) for p in points]
            motion = {}
            for i in (0, 1):
                motion[f"r{i}"] = [
                    min(0., min(m[f"r{i}_0"] for m in motions)-pad*self.error_scale(action,f"motion:r{i}_0")),
                    max(0., max(m[f"r{i}_1"] for m in motions)+pad*self.error_scale(action,f"motion:r{i}_1")),
                    min(0., min(m[f"r{i}_2"] for m in motions)-pad*self.error_scale(action,f"motion:r{i}_2")),
                    max(0., max(m[f"r{i}_3"] for m in motions)+pad*self.error_scale(action,f"motion:r{i}_3"))]
            motion["target_boundary"] = max(m["target_boundary"] for m in motions)+pad*self.error_scale(action,"motion:target_boundary")
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
                    motion_horizon=self.packet.get("motion_horizon"),
                    calibration=cal, assurance=self.packet["guarantee"])
