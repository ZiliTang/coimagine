"""Independent containment checks for realized one-step prediction targets."""


def motion_contains(bound, actual, tolerance=1e-7):
    for robot in ("r0", "r1"):
        for j in (0,2):
            if actual[robot][j] < bound[robot][j]-tolerance:
                return False
        for j in (1,3):
            if actual[robot][j] > bound[robot][j]+tolerance:
                return False
    return actual["target_boundary"] <= bound["target_boundary"]+tolerance


def semantic_contains(updates, actual, tolerance=1e-7):
    for key, interval in updates.items():
        if interval is None:
            continue
        if key not in actual:
            return False
        if type(interval) is bool:
            if actual[key] is not interval:
                return False
        elif not interval[0]-tolerance <= actual[key] <= interval[1]+tolerance:
            return False
    return True


def realized_coverage(forecast, motion, next_semantic=None):
    branches = forecast.get("branches", [])
    covered_motion = [motion_contains(b["motion"],motion) for b in branches]
    joint = None if next_semantic is None else any(
        covered and semantic_contains(b["updates"],next_semantic)
        for b,covered in zip(branches,covered_motion))
    return dict(root_motion_covered=any(covered_motion), joint_endpoint_covered=joint,
                model_id=forecast.get("model_id"),
                scope="actual command path and next public semantics; unexecuted braking tail not measured")
