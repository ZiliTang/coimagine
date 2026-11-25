"""First-round authored candidate program; prediction veto disabled in development.

Public missing-area differences distinguish which robot still needs a view.
Short commands preserve frequent sensing and reassessment opportunities.
"""


def decide(z):
    candidates = []
    if z["seen"]:
        for d in range(8):
            if z[f"align_{d}"] > 0.7:
                candidates.append((f"push_both_{d}_4", 100 + z[f"align_{d}"]))
                candidates.append((f"push_r0_{d}_4", 95 + z[f"align_{d}"]))
                candidates.append((f"push_r1_{d}_4", 94 + z[f"align_{d}"]))
                if z[f"missing:push_r0_{d}_4"] > z[f"missing:push_r1_{d}_4"] + 0.003:
                    for e in range(8):
                        candidates.append((f"move_0_{e}", 35 + 5 * z[f"normal_align_{e}"]))
        for d in range(8):
            if z["r1_tangent_position"] < Q(0.85, "m"):
                candidates.append((f"move_1_{d}", 20 + 5 * z[f"tangent_align_{d}"]))
            elif z["r1_normal_position"] < Q(0.55, "m"):
                candidates.append((f"move_1_{d}", 20 + 5 * z[f"normal_align_{d}"]))
            else:
                candidates.append((f"move_1_{d}", 2 + z[f"align_{d}"]))
        candidates.append(("orbit_0_ccw", -2))
        candidates.append(("orbit_0_cw", -3))
    else:
        for d in range(8):
            candidates.append((f"move_1_{d}", z[f"align_{d}"]))
    candidates.append(("brake", -100))
    return candidates
