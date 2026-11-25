"""Development revision: align one pusher, keep the observer ahead and aside.

All priorities are authored semantic expressions. Process prediction may reject
any proposal, after which the same ranked list supplies preparation alternatives.
"""


def decide(z):
    candidates = []
    if z["seen"]:
        for d in range(8):
            if z[f"align_{d}"] > 0.7:
                candidates.append((f"push_r0_{d}_4", 100 + z[f"align_{d}"]))
            if z["r0_normal_position"] < Q(-0.18, "m"):
                if z["r0_tangent_position"] < Q(-0.06, "m"):
                    candidates.append((f"move_0_{d}", 120 + 5*z[f"tangent_align_{d}"]))
                elif z["r0_tangent_position"] > Q(0.06, "m"):
                    candidates.append((f"move_0_{d}", 120 - 5*z[f"tangent_align_{d}"]))
            if z["r1_tangent_position"] < Q(0.85, "m"):
                candidates.append((f"move_1_{d}", 80 + 5*z[f"tangent_align_{d}"]))
            elif z["r1_normal_position"] < Q(0.85, "m"):
                candidates.append((f"move_1_{d}", 80 + 5*z[f"normal_align_{d}"]))
            if z["r0_normal_position"] < Q(-0.35, "m"):
                candidates.append((f"move_0_{d}", 30 + 5*z[f"normal_align_{d}"]))
        candidates.append(("orbit_0_ccw", -20))
        candidates.append(("orbit_0_cw", -21))
    candidates.append(("brake", -100))
    return candidates
