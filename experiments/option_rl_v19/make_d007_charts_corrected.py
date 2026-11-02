"""Corrected D007 chain charts: object frame + fixed body tangent at 1.95 s."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/agent_exchange/D007_20261030_002"
CH = OUT / "chains"; CH.mkdir(exist_ok=True)


def load(rid):
    return json.loads((OUT / "runs" / f"{rid}.json").read_text())


def eff(t, side):
    return t["p3"][str(side)]["info"]["effect"]


def body(point, obj):
    dx, dy = point[0] - obj[0], point[1] - obj[1]
    c, s = math.cos(obj[2]), math.sin(obj[2])
    return c*dx + s*dy, -s*dx + c*dy


def axis_at(run, side, tick=39):
    t = run["result"]["ticks"][tick]
    n = eff(t, side)["face_normal"]
    assert n is not None
    obj = t["p4"]["before"]["object"]
    c, s = math.cos(obj[2]), math.sin(obj[2])
    tx, ty = -n[1], n[0]
    return (c*tx + s*ty, -s*tx + c*ty)


def series(run, baseline, side):
    ax = axis_at(baseline, side)
    ts, tgt, rob, et, lat, ang = [], [], [], [], [], []
    for t in run["result"]["ticks"]:
        b = t.get("p4", {}).get("before")
        if not b: continue
        obj = b["object"]
        e = eff(t, side)
        tg = e.get("target_final")
        if tg is None: continue
        ts.append(t["t_start"])
        tgt.append(sum(x*y for x, y in zip(body(tg, obj), ax)))
        rob.append(sum(x*y for x, y in zip(body(b["robots"][side], obj), ax)))
        et.append(e.get("e_t", math.nan)); lat.append(e.get("lateral", math.nan))
        ang.append(obj[2])
    return ts, tgt, rob, et, lat, ang


def main():
    for rid in ("013_probe", "014_probe", "015_probe", "016_probe", "021_probe", "022_probe"):
        d = load(rid)
        case, prof = d["case"], d["profile"]
        b = load(next(p.stem for p in (OUT / "runs").glob("*.json")
                      if (lambda x: x["tag"] == "baseline" and x["case"] == case and x["profile"] == prof)(json.loads(p.read_text()))))
        side = d["meta"]["side"]
        ts, tg, rb, et, lat, ang = series(d, b, side)
        fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        ax[0].plot(ts, tg, label="target tangent"); ax[0].plot(ts, rb, label="robot tangent")
        ax[0].set_ylabel("body +t (m)"); ax[0].legend(fontsize=8)
        ax[1].plot(ts, et, label="e_t"); ax[1].plot(ts, lat, label="lateral")
        ax[1].set_ylabel("m / m/s"); ax[1].legend(fontsize=8)
        ax[2].plot(ts, ang, label="object angle"); ax[2].set_ylabel("rad")
        ax[2].legend(fontsize=8); ax[2].set_xlabel("t (s)")
        ax[0].set_title(f"{rid} {case} {prof} side{side} off{d['meta']['offset']:+.2f}")
        fig.tight_layout(); fig.savefig(CH / f"{rid}.png", dpi=120); plt.close(fig)
    print("charts", len(list(CH.glob("*.png"))))


if __name__ == "__main__":
    main()
