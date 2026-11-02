"""Offline D007 analysis from the 36 raw runs; no physics."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/agent_exchange/D007_20261030_002"


def load():
    s = json.loads((OUT / "summary.json").read_text())
    raw = {}
    for p in sorted((OUT / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        raw[d["id"]] = d
    return s, raw


def p4after(t):
    return (t.get("p4") or {}).get("after") or {}


def eff(t, side):
    return (((t.get("p3") or {}).get(str(side)) or {}).get("info") or {}).get("effect") or {}


def p0obs(t, side):
    return (((t.get("p0") or {}).get(str(side)) or {}).get("obs") or {})


def p0flags(t, side):
    return (((t.get("p0") or {}).get(str(side)) or {}).get("flags") or {})


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def basic(run):
    ticks = run["result"]["ticks"]
    o0 = p4after(ticks[0]).get("object"); o1 = p4after(ticks[-1]).get("object")
    angles = [wrap(p4after(t)["object"][2]) for t in ticks if p4after(t).get("object")]
    th0 = angles[0] if angles else 0.0
    maxang = max((abs(wrap(a - th0)) for a in angles), default=0.0)
    omega_int = sum(abs(p4after(t)["object"][5]) * 0.05 for t in ticks if p4after(t).get("object"))
    both = sum(1 for t in ticks if p0obs(t,0).get("contacting_object") and p0obs(t,1).get("contacting_object"))
    ready = sum(1 for t in ticks if p0flags(t,0).get("ready") and p0flags(t,1).get("ready"))
    # action direction sum, from p3 effect direction of both sides
    dirs = []
    for side in (0,1):
        e = eff(ticks[min(1,len(ticks)-1)], side)
        if e.get("direction"):
            dirs.append(e["direction"])
    if dirs:
        dx = sum(d[0] for d in dirs); dy = sum(d[1] for d in dirs); n = math.hypot(dx,dy)
        direction = (dx/n, dy/n) if n > 1e-9 else (0.0,0.0)
    else:
        direction = (0.0,0.0)
    disp = math.hypot(o1[0]-o0[0], o1[1]-o0[1]) if o0 and o1 else 0.0
    along = ((o1[0]-o0[0])*direction[0] + (o1[1]-o0[1])*direction[1]) if o1 else 0.0
    return dict(periods=len(ticks), success=bool(run["result"]["success"]),
                termination=run["result"].get("termination"),
                object_disp=round(disp,6), progress_along=round(along,6),
                angle_end=round(angles[-1],6) if angles else None,
                angle_max_abs=round(maxang,6), integral_abs_omega=round(omega_int,6),
                both_contact_ticks=both, near_ready_ticks=ready,
                goal_distance_end=round(p4after(ticks[-1]).get("true_goal_distance", 0.0),6))


def traces(run):
    ticks = run["result"]["ticks"]
    out = []
    for t in ticks:
        a = p4after(t)
        if not a: continue
        o = a["object"]; rs = a["robots"]
        e0, e1 = eff(t,0), eff(t,1)
        out.append(dict(t=float(t["t_start"]), obj=o, robots=rs,
                        c=[bool(p0obs(t,0).get("contacting_object")), bool(p0obs(t,1).get("contacting_object"))],
                        e=[e0,e1], ready=[bool(p0flags(t,0).get("ready")), bool(p0flags(t,1).get("ready"))],
                        p4=t.get("p4") or {}))
    return out


def body_t_at(tr, side, tick):
    e = tr[tick]["e"][side]
    n = e.get("n")
    if not n: return (0.0,0.0)
    return (-n[1], n[0])


def project(tr, idx, side, t):
    a = tr[idx]; o = a["robots"][side]
    return o[0]*t[0] + o[1]*t[1]


def run_metrics(run):
    tr = traces(run)
    return tr


def probe_metrics(probe, baseline, trp, trb):
    side = probe["side"]; off = probe["offset"]
    # choose face at 2.0 s (tick 40) from probe; fall back to first target
    idx = next((i for i,a in enumerate(trp) if a["t"] >= 2.0), 0)
    t = body_t_at(trp, side, idx)
    if abs(t[0])+abs(t[1]) < 1e-9: t = (0.0,1.0)
    # baseline same profile: target/robot relative motion
    def mean_proj(tr, arr, side, t, t0=2.8, t1=3.0):
        vals=[]
        for i,a in enumerate(tr):
            if t0 <= a["t"] < t1:
                vals.append(a["robots"][side][0]*t[0] + a["robots"][side][1]*t[1])
        return sum(vals)/len(vals) if vals else None
    def target_proj(tr, side, t, t0=2.8, t1=3.0):
        vals=[]
        for i,a in enumerate(tr):
            if t0 <= a["t"] < t1:
                e=a["e"][side]; tg=e.get("target_final") or e.get("contact_target")
                if tg: vals.append(tg[0]*t[0]+tg[1]*t[1])
        return sum(vals)/len(vals) if vals else None
    rb = mean_proj(trb, trb, side, t); rp = mean_proj(trp, trp, side, t)
    tb = target_proj(trb, side, t); tp = target_proj(trp, side, t)
    dq_r = None if rb is None or rp is None else rp-rb
    dq_t = None if tb is None or tp is None else tp-tb
    g_r = None if dq_r is None else (1 if off>=0 else -1)*dq_r/abs(off)
    g_t = None if dq_t is None else (1 if off>=0 else -1)*dq_t/abs(off)
    # e_t RMS 2-3
    et=[]
    for a in trp:
        if 2.0 <= a["t"] < 3.0:
            v=a["e"][side].get("e_t")
            if v is not None: et.append(v*v)
    rms = math.sqrt(sum(et)/len(et)) if et else None
    # requested target-delta support (.02 m cumulative >=.5s)
    support=0
    for i,a in enumerate(trp):
        if 2.0 <= a["t"] < 3.0:
            eb=a["e"][side]; tg=eb.get("target_final") or eb.get("contact_target")
            if not tg: continue
            # compare by time-index against baseline target, same side
            j=min(i,len(trb)-1); tg0=trb[j]["e"][side].get("target_final") or trb[j]["e"][side].get("contact_target")
            if tg0:
                d=(tg[0]-tg0[0])*t[0]+(tg[1]-tg0[1])*t[1]
                if abs(d) >= .02: support += 1
    contact=sum(1 for a in trp if a["c"][side])
    return dict(side=side, offset=off, t=t, dq_robot=dq_r, dq_target=dq_t,
                G_robot=g_r, G_target=g_t, rms_et=rms,
                target_support_ticks=support, contact_ticks=contact,
                valid=bool((dq_t is not None and abs(dq_t)>=.02 and support*0.05>=.5 and contact*0.05>=.5)))


def _numeric_diff(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        n = min(len(a), len(b)); m = 0.0
        for x, y in zip(a[:n], b[:n]):
            m = max(m, _numeric_diff(x, y))
        return m
    try:
        return abs(float(a) - float(b))
    except Exception:
        return 0.0


def maxdiff(a,b,key,get,max_t=None):
    n=min(len(a),len(b)); m=0.0
    for i in range(n):
        if max_t is not None and a[i].get("t", 0.0) >= max_t: break
        va=get(a[i],key); vb=get(b[i],key)
        if va is None or vb is None: continue
        m=max(m,_numeric_diff(va,vb))
    return m


def main():
    s, raw = load()
    by = {(d["case"],d["profile"],d["tag"]):d for d in raw.values()}
    base = {(d["case"],d["profile"]):d for d in raw.values() if d["tag"]=="baseline"}
    probes = {}
    for d in raw.values():
        if d["tag"]=="probe":
            meta=d["meta"]; probes[(d["case"],d["profile"],meta["side"],meta["offset"])] = d
    # basic rows
    rows=[]
    for d in raw.values():
        if d["tag"] == "nonint": continue
        b=basic(d); rows.append(dict(rid=d["id"],tag=d["tag"],case=d["case"],profile=d["profile"],**b))
    # probe metrics / prefix
    pm=[]; pairs={}
    for key,d in probes.items():
        case,profile,side,off=key; b=base[(case,profile)]
        trp=traces(d); trb=traces(b)
        m=probe_metrics(dict(side=side,offset=off), b, trp, trb)
        m.update(rid=d["id"], case=case, profile=profile,
                 prefix_pose=maxdiff(trp,trb,"obj",lambda a,k:a[k], max_t=2.0),
                 prefix_robots=maxdiff(trp,trb,"robots",lambda a,k:a[k], max_t=2.0))
        pm.append(m); pairs[key]=m
    # C gate
    common=[]
    for (case,profile,side,off),m in pairs.items():
        if profile!="C": continue
        b=pairs.get((case,"B",side,off))
        if m["valid"] and b and b["valid"]:
            common.append((case,side,off,b,m))
    def med(vals):
        vals=[v for v in vals if v is not None]
        return statistics.median(vals) if vals else None
    gate=dict(common_valid=len(common),
              median_G_B=med([x[3]["G_robot"] for x in common]),
              median_G_C=med([x[4]["G_robot"] for x in common]),
              median_rms_B=med([x[3]["rms_et"] for x in common]),
              median_rms_C=med([x[4]["rms_et"] for x in common]))
    # baseline A/C loss
    base_loss=[]
    for case in ("adjacent","adjacent_mirror","same_face","same_face_closer"):
        a=basic(base[(case,"A")]); c=basic(base[(case,"C")])
        base_loss.append(dict(case=case, progress_loss=round(a["progress_along"]-c["progress_along"],6),
                              angle_increase=round(c["angle_max_abs"]-a["angle_max_abs"],6),
                              contact_ratio=round(c["both_contact_ticks"]/max(1,a["both_contact_ticks"]),4)))
    # witness
    wit=[]
    for d in raw.values():
        if d["tag"]=="witness":
            b=basic(d); wit.append(dict(case=d["case"],profile=d["profile"],periods=b["periods"],success=b["success"]))
    # noninterference vs baseline
    ni=[]
    for d in raw.values():
        if d["tag"]!="nonint": continue
        b=base[("adjacent",d["profile"])]
        trb=traces(b); trn=[]
        # nonint has trace without p4 structure; compare object/robots lists
        for x in d["result"].get("trace",[]):
            trn.append(dict(t=x["t"], obj=x["object"], robots=x["robots"], e=[{},{}], c=[False,False], ready=[False,False], p4={}))
        # trace is sampled every decision (0.2 s); compare to the matching
        # control tick in the instrumented baseline, not by trace index.
        m_obj = 0.0; m_rob = 0.0
        for x in trn:
            j = int(round(x["t"] / 0.05)) + 3
            if j < 0 or j >= len(trb): continue
            m_obj = max(m_obj, _numeric_diff(x["obj"], trb[j]["obj"]))
            m_rob = max(m_rob, _numeric_diff(x["robots"], trb[j]["robots"]))
        ni.append(dict(profile=d["profile"], sampled="0.2s",
                       pose_err=m_obj, robot_err=m_rob))
    out=dict(rows=rows, probes=pm, gate=gate, baseline_loss=base_loss,
             witnesses=wit, nonint=ni, source="D007_20261030_001")
    (OUT/"analysis.json").write_text(json.dumps(out,indent=1))
    with (OUT/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT/"probes.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(pm[0].keys())); w.writeheader(); w.writerows(pm)
    print(json.dumps(dict(gate=gate, baseline_loss=base_loss, witnesses=wit, nonint=ni),indent=1))
    print('mean G B', statistics.mean([x["G_robot"] for x in pm if x["profile"]=="B" and x["G_robot"] is not None]) if any(x["profile"]=="B" and x["G_robot"] is not None for x in pm) else None)
    print('mean G C', statistics.mean([x["G_robot"] for x in pm if x["profile"]=="C" and x["G_robot"] is not None]) if any(x["profile"]=="C" and x["G_robot"] is not None for x in pm) else None)


if __name__ == "__main__":
    main()
