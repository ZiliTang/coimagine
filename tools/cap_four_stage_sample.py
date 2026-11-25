"""Bounded offline fork collection on declared fit/calibration rooms only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
import traceback

from tools.cap_symbolic_general import ROOT,read,save,sha
from experiments.planar_tasks.authoring_v2.four_stage.development_forks import collect_room
from experiments.planar_tasks.authoring_v2.four_stage.native_geometry import build


BASE=ROOT/"outputs/semantic_rl/cap_four_stage_v1"


def worker(job):
    try:
        return collect_room(job["fixture"],job["room"],Path(job["out"]),curriculum=job["curriculum"],
                            **job.get("sampling",{}))
    except BaseException:
        error=dict(room=job["room"],curriculum=job["curriculum"],status="error",traceback=traceback.format_exc())
        save(Path(job["out"])/"worker_error.json",error)
        return error


def collect(name,fit_start,fit_count,cal_count,profile="warmup"):
    out=BASE/"data"/name
    if out.exists():raise FileExistsError(out)
    build()
    panels=read(BASE/"rooms.json")
    selected={"fit":list(panels["rooms"]["fit"].items())[fit_start:fit_start+fit_count],
              "calibration":list(panels["rooms"]["calibration"].items())[:cal_count]}
    files=set(ROOT.glob("experiments/**/*.py"))|set(ROOT.glob("src/coimagine/**/*.py"))|set(ROOT.glob("tools/**/*.py"))
    files|={p for p in (ROOT/"pyproject.toml",ROOT/"uv.lock") if p.exists()}
    files|={ROOT/"src/coimagine/random_start/lidar_kernel.c",
            ROOT/"experiments/planar_tasks/authoring_v2/four_stage/geometry_kernel.cpp",Path(__file__)}
    hashes={}
    for path in sorted(files):
        relative=str(path.relative_to(ROOT));dest=out/"sources"/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[relative]=sha(path)
    save(out/"sources.json",hashes)
    curricula=["joint","leader"] if profile=="warmup" else ["leader"]
    sampling={} if profile=="warmup" else dict(
        prefixes=[0,2,4,6,8,10,12,16,20,24,32,40,48,64,80,96],
        actions=[f"push_r0_{d}_4" for d in (0,1,2)]+
                [f"move_{i}_{d}" for i in (0,1) for d in (1,2,3)]+["orbit_0_ccw","brake"])
    save(out/"protocol.json",dict(fit_start=fit_start,fit_count=fit_count,cal_count=cal_count,profile=profile,
        rooms={s:[k for k,_ in rows] for s,rows in selected.items()},curricula=curricula,
        room_identity="Both curricula retain the SAME physical room ID; never count them as independent calibration rooms.",
        actions=sampling.get("actions","nine short PUSH plus four preparation/stop controls; public startup check only"),
        prefixes=sampling.get("prefixes",[0,4,12,24,48,96,160,240,320]),brake_ticks=40,brake_chunk=40,
        freeze_sha=sha(out/"sources.json"),scope="offline effect data; no confirmation rooms; neither a safe deployed policy nor an oracle"))
    jobs=[]
    for split,rows in selected.items():
        for room,fixture in rows:
            for curriculum in curricula:
                jobs.append(dict(room=room,fixture=fixture,curriculum=curriculum,
                                 sampling=sampling,out=str(out/split/room/curriculum)))
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            result=future.result();results.append(result)
            save(out/"progress.json",dict(completed=len(results),total=len(jobs),rows=results,
                                          wall_seconds=time.perf_counter()-started))
            print(result,flush=True)
    save(out/"complete.json",dict(rows=results,wall_seconds=time.perf_counter()-started,
                                  errors=sum(r.get("status")=="error" or r.get("errors",0)>0 for r in results)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",required=True)
    p.add_argument("--fit-start",type=int,default=0)
    p.add_argument("--fit-count",type=int,default=16)
    p.add_argument("--cal-count",type=int,default=16)
    p.add_argument("--profile",choices=["warmup","dense_transitions"],default="warmup")
    a=p.parse_args();collect(a.name,a.fit_start,a.fit_count,a.cal_count,a.profile)
