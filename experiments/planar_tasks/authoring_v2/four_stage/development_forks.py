"""OFFLINE ONLY: exact same-prefix physics forks, never a deployment rollout.

POSIX copy-on-write retains controller and contact-solver state. The parent
executes a public-menu-defined short-push curriculum; forked alternatives are
deliberately measured even when the current empirical certificate is unknown.
No environment/fixture/state handle is exported to the deployed predictor.
"""
from pathlib import Path
import os
import signal
import time
import traceback

from .frontend import OPTIONS
from .native_frontend import NativeFrontend
from .physical import command_and_stop
from ..symbolic_general.simulation import observation
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory
from experiments.planar_tasks.relational_search.contract import digest
from tools.cap_symbolic_general import save,read


def fork_call(function, path, *, timeout_seconds=60):
    """Host-development transaction; all mutations occur in the child process."""
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();pid=os.fork()
    if pid==0:
        try:
            save(path,function())
            os._exit(0)
        except BaseException:
            save(path,dict(status="error",traceback=traceback.format_exc()))
            os._exit(1)
    while True:
        child,status=os.waitpid(pid,os.WNOHANG)
        if child:
            result=read(path) if path.exists() else dict(status="error",error="child returned without artifact")
            result["fork_seconds"]=time.perf_counter()-started
            result["child_exit"]=os.waitstatus_to_exitcode(status)
            return result
        if time.perf_counter()-started>=timeout_seconds:
            os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
            result=dict(status="timeout",reservation_retained=True,fork_seconds=time.perf_counter()-started)
            if not path.exists():save(path,result)
            return result
        time.sleep(.005)


def collect_room(fixture,room,path,*,prefixes=(0,4,12,24,48,96,160,240,320),
                 directions=(0,1,2),include_preparation=True,curriculum="joint",actions=None):
    path=Path(path)
    if (path/"summary.json").exists():
        return read(path/"summary.json")
    if (path/"started.json").exists():
        raise ValueError("interrupted collection requires explicit new version")
    if curriculum not in ("joint","leader"):
        raise ValueError("unknown public exploration curriculum")
    if actions is None:
        actions=[f"push_{role}_{d}_4" for d in directions for role in ("both","r0","r1")]
        if include_preparation:
            actions += ["move_0_1","move_1_3","orbit_0_ccw","brake"]
    else:
        actions=list(actions)
        if len(actions)!=len(set(actions)) or not set(actions)<=set(OPTIONS):
            raise ValueError("explicit sampling actions must be unique host options")
    reserved=4*max(prefixes)+sum(OPTIONS[a]["ticks"]+40 for a in actions)*len(prefixes)
    save(path/"started.json",dict(room=room,curriculum=curriculum,prefixes=list(prefixes),actions=actions,reserved_ticks=reserved,
                                 mode="offline unknown-candidate measurement; not certified deployment"))
    world=Environment(fixture,EventConfig(mode="duration"),fast_feedback=True,
                      record_details=False,history_factory=MemoryHistory)
    frontend=NativeFrontend()  # full host dictionary retained for later author dependencies
    prefix_rows=[];records=[];charged=0;started=time.perf_counter()
    for index in range(max(prefixes)+1):
        if world.env.termination:break
        prepared=frontend.prepare_history(world.history)
        root_hash=digest(prepared["observation"])
        if index in prefixes:
            for action in actions:
                destination=path/"branches"/f"p{index:04d}_{action}.json.gz"
                if not prepared["symbols"]["legal:"+action]:
                    row=dict(status="not_startable",room=room,prefix=index,action=action,charged_ticks=0,public_sha256=root_hash)
                    save(destination,row)
                else:
                    def evaluate():
                        if digest(observation(world))!=root_hash:
                            raise ValueError("fork did not preserve public boundary")
                        measured=command_and_stop(world,prepared["commands"][action],brake_chunk=40,regions=prepared["regions"][action])
                        frontend.record_command(action)
                        endpoint=frontend.prepare(measured["endpoint"])
                        return dict(status="complete",room=room,curriculum=curriculum,prefix=index,action=action,
                            public_sha256=root_hash,public_context=prepared["public_context"],
                            semantic=prepared["symbols"],next_semantic=endpoint["symbols"],
                            process=measured["process"],measured=measured,
                            charged_ticks=world.env.total_periods-prepared["observation"]["tick"])
                    row=fork_call(evaluate,destination)
                    if digest(observation(world))!=root_hash:
                        raise ValueError("offline branch mutated parent")
                charged+=row.get("charged_ticks",OPTIONS[action]["ticks"]+40)
                records.append(dict(action=action,prefix=index,status=row["status"],artifact=str(destination),
                                    charged_ticks=row.get("charged_ticks",OPTIONS[action]["ticks"]+40)))
                save(path/"progress.json",dict(room=room,completed=len(records),charged_ticks=charged,
                                               prefix=index,elapsed=time.perf_counter()-started))
        if index==max(prefixes):break
        # Exploration chooses only from public startup menus; it is not a
        # hidden-truth-selected teacher and does not claim process safety.
        z=prepared["symbols"]
        if curriculum=="leader" and z["seen"]:
            if z["r0_tangent_position"] < -.06:
                selected="move_0_3"
            elif z["r0_tangent_position"] > .06:
                selected="move_0_7"
            elif z["r1_tangent_position"] < .85:
                selected="move_1_3"
            elif z["r1_normal_position"] < .85:
                selected="move_1_1"
            else:
                selected="push_r0_1_4"
            if not z["legal:"+selected]:selected="brake"
        else:
            selected=next((a for a in ("push_both_1_4","push_r0_1_4","push_r1_1_4")
                           if z["legal:"+a]),"brake")
        command=prepared["commands"][selected]
        event=world.step(command)
        prefix_rows.append(dict(public=prepared["observation"],public_sha256=root_hash,
                                selected=selected,command=command,event=event))
        frontend.record_command(selected)
    summary=dict(room=room,curriculum=curriculum,branches=len(records),complete=sum(r["status"]=="complete" for r in records),
                 errors=sum(r["status"] in ("error","timeout") for r in records),
                 charged_branch_ticks=charged,charged_prefix_ticks=world.env.total_periods,
                 reserved_ticks=reserved,wall_seconds=time.perf_counter()-started,
                 prefix_termination=world.env.termination,prefix_success=bool(world.env.success))
    save(path/"prefix.json.gz",dict(fixture=fixture,room=room,rows=prefix_rows))
    save(path/"index.json",records);save(path/"summary.json",summary)
    return summary
