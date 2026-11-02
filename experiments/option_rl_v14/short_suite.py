"""Bounded sequential developer ablations; no automatic long-training expansion."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from experiments.option_rl_v14.live import atomic_json,read_json
from experiments.option_rl_v13.supervise import fault

ROOT=Path(__file__).resolve().parents[2]
VARIANTS=[
    ('shared_mc_02',['--shared','--advantage','mc','--hold-seconds','.2']),
    ('split_mc_02',['--advantage','mc','--hold-seconds','.2']),
    ('split_gae_02',['--advantage','gae','--hold-seconds','.2']),
    ('split_gae_08',['--advantage','gae','--hold-seconds','.8']),
]


def run(out,panel,iterations=40,seconds_per_variant=240,workers=8,only=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);panel=Path(panel).resolve()
    variants=[v for v in VARIANTS if only is None or v[0]==only]
    if not variants:raise ValueError('unknown variant')
    plan=dict(variants=variants,iterations=iterations,seconds_per_variant=seconds_per_variant,workers=workers,
        panel=str(panel),panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
        scope='8 fixed developer tasks + 8 perturbations; one training seed; not random-task generalization',
        shared_settings='extended context, flat encoder, independent legal action heads, entropy .02, gamma1, targetKL .03, unchanged v13 controller/reward')
    atomic_json(out/'plan.json',plan);results=[]
    for name,extra in variants:
        dest=out/name
        command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiments/option_rl_v14/train.py'),
            '--out',str(dest),'--panel','micro','--panel-path',str(panel),'--iterations',str(iterations),
            '--hours',str(seconds_per_variant/3600),'--episodes','8','--workers',str(workers),
            '--eval-every','10','--render-every','0',*extra]
        atomic_json(out/'status.json',dict(current=name,completed=results,command=command))
        with (out/f'{name}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(out/f'{name}_process.json',dict(pid=child.pid,supervisor_pid=os.getpid(),command=command))
            started=time.monotonic();reason=None
            while child.poll() is None:
                h=read_json(dest/'history.json',[]) or [];live=read_json(dest/'live_status.json',{}) or {}
                reason=fault(h,live)
                manifest=read_json(dest/'manifest.json',{}) or {}
                for source,digest in manifest.get('source_sha256',{}).items():
                    if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
                        reason='runtime_source_changed:'+source;break
                if time.monotonic()-started>seconds_per_variant+90:reason='suite_wall_overrun'
                if reason and dest.exists():atomic_json(dest/'STOP.json',dict(reason=reason,source='short_suite'))
                if time.monotonic()-started>seconds_per_variant+150:
                    import signal
                    os.killpg(child.pid,signal.SIGTERM)
                time.sleep(5)
            result=read_json(dest/'result.json',{}) or {}
            results.append(dict(name=name,exit_code=child.returncode,result=result,seconds=time.monotonic()-started,stop_reason=reason))
            atomic_json(out/'results.json',results)
        if child.returncode!=0 or result.get('stop_reason'):
            atomic_json(out/'status.json',dict(current=None,status='stopped_on_fault',completed=results));return 1
    atomic_json(out/'status.json',dict(current=None,status='complete',completed=results));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--seconds-per-variant',type=float,default=240);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--only',choices=[v[0] for v in VARIANTS])
    raise SystemExit(run(**vars(p.parse_args())))
