"""One bounded continuation, preserving optimizer/RNG and every evaluation snapshot."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from experiments.option_rl_v15.live import atomic_json,read_json
from experiments.option_rl_v13.supervise import fault

ROOT=Path(__file__).resolve().parents[2]


def run(parent,job,iterations=60,seconds=480,workers=8,eval_every=5):
    import torch
    parent=Path(parent).resolve();job=Path(job).resolve();data=torch.load(parent,map_location='cpu',weights_only=False)
    if job.exists() and any(job.iterdir()):raise ValueError('preserve old continuation jobs')
    job.mkdir(parents=True,exist_ok=True);out=job/'run16_001'
    config=data['manifest']['config'];model=data['model_config'];rollout=data['rollout_config']
    command=[str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).with_name('train.py')),
        '--out',str(out),'--resume-from',str(parent),'--iterations',str(iterations),'--hours',str(seconds/3600),
        '--episodes',str(config['episodes']),'--seed',str(data['manifest']['seed']),'--workers',str(workers),
        '--hidden',str(data['hidden']),'--epochs',str(config['epochs']),'--lr',str(config['lr']),'--entropy',str(config['entropy']),
        '--advantage',config['advantage'],'--trace-seconds',str(config['trace_seconds']),'--hold-seconds',str(rollout['hold_seconds']),
        '--target-kl',str(config['target_kl']),'--context',rollout['context'],'--encoder',model['encoder'],
        '--panel',config['panel'],'--eval-every',str(eval_every),'--render-every','0']
    if config.get('panel_path'):command+=['--panel-path',str(Path(config['panel_path']).resolve())]
    if not model['separate']:command+=['--shared']
    if model.get('coupled'):command+=['--coupled']
    if model.get('autoregressive'):command+=['--autoregressive']
    plan=dict(command=command,parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        parent_iteration=data['history'][-1]['iteration'],new_iterations=iterations,seconds=seconds,
        workers=workers,eval_every=eval_every,scope='same v15 policy/controller/reward; exact bounded continuation; no automatic next job')
    atomic_json(job/'plan.json',plan);started=time.monotonic();saved=False
    with (job/'training.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(job/'processes.json',dict(supervisor_pid=os.getpid(),trainer_pid=child.pid,run=str(out)))
        while child.poll() is None:
            h=read_json(out/'history.json',[]) or [];live=read_json(out/'live_status.json',{}) or {};reason=fault(h,live)
            manifest=read_json(out/'manifest.json',{}) or {}
            if manifest and not saved:
                for name in manifest['source_sha256']:
                    dest=job/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
                shutil.copy2(out/'manifest.json',job/'source_snapshot/manifest.json');saved=True
            for name,digest in manifest.get('source_sha256',{}).items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:reason='runtime_source_changed:'+name;break
            if time.monotonic()-started>seconds+90:reason='continuation_wall_overrun'
            if reason and out.exists():atomic_json(out/'STOP.json',dict(reason=reason,source='continuation_supervisor'))
            if time.monotonic()-started>seconds+150:os.killpg(child.pid,signal.SIGTERM)
            atomic_json(job/'status.json',dict(status='running',iteration=h[-1]['iteration'] if h else None,reason=reason,phase=live.get('phase')))
            time.sleep(5)
        result=read_json(out/'result.json',{}) or {}
        final=dict(exit_code=child.returncode,result=result,seconds=time.monotonic()-started)
        atomic_json(job/'supervisor_result.json',final);atomic_json(job/'status.json',dict(status='complete' if child.returncode==0 else 'failed',**final))
        return child.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=60);p.add_argument('--seconds',type=float,default=480)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--eval-every',type=int,default=5)
    raise SystemExit(run(**vars(p.parse_args())))
