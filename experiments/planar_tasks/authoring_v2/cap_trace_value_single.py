"""User-authorized seed17 continuation after measured, exact performance fixes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data as F
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime as W
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.fast_learner import train_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ReplayCache
from .cap_trace_value_long import ROOT, sha, phase


def prepare(parent, paused, performance, out):
    if out.exists():raise FileExistsError(out)
    old = read(paused/'inputs.json')
    for rel,expected in old['sources'].items():
        assert sha(ROOT/rel)==expected, 'Frozen original source changed: '+rel
    checkpoint = paused/'models/GLOBAL_s17/online_r9_interrupted.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['online_updates']==12023 and state['updates']==23
    assert state['seed']==17 and state['arm']=='GLOBAL'
    out.mkdir(parents=True)
    files = list(old['sources']) + [str(Path(__file__).relative_to(ROOT)),
        'experiments/planar_tasks/semantic_rl/program_trace_value_v1/fast_runtime.py',
        'experiments/planar_tasks/semantic_rl/program_trace_value_v1/fast_learner.py',
        'experiments/planar_tasks/semantic_rl/program_trace_value_v1/resident_replay.py',
        'tools/cap_trace_value_single_report.py', 'tests/test_cap_trace_value_perf.py']
    record = dict(parent=str(parent),paused=str(paused),performance=str(performance),started=time.time(),
        bundle=old['bundle'],priority_bundle=old['priority_bundle'],
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        sources={rel:sha(ROOT/rel) for rel in files},
        original_online_panel_sha256=sha(parent/'panels/online_training.json'),
        diagnostic_panel_sha256=sha(parent/'panels/final_test.json'),
        sealed_confirmation_sha256=sha(paused/'panels/confirmation.json'),
        protocol=dict(seed=17,other_seeds='29/43 paused; no training or selection',
            resumed_episodes=450,resumed_online_updates=12023,total_episodes=2000,total_online_updates=60000,
            new_training_episodes=1550,new_updates=47977,first_round_updates=1477,
            new_monitor_executions=256,new_diagnostic_executions=600,new_physical_executions=2406,
            confirmation_executions=0,confirmation='sealed for later multi-seed confirmation',
            fixed_online_roots=200,epsilon=.05,updates_per_round=1500,episodes_per_round=50,
            unchanged='network/reward/gamma/normalizer/optimizer/target/clip/replay ratio/boundary sampling',
            workers=12,learner_threads=4,worker_threads=1,wall_limit_seconds=21600,
            metrics='reward, TD loss, epsilon-greedy entropy, Q, gradient, sampling and compute seconds',
            selection='fixed final2000; diagnostic-only single seed, no best checkpoint selection',
            optimizations='resident lossless LZ4 rows, cached auditable history, scoped physical sensors, single JSON write',
            physical_parity='4 paired complete trajectories, public/action/score/history/frame/transition exact',
            training_parity='100 real-sized updates, all model/Adam/target/RNG exact at 1/2/4 threads'))
    save(out/'inputs.json',record)
    for rel in record['sources']:
        target=out/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    # Reuse immutable results as provenance, never overwrite the paused run.
    inherited={}
    for pattern in ('online/GLOBAL_s17/r*.json','models/GLOBAL_s17/online_r*.json',
                    'models/GLOBAL_s17/online_r*_metrics.jsonl','monitor/GLOBAL_s17_e*.json',
                    'diagnostic/GLOBAL_s17_e*.json'):
        for source in paused.glob(pattern):
            if source.name.endswith('_progress.json') or source.name=='online_r9_metrics.jsonl':continue
            rel=source.relative_to(paused)
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
            inherited[str(rel)]=dict(source=str(source),sha256=sha(source))
    save(out/'inherited.json',inherited)
    # Packed files are a derived cache of unchanged original arrays.
    for source in (performance/'packed').glob('*/*'):
        dest=out/'packed'/source.relative_to(performance/'packed')
        dest.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,dest)
    save(out/'pause_resume.json',dict(updates_preserved=12023,episodes_preserved=450,
         interrupted_round=9,remaining_updates_in_round=1477,
         unsaved_fine_grained_metric_updates=23,discarded_training_updates=0,
         original_stop='user requested pause; STOP checkpoint saved, supervisor then exited'))
    return record


def verify(out):
    d=read(out/'inputs.json')
    assert all(sha(ROOT/p)==v for p,v in d['sources'].items())
    assert sha(d['checkpoint'])==d['checkpoint_sha256']
    assert sha(Path(d['parent'])/'panels/online_training.json')==d['original_online_panel_sha256']
    assert sha(Path(d['parent'])/'panels/final_test.json')==d['diagnostic_panel_sha256']
    assert sha(Path(d['paused'])/'panels/confirmation.json')==d['sealed_confirmation_sha256']


def batch(pool,out,jobs,name):
    if (out/(name+'.json')).exists():raise FileExistsError(name)
    start=time.monotonic();results=[]
    futures=[pool.submit(W.run,job) for job in jobs]
    for future in as_completed(futures):
        results.append(future.result())
        save(out/(name+'_progress.json'),dict(completed=len(results),total=len(jobs),seconds=time.monotonic()-start))
        if len(results)%12==0 or len(results)==len(jobs):
            print('SINGLE_PHYSICAL',name,len(results),'/',len(jobs),'seconds',round(time.monotonic()-start,1),flush=True)
    complete=len(results)==len(jobs) and all(r['complete'] and not r.get('error') for r in results)
    save(out/(name+'.json'),dict(complete=complete,results=results,seconds=time.monotonic()-start))
    if not complete:raise RuntimeError('Incomplete physical batch '+name)
    return results


def evaluate(pool,out,checkpoint,episodes,cases,panel):
    condition=f'GLOBAL_s17_e{episodes}'
    jobs=[dict(directory=str(out/panel/condition/c['case_id']),case=c,checkpoint=str(checkpoint),
               condition=condition) for c in cases]
    return batch(pool,out,jobs,f'{panel}/{condition}')


def run(parent,paused,performance,out):
    record=prepare(parent,paused,performance,out)
    torch.set_num_threads(4)
    from tools.cap_trace_value_single_report import render_live, final_report
    cache=ReplayCache(out/'packed')
    phase(out,'load_resident_replay',seed=17)
    offline=cache.get(parent/'arrays/train')
    paths=[parent/'replay/GLOBAL_s17/r4']+[paused/'replay/GLOBAL_s17'/f'block{r}' for r in range(5,10)]
    online=cache.online(paths)
    save(out/'memory.json',dict(resident_replay_bytes=cache.resident_bytes,offline_boundaries=len(offline),
                               online_boundaries=len(online)))
    cases=read(parent/'panels/online_training.json')['cases']
    by_id={c['case_id']:c for c in cases}
    order=read(parent/'models/GLOBAL_s17/online_order.json')
    development={c['case_id']:c for c in read(Path(record['bundle'])/'scenes.json')}
    monitor=[development[c] for c in read(parent/'split.json')['monitor']]
    diagnostic=read(parent/'panels/final_test.json')['cases']
    checkpoint=Path(record['checkpoint'])
    # Finish exactly the interrupted update round; no new physics or RNG reset.
    phase(out,'fit',seed=17,round=9,episodes=450,resume_updates=12023)
    train_round(checkpoint,out,17,9,offline,online,updates=1477)
    checkpoint=out/'models/GLOBAL_s17/online_r9.pt'
    render_live(out)
    with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn'),initializer=W.initialize,
                             initargs=(record['bundle'],record['priority_bundle'])) as pool:
        for r in range(10,41):
            if (out/'STOP').exists():raise RuntimeError('STOP requested')
            phase(out,'collect',seed=17,round=r,episodes_before=(r-1)*50)
            start=((r-1)*50)%200
            selected=[by_id[c] for c in order[start:start+50]]
            jobs=[dict(directory=str(out/'online/GLOBAL_s17'/f'r{r}'/c['case_id']),case=c,
                condition=f'GLOBAL_s17_online_r{r}',checkpoint=str(checkpoint),online=True,epsilon=.05,
                action_seed=170000+(r-1)*50+j,root_number=760+cases.index(c)) for j,c in enumerate(selected)]
            rows=batch(pool,out,jobs,f'online/GLOBAL_s17/r{r}')
            indices={c['case_id']:i for i,c in enumerate(selected)}
            rows.sort(key=lambda row:indices[row['case_id']])
            replay=out/'replay/GLOBAL_s17'/f'block{r}'
            F.assemble(replay,[row['transition_shard'] for row in rows])
            paths.append(replay);online=cache.online(paths)
            save(out/'memory.json',dict(resident_replay_bytes=cache.resident_bytes,
                 offline_boundaries=len(offline),online_boundaries=len(online)))
            phase(out,'fit',seed=17,round=r,episodes=r*50)
            train_round(checkpoint,out,17,r,offline,online)
            checkpoint=out/'models/GLOBAL_s17'/f'online_r{r}.pt'
            if r%4==0:evaluate(pool,out,checkpoint,r*50,monitor,'monitor')
            if r in (16,24,40):
                verify(out)
                evaluate(pool,out,checkpoint,r*50,diagnostic,'diagnostic')
            render_live(out)
    verify(out)
    save(out/'ALL_MODELS_FROZEN.json',dict(at=time.time(),files={str(p.relative_to(out)):sha(p)
         for p in (out/'models').rglob('*.pt')},new_confirmation_executions=0))
    save(out/'FINISHED.json',dict(complete=True,seed=17,episodes=2000,online_updates=60000,
         new_training_executions=1550,new_monitor_executions=256,new_diagnostic_executions=600,
         new_physical_executions=2406,new_confirmation_executions=0,seconds=time.time()-record['started']))
    phase(out,'complete',seed=17)
    final_report(out)


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent','paused','performance','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=Path(args.out).resolve()
    try:run(Path(args.parent).resolve(),Path(args.paused).resolve(),Path(args.performance).resolve(),out)
    except BaseException as e:
        if out.exists():save(out/'FAILED.json',dict(at=time.time(),type=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__':main()
