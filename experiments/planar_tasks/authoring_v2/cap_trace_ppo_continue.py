"""Replay the PPO optimizer and continue paired GA/GLOBAL seed17 for 40 rounds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import torch
from . import cap_trace_ppo_comparison as previous
from . import cap_trace_value_ppo as ppo
from .cap_trace_value_shuffle import schedule
from .random_room_tasks import generate
from .cap_trace_value_long import phase, sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.relational_search.contract import digest

COMPARISON = previous.BASE/'cap_trace_ppo_confirm_20261118_01'
START = {'GA': COMPARISON/'GA_s17', 'GLOBAL': previous.REFERENCE}


def prepare(out, cases, ref):
    frozen = read(COMPARISON/'FROZEN.json')
    prior = read(COMPARISON/'protocol.json')
    assert read(COMPARISON/'FINISHED.json')['complete']
    source_hashes = {**prior['sources'], **prior['ppo']['source_hashes']}
    assert all(sha(f)==h for f,h in {**source_hashes,**prior['parents']}.items())
    starts = {str(d/name): frozen[str(d/name)] for d in START.values()
              for name in ('initial.pt','actor_critic_r40.pt')}
    assert all(sha(f)==h for f,h in starts.items())
    order = read(previous.PARENT/'models/GLOBAL_s17/online_order.json')
    blocks = schedule(cases,order,cycles=19)
    assert blocks[:36]==read(previous.REFERENCE/'schedule.json')
    save(out/'schedule.json',blocks[36:])
    old = read(previous.SEALED)['cases']
    assert sha(previous.SEALED)==prior['sealed_sha']
    known = read(Path(ref['bundle'])/'used_scene_identities.json') + cases + old
    known += read(previous.PARENT/'panels/final_test.json')['cases']
    contents={c['content_sha256'] for c in known}
    ids={c['case_id'] for c in known}
    panel=[]
    for r in generate(2026111841)['cases']:
        f=r['fixture']
        content=digest({k:v for k,v in f.items() if k!='scene_id'})
        case_id='continue_confirm_'+r['id']
        assert case_id not in ids and content not in contents
        ids.add(case_id); contents.add(content)
        panel.append(dict(case_id=case_id,fixture=f,fixture_sha256=digest(f),content_sha256=content,movable_count=r['movable_count']))
    assert all(sum(c['movable_count']==n for c in panel)==50 for n in range(2,6))
    save(out/'confirmation.json',dict(cases=panel,generator_seed=2026111841))
    for f in (Path(__file__),Path('tools/cap_trace_ppo_continue_report.py')):
        source_hashes[str(f)]=sha(f)
        shutil.copyfile(f,out/f.name)
    save(out/'protocol.json',dict(seed=17,starts=starts,sources=source_hashes,
        schedule_sha256=sha(out/'schedule.json'),panel_sha256=sha(out/'confirmation.json'),
        primary='paired (GLOBAL4000 - GA4000) minus (GLOBAL2000 - GA2000) on fresh 200 roots',
        secondary='each arm post-pre, endpoint gap; previously viewed panel at 2000,3000,4000',
        units='10000 paired scene bootstrap draws stratified by object count; conditional on seed17',
        selection='both arms tied for best on previous panel; shared seed17 chosen from ties',
        training_episodes=4000,diagnostic_executions=800,confirmation_executions=800,
        ppo=prior['ppo'],optimizer='deterministic reconstruction against every old round checkpoint',
        historical_frames='prior trajectory update replay only; no new episodes used in reconstruction'))


def optimizer_for(model):
    return torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                             dict(params=model.critic.parameters(),lr=3e-4)])


def recover(out,arm):
    directory=START[arm]
    model=previous.MatchedPolicy(previous.origin(arm,17))
    model.load_state_dict(torch.load(directory/'initial.pt',weights_only=True))
    optimizer=optimizer_for(model)
    rng=np.random.default_rng(17)
    steps=0
    for r in range(5,41):
        phase(out,'recover_optimizer',arm=arm,round=r)
        rows=read(directory/'training'/f'r{r}'/'results.json')
        metrics=ppo.fit(model,optimizer,rows,rng)
        expected=torch.load(directory/f'actor_critic_r{r}.pt',weights_only=True)
        assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items()), (arm,r)
        steps+=metrics['updates']
    torch.save(dict(optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state),
               out/arm/'recovered.pt')
    save(out/arm/'recovery.json',dict(matched_rounds=36,updates=steps,all_tensors_equal=True))
    return model,optimizer,rng


def evaluate(pool,out,arm,r,checkpoint,panel,label):
    phase(out,label,arm=arm,round=r)
    dest=out/label/f'{arm}_r{r}'
    jobs=[(c,dest/c['case_id'],previous.origin(arm,17),checkpoint,17,False) for c in panel]
    return ppo.collect(pool,jobs,dest)


def run(out):
    from tools.cap_trace_ppo_continue_report import curves,report
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    ref=read(previous.SHUFFLE/'inputs.json')
    cases=read(previous.PARENT/'panels/online_training.json')['cases']
    prepare(out,cases,ref)
    by_id={c['case_id']:c for c in cases}
    schedules=read(out/'schedule.json')
    proto=read(out/'protocol.json')
    states={}
    for arm in START:
        (out/arm).mkdir()
        states[arm]=recover(out,arm)
    current={arm:d/'actor_critic_r40.pt' for arm,d in START.items()}
    history={arm:[] for arm in START}
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=previous.init_worker,
                             initargs=(ref['bundle'],ref['priority_bundle'])) as pool:
        for block in schedules:
            r=block['round']
            for arm,(model,opt,rng) in states.items():
                phase(out,'collect',arm=arm,round=r)
                dest=out/arm/'training'/f'r{r}'
                rows=ppo.collect(pool,[(by_id[c['case_id']],dest/c['case_id'],previous.origin(arm,17),
                    current[arm],c['action_seed'],True) for c in block['cases']],dest)
                phase(out,'fit',arm=arm,round=r)
                metrics=ppo.fit(model,opt,rows,rng)
                current[arm]=out/arm/f'r{r}.pt'
                torch.save(model.state_dict(),current[arm])
                torch.save(dict(optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,round=r),
                           out/arm/'latest_optimizer.pt')
                history[arm].append(dict(round=r,count=len(rows),success=sum(c['success'] for c in rows),**metrics))
                save(out/arm/'history.json',history[arm])
            if r%4==0: curves(out)
            if r in (60,80):
                for arm in START:
                    evaluate(pool,out,arm,r,current[arm],read(previous.SEALED)['cases'],'diagnostic')
                curves(out)
        assert sha(out/'schedule.json')==proto['schedule_sha256']
        assert sha(out/'confirmation.json')==proto['panel_sha256']
        assert all(sha(f)==h for f,h in {**proto['sources'],**proto['starts']}.items())
        frozen={arm:{40:START[arm]/'actor_critic_r40.pt',80:current[arm]} for arm in START}
        save(out/'FROZEN.json',{str(f):sha(f) for d in frozen.values() for f in d.values()})
        panel=read(out/'confirmation.json')['cases']
        for arm,versions in frozen.items():
            for r,cp in versions.items():evaluate(pool,out,arm,r,cp,panel,'confirmation')
    report(out)
    phase(out,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
