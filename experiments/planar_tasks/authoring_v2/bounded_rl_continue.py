"""Explicit long continuation of the frozen bounded direction pilot."""
from . import bounded_rl_study as S
from .random_curriculum import build
from .semantic_rl_resume import sha
from pathlib import Path
import argparse,shutil,time,traceback
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

B=S.BASE
PREVIOUS=B/'random_parameter_direction_20261116_01'
PARENT=B/'random_parameter_direction_long_parent_20261116_01'
TRAIN=B/'random_parameter_direction_long_20261116_01'
DIAG=B/'random_parameter_direction_long_validation_20261116_01'
EVAL=B/'random_parameter_direction_long_evaluation_20261116_01'
HOST=Path('outputs/semantic_rl_authors/random_parameter_direction_long_20261116_01')
ROUNDS=992


def restore(checkpoint,optimizer_path,source):
    model,_=S.BoundedPolicy.load(checkpoint,S.digest(source))
    if (model.direction_radius,model.radius)!=(1,4):raise ValueError('wrong parameter support')
    payload=torch.load(optimizer_path,map_location='cpu',weights_only=True)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);opt.load_state_dict(payload['optimizer'])
    if not opt.state or any(g['lr']!=3e-4 for g in opt.param_groups):raise ValueError('missing Adam state or wrong learning rate')
    for param,state in opt.state.items():
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=param.shape or not torch.isfinite(state[key]).all():raise ValueError('invalid Adam moments')
    return model,opt


def make(path,n,rounds,profile):
    c=S.Campaign.create(path,limits=dict(attempts=n,resets=n,execution_ticks=n*1800,rounds=rounds,fits=0,author_turns=0),model=dict(provider='none',name=profile))
    with c.locked():c.state.update(profile=profile,runs=[],phase='ready');c._save()
    return c


def prepare():
    old_seal=S.Campaign(PREVIOUS).verify_frozen();input_seal=S.Campaign(S.PARENT).verify_frozen()
    state=S.read(PREVIOUS/'campaign.json');assert state['completion']=='complete' and state['completed_rounds']==32
    assert sha(PREVIOUS/'models/final.pt')==sha(PREVIOUS/'models/round_0031.pt')
    source=(S.PARENT/'inputs/source.py').read_text();model,opt=restore(PREVIOUS/'models/final.pt',PREVIOUS/'models/optimizer_0031.pt',source)
    ref,_=S.Policy.load(S.PARENT/'inputs/supervisor.pt',S.digest(source));assert S.state_digest(model.supervisor)==S.state_digest(ref)
    old_panels={k:S.read(S.PARENT/'panels'/f'{k}.json') for k in ('train','validation','test','anchors')}
    excluded=[r['fixture_sha256'] for p in old_panels.values() for r in p['cases']]
    excluded += [r['fixture_sha256'] for k in ('train','test','roots') for r in S.read(S.OLD/'panels'/f'{k}.json')['cases']]
    panels=build(old_panels['anchors']['cases'],rounds=ROUNDS,seeds=(2026111614,2026111615,2026111616),exclude=excluded)
    p=make(PARENT,0,0,'bounded-direction-long-input-v1');(PARENT/'inputs').mkdir()
    for source_path,target in [(S.PARENT/'inputs/source.py','source.py'),(S.PARENT/'inputs/supervisor.pt','supervisor.pt'),(PREVIOUS/'models/final.pt','initial.pt'),(PREVIOUS/'models/optimizer_0031.pt','optimizer.pt')]:shutil.copyfile(source_path,PARENT/'inputs'/target)
    for k in ('train','test'):S.save(PARENT/'panels'/f'{k}.json',panels[k])
    for k in ('anchors','validation'):S.save(PARENT/'panels'/f'{k}.json',old_panels[k])
    old_metrics=S.read(PREVIOUS/'metrics.json')
    protocol=dict(S.read(S.PARENT/'protocol.json'),arms={'direction':1},rounds=ROUNDS,episodes_per_arm=ROUNDS*16,arm_wall_seconds=10800,diagnostic_interval=32,round_offset=32,episode_offset=512,optimizer_step_offset=len(old_metrics['updates']),train_seed=2026111614,test_seed=2026111616,parent_seal=old_seal,input_seal=input_seal,stop='STOP file after current batch; two consecutive diagnostics anchors<11/13; errors stop; 300s wall reserve',sampling='new seeded draws from identical distribution, geometry-only rejection; not exact extension of the old pre-generated sequence',rng='rollout seed=2026111607+global_round*16+i; PPO shuffle seed=2026111607+global_round; no dependence on global RNG',selection='last complete checkpoint; never select on test; fresh48 only after training freezes',checkpoint_sha256=sha(PREVIOUS/'models/final.pt'),optimizer_sha256=sha(PREVIOUS/'models/optimizer_0031.pt'))
    S.save(PARENT/'protocol.json',protocol)
    S.save(PARENT/'acceptance.json',dict(actor_exact_copy=True,supervisor_unchanged=True,optimizer_loaded=True,adam_parameters=len(opt.state),adam_steps=sorted({float(v['step']) for v in opt.state.values()})))
    S.save(HOST/'PREPARED.json',dict(seal=S.finish(p,'complete'),protocol=protocol));print('PREPARED',flush=True)


def run():
    torch.set_num_threads(1);seal=S.Campaign(PARENT).verify_frozen();protocol=S.read(PARENT/'protocol.json')
    # Reused collector references the original frozen source, which must be identical.
    source=(PARENT/'inputs/source.py').read_text();assert source==(S.PARENT/'inputs/source.py').read_text()
    model,opt=restore(PARENT/'inputs/initial.pt',PARENT/'inputs/optimizer.pt',source)
    train=S.read(PARENT/'panels/train.json')['cases'];anchors=S.read(PARENT/'panels/anchors.json')['cases'];val=S.read(PARENT/'panels/validation.json')['cases']
    c=make(TRAIN,ROUNDS*16,ROUNDS,'semantic-ppo-parallel-bounded-long-v1');d=make(DIAG,(ROUNDS//32)*37,0,'bounded-long-validation-v1')
    (TRAIN/'inputs').mkdir();(TRAIN/'models').mkdir();shutil.copyfile(PARENT/'inputs/source.py',TRAIN/'inputs/source.py');S.save(TRAIN/'inputs/provenance.json',dict(parent_seal=seal,schema=model.schema));S.save(TRAIN/'protocol.json',protocol)
    checkpoint=TRAIN/'models/initial.pt';model.save(checkpoint,dict(resumed_from=str(PREVIOUS)))
    metrics=dict(rounds=[],updates=[],diagnostics=[]);S.save(TRAIN/'metrics.json',metrics)
    with c.locked():c.state.update(config=dict(protocol,direction_radius=1,feature_runtime='typed_stream',execution_mode='duration'),completed_rounds=0,continuation=dict(parent=str(PREVIOUS),round_offset=32,episode_offset=512,optimizer_step_offset=protocol['optimizer_step_offset'],optimizer_restored=True),training_schedule=[x['id'] for x in train]);c._save()
    start=time.monotonic();completion='complete';bad_previous=False;io=S.SaveTiming()
    try:
        with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
            for r in range(ROUNDS):
                if (HOST/'STOP').exists():completion='stopped_user';break
                if time.monotonic()-start>10500:completion='incomplete_wall_admission';break
                global_round=32+r
                with c.locked():c.state.update(phase='collecting',active_round=r,policy_checkpoint=str(checkpoint.relative_to(TRAIN)));c._save()
                t=time.monotonic();rows,episodes=S.collect(c,train[r*16:(r+1)*16],checkpoint,'direction',pool,io,training=True,round_number=global_round);collection=time.monotonic()-t
                t=time.monotonic();update=S.bounded_ppo_update(model,opt,episodes,epochs=4,minibatch_size=64,seed=2026111607+global_round);update_seconds=time.monotonic()-t
                for u in update['curve']:metrics['updates'].append(dict(u,round=global_round,optimizer_step=protocol['optimizer_step_offset']+len(metrics['updates'])+1,grad_norm=u['actor_grad_norm']))
                checkpoint=TRAIN/'models'/f'round_{r:04d}.pt';io.call(model.save,checkpoint,dict(round=global_round));io.call(torch.save,dict(optimizer=opt.state_dict(),next_global_round=global_round+1),TRAIN/'models'/f'optimizer_{r:04d}.pt')
                ticks=sum(x['ticks'] for x in rows);decisions=sum(x['decisions'] for x in rows)
                point=dict(S.summary(rows),round=global_round,success_rate=sum(x['success'] for x in rows)/16,ticks=ticks,decisions=decisions,collection_seconds=collection,update_seconds=update_seconds,ticks_per_second=ticks/collection,decisions_per_second=decisions/collection,updates=update['updates'],total_optimizer_steps=protocol['optimizer_step_offset']+len(metrics['updates']),logp_recompute_error=update['logp_recompute_error'],early_stop_kl=update['early_stop_kl'],wall_truncations=0,tool_errors=0,elapsed_seconds=time.monotonic()-start,checkpoint=str(checkpoint.relative_to(TRAIN)),parameter_changes={k:sum(x['parameter_audit'][k] for x in rows) for k in ('duration_changed','speed_changed','direction_changed','violations')})
                metrics['rounds'].append(point);io.call(S.save,TRAIN/'metrics.json',metrics)
                with c.locked():c.state.update(completed_rounds=r+1,policy_checkpoint=str(checkpoint.relative_to(TRAIN)));c.state['used']['rounds']=r+1;c._save()
                io.log('direction-long round'+str(global_round+1),details=sum(x['record_details'] for x in rows));print('ROUND',global_round+1,point['success'],'/16',flush=True)
                if (HOST/'STOP').exists():completion='stopped_user';break
                if (r+1)%32==0:
                    rr,_=S.collect(d,anchors+val,checkpoint,'direction',pool,io,round_number=global_round+1);a=S.summary(rr[:13]);v=S.summary(rr[13:]);diagnostic=dict(round=global_round+1,anchors=a,validation=v)
                    metrics['diagnostics'].append(diagnostic);S.save(TRAIN/'metrics.json',metrics);S.save(HOST/'LATEST_DIAGNOSTIC.json',diagnostic);print('DIAGNOSTIC',diagnostic,flush=True)
                    bad=a['success']<11
                    if bad and bad_previous:completion='stopped_anchor_regression';break
                    bad_previous=bad
            shutil.copyfile(checkpoint,TRAIN/'models/final.pt');finished=dict(training_seal=S.finish(c,completion),diagnostic_seal=S.finish(d,completion),completion=completion)
            S.save(HOST/'TRAIN_FINISHED.json',finished)
            if completion=='stopped_user' or (HOST/'STOP').exists():S.save(HOST/'FINISHED.json',dict(finished,evaluation='not_run_user_stop'));return
            # Fresh test is never loaded before final checkpoint is frozen.
            test=S.read(PARENT/'panels/test.json')['cases'];e=make(EVAL,183,0,'bounded-long-final-evaluation-v1');allrows=[]
            for name,ckpt in [('supervised',PARENT/'inputs/supervisor.pt'),('pilot',PARENT/'inputs/initial.pt'),('long',TRAIN/'models/final.pt')]:
                rows,_=S.collect(e,anchors+test,ckpt,name,pool,io,all_details=True);allrows+=rows;print('FINAL',name,'anchors',S.summary(rows[:13]),'test',S.summary(rows[13:]),flush=True)
            S.save(EVAL/'comparison.json',allrows);S.save(EVAL/'selection.json',dict(parent=seal,completion=completion,selection='fixed final',fresh_test=True));S.save(HOST/'FINISHED.json',dict(finished,evaluation_seal=S.finish(e,'complete')));io.log('finished')
    except BaseException:
        with c.locked():c.state.update(phase='failed',completion='incomplete_error');c._save()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);a=p.parse_args();HOST.mkdir(parents=True,exist_ok=True)
    try:prepare() if a.command=='prepare' else run()
    except BaseException as exc:S.save(HOST/(a.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
