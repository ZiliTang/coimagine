"""One-step interventions at the 44 recorded, legal-but-missing proposal boundaries."""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from tools import cap_general_actor_availability as audit

ref = audit.ref
OUT = audit.OUT / 'candidate_sweep'
DOCS = Path(__file__).resolve().parents[1] / 'docs/research/key_findings/2026-11-20_hard_actor_fit'


@torch.inference_mode()
def execute(job):
    fixture, room, branch, fork, snapshot, index = job
    actor = branch['actor']
    mark = branch['first']
    model = ref.MODELS[actor]
    old = mark['action']
    menu = snapshot['pool']
    treatment = (menu + [old])[index]
    previous_offer = previous_execution = None
    reached = False
    with audit.world(fixture) as (env, library):
        while not audit.ended(env):
            step = env.decision_count
            panel = library.panel(env.history)
            action = ref.choose(model, panel)
            if step == fork['cut']:
                assert env.history.sha256 == fork['prefix'] and action == fork['actions'][actor]
                action = dict(action, ticks=action['ticks'] + fork['micro'][actor]['requested_delta'])
            if step == mark['step']:
                assert env.history.sha256 == mark['history'] == snapshot['h']
                assert panel[4] == menu and panel[5] == snapshot['raw'] and old not in menu
                assert action == mark['now']['chosen']
                reached = True
                action = treatment
                ref.dq.validate_action(action, ref.dq.current_event(env.history)['observation'])
            if step < mark['step']:
                if old in panel[4]:
                    previous_offer = mark['step'] - step
                if action == old:
                    previous_execution = mark['step'] - step
            audit.advance(env, action)
        assert reached
        ticks = env.env.total_periods
        assert env.env.termination in ('success', 'task_budget_exhausted') or (
            env.env.termination is None and (ticks == 1800 or env.decision_count == 300))
        success = int(env.env.success)
        chosen = treatment == mark['now']['chosen']
        if chosen:
            assert success == branch['perturbed']
        if index == len(menu):
            assert success == branch['one_forced_original']
        return dict(room=room, actor=actor, index=index, n_offered=len(menu), chosen=chosen,
                    old=index==len(menu), action=treatment, success=success, ticks=ticks,
                    boundaries=env.decision_count, history=env.history.sha256,
                    prefix=mark['history'], step=mark['step'], previous_offer=previous_offer,
                    previous_execution=previous_execution,
                    sources=[s for s,p in zip(ref.SIDS,snapshot['raw']) if p==treatment])


def main():
    torch.set_num_threads(1)
    _, rooms, ref.SIDS = ref.scale.inputs()
    protocol = ref.read(ref.OUT/'protocol.json')
    assert ref.campaign.base.digest(rooms)==protocol['panel'] and protocol['sids']==ref.SIDS
    assert ref.campaign.sha(ref.scale.OUT/'actor.pt')==protocol['old']
    assert ref.campaign.sha(ref.fit.OUT/'best_train.pt')==protocol['layernorm']
    ref.MODELS=[ref.sets.checkpoint(ref.scale.OUT), ref.fit.load_best()]
    ref.dq.MODULES=[ref.dq.load_native((ref.campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in ref.SIDS]
    jobs=[]
    for path in sorted((audit.OUT/'rooms').glob('*.json')):
        room=path.stem
        for b in ref.read(path)['branches']:
            if not b['primary'] or b['first']['stage']!='proposal':
                continue
            fork=ref.read(ref.OUT/'rooms'/path.name)
            snap=ref.read(audit.OUT/'geometry'/f"{room}_{b['actor']}.json")['perturbed']['first']
            assert len(snap['pool'])==b['first']['now']['pool']
            jobs.extend((rooms[room],room,b,fork,snap,i) for i in range(len(snap['pool'])+1))
    assert len(jobs)==145
    OUT.mkdir(exist_ok=True)
    ref.save(OUT/'progress.json',dict(completed=0,total=len(jobs)))
    results=[]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        for f in as_completed([pool.submit(execute,job) for job in jobs]):
            results.append(f.result())
            ref.save(OUT/'progress.json',dict(completed=len(results),total=len(jobs)))
    results.sort(key=lambda r:(r['room'],r['actor'],r['index']))
    groups={}
    for r in results:
        groups.setdefault((r['room'],r['actor']),[]).append(r)
    assert len(groups)==44
    cases=[]
    for (room,actor),rows in groups.items():
        offered=rows[:-1]
        old=rows[-1]
        assert old['old'] and old['n_offered']==len(offered) and sum(r['chosen'] for r in offered)==1
        chosen=next(r for r in offered if r['chosen'])
        cases.append(dict(room=room, actor=actor, prefix=old['prefix'], step=old['step'],
                          offered=len(offered), successful_offered=sum(r['success'] for r in offered),
                          selected=chosen['success'], old=old['success'], previous_offer=old['previous_offer'],
                          previous_execution=old['previous_execution']))
    assert sum(r['selected'] for r in cases)==27 and sum(r['old'] for r in cases)==18
    def counts(subset):
        return dict(n=len(subset),selected=sum(r['selected'] for r in subset),
                    any_offered=sum(bool(r['successful_offered']) for r in subset),
                    only_old=sum(not r['successful_offered'] and r['old'] for r in subset),
                    neither=sum(not r['successful_offered'] and not r['old'] for r in subset),
                    both=sum(bool(r['successful_offered']) and r['old'] for r in subset))
    stats=dict(all=counts(cases), failed=counts([r for r in cases if not r['selected']]),
               by_actor=[counts([r for r in cases if r['actor']==i]) for i in range(2)],
               runs=len(results),rooms=len(set(r['room'] for r in cases)),sha256=ref.campaign.sha(__file__),
               panel=protocol['panel'], old=protocol['old'],layernorm=protocol['layernorm'])
    with (DOCS/'candidate_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)
    ref.save(OUT/'actions.json',results)
    ref.save(OUT/'FINISHED.json',stats)
    print(stats,flush=True)


if __name__=='__main__':
    main()
