"""Compare a single intervention with receding-horizon CAP action selection on fresh rooms."""
import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from tools import cap_actor_rollout_value as run
from tools.cap_actor_rollout_critic import load_model

OUT = run.campaign.OUT / 'actor_replan_20261121_s17'
DOC = Path(__file__).resolve().parents[1] / 'docs/research/key_findings/2026-11-21_actor_replan'
ARMS = ('actor', 'value_once', 'flags_once', 'value_every', 'flags_every')
VALUE = None


def freeze():
    old, held, members = run.initialize()
    source = run.ref.read(run.OUT/'protocol.json')
    critic_sha = run.ref.read(run.OUT/'fit.json')['sha']
    assert run.campaign.sha(run.OUT/'value.pt') == critic_sha
    assert run.campaign.sha(run.ref.fit.OUT/'best_train.pt') == source['actor'] and members == source['sids']
    prev = run.ref.read(run.OUT/'panels.json')
    historical = run.ref.read(run.campaign.base.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')
    used = {r['content_sha256'] for r in historical}
    used.update(run.ref.read(run.campaign.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for group in [old, held, *prev.values(), run.ref.read(run.ref.scale.OUT/'natural.json'),
                  run.ref.read(run.ref.scale.OUT/'variants.json'), run.ref.read(run.campaign.base.OUT/'held.json'),
                  run.ref.read(run.campaign.OUT.parent/'cap_regional_extension_20261119_01/held.json')]:
        used.update(run.campaign.input_hash(f) for f in group.values())
    panel = {}
    counts = dict.fromkeys(range(2,6),0)
    seed = 66221015
    for entry in run.campaign.base.rooms.generate(seed)['cases']:
        n, fixture = entry['movable_count'], entry['fixture']
        if counts[n] == 50:
            continue
        digest = run.campaign.input_hash(fixture)
        assert digest not in used
        used.add(digest)
        panel[f'new_{n}_{counts[n]:03d}'] = fixture
        counts[n] += 1
    assert list(counts.values()) == [50]*4
    protocol = dict(seed=seed, geometry=run.campaign.base.digest(panel), actor=source['actor'], critic=critic_sha,
                    members=members, arms=ARMS, once_offsets=[0,5,12,25], every_start=0, ticks=1800, decisions=300,
                    flags='min blocked steps in four frozen-actor decisions; early termination stops count; ties prefer actor then candidate order',
                    value='one decision then terminal success or frozen V; same tie break',
                    evaluator=run.campaign.sha(__file__))
    OUT.mkdir(exist_ok=True)
    for name, data in [('panel',panel),('protocol',protocol)]:
        file = OUT/(name+'.json')
        if file.exists():
            assert run.campaign.base.digest(data) == run.campaign.base.digest(run.ref.read(file))
        else:
            run.ref.save(file,data)
    return panel


def simulate(env, library, first, rule):
    tick = env.env.total_periods
    command = first
    flags = 0
    horizon = 1 if rule == 'value' else 4
    for step in range(1,horizon+1):
        reason = run.runtime.advance(env,command)
        if step == 1:
            successor = env.history.sha256
        obs = run.ref.dq.current_event(env.history)['observation']
        flags += int(reason == 'blocked' or any(
            r['execution'].get('direction_unavailable') or
            r['execution'].get('stage') in ('blocked','move_blocked','yield_blocked') for r in obs['robots']))
        done = run.runtime.ended(env)
        if done or step == horizon:
            break
        command = run.ref.choose(run.ACTOR,library.panel(env.history))
    if rule == 'flags':
        score = -flags
    elif done:
        score = run.terminal(env)
    else:
        state = library.panel(env.history)[0]
        score = float(VALUE(torch.from_numpy(state[None]))[0])
        assert np.isfinite(score)
    return dict(score=score, successor=successor, ticks=env.env.total_periods-tick, steps=step)


def fork_candidate(env, library, action, rule):
    # Linux copy-on-write preserves solver contacts, CAP memory and the exact public prefix.
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            data = simulate(env,library,action,rule)
        except BaseException:
            data = {'error':traceback.format_exc()[-3000:]}
        try:
            os.write(writer,json.dumps(data).encode())
        finally:
            os._exit(0)
    os.close(writer)
    try:
        with os.fdopen(reader,'rb') as stream:
            payload = stream.read()
    finally:
        _, status = os.waitpid(pid,0)
    if status or not payload:
        raise RuntimeError(f'lookahead child {pid} exited with status {status}')
    data = json.loads(payload)
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data


@torch.inference_mode()
def episode(job):
    key, fixture, arm = job
    rule, _, frequency = arm.partition('_')
    offset = (0,5,12,25)[int(key[-3:])%4] if frequency == 'once' else 0
    stats = dict(room=key,arm=arm,plans=0,overrides=0,candidates_simulated=0,sim_ticks=0,
                 sim_steps=0,plan_seconds=0.,multi=0)
    start = time.monotonic()
    with run.runtime.world(fixture) as (env,library):
        while not run.runtime.ended(env):
            panel = library.panel(env.history)
            pool = panel[4]
            proposed = run.ref.choose(run.ACTOR,panel)
            action = proposed
            stats['multi'] += len(pool)>1
            expected = None
            if (arm != 'actor' and len(pool)>1 and env.decision_count >= offset
                    and (frequency == 'every' or stats['plans']==0)):
                before = time.monotonic()
                best = -float('inf')
                # Evaluating the preferred proposal first implements the frozen tie break.
                for candidate in [proposed, *(a for a in pool if a != proposed)]:
                    branch = fork_candidate(env,library,candidate,rule)
                    stats['candidates_simulated'] += 1
                    stats['sim_steps'] += branch['steps']
                    stats['sim_ticks'] += branch['ticks']
                    if branch['score'] > best:
                        best, action, expected = branch['score'], candidate, branch['successor']
                    if rule == 'flags' and best == 0:
                        break
                stats['plans'] += 1
                stats['overrides'] += action != proposed
                stats['plan_seconds'] += time.monotonic()-before
            run.runtime.advance(env,action)
            if expected is not None:
                assert expected == env.history.sha256
        stats.update(success=run.terminal(env),ticks=env.env.total_periods,decisions=env.decision_count,
                     history=env.history.sha256,seconds=time.monotonic()-start)
    return stats


def summarize(panel, dest):
    rows = {arm:[run.ref.read(dest/arm/(key+'.json')) for key in sorted(panel)] for arm in ARMS}
    objects = np.array([int(key.split('_')[1]) for key in sorted(panel)])
    def paired(a,b):
        return run.ref.scale.paired(*[np.array([r['success'] for r in rows[x]]) for x in (a,b)],objects)
    stats = {}
    for arm, group in rows.items():
        for actor,row in zip(rows['actor'],group):
            assert row['arm']==arm and row['room']==actor['room']
            if row['overrides']==0:
                assert row['history']==actor['history']
            if arm.endswith('every'):
                assert row['plans']==row['multi']
        stats[arm] = {k:sum(r[k] for r in group) for k in ('success','plans','multi','overrides','sim_ticks','sim_steps',
                          'plan_seconds','candidates_simulated','ticks','decisions','seconds')}
        stats[arm]['rooms_with_overrides'] = sum(r['overrides']>0 for r in group)
        stats[arm]['vs_actor'] = paired('actor',arm)
    compare = {rule:paired(rule+'_once',rule+'_every') for rule in ('value','flags')}
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'per_room.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows['actor'][0]))
        writer.writeheader()
        for group in rows.values():
            writer.writerows(group)
    run.ref.save(OUT/'FINISHED.json',dict(rooms=len(panel),arms=stats,frequency=compare))


def main():
    global VALUE
    panel = freeze()
    VALUE = load_model()
    dest = OUT/'episodes'
    jobs = []
    completed = {}
    for arm in ARMS:
        (dest/arm).mkdir(parents=True,exist_ok=True)
        completed[arm] = sum((dest/arm/(k+'.json')).exists() for k in panel)
    # Mix room strata and treatments throughout collection.
    for key in sorted(panel,key=lambda k:(k[-3:],k)):
        for arm in ARMS:
            if not (dest/arm/(key+'.json')).exists():
                jobs.append((key,panel[key],arm))
    def progress():
        run.ref.save(OUT/'progress.json',dict(completed=completed,total=sum(completed.values()),goal=len(panel)*len(ARMS)))
    progress()
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as workers:
        for future in as_completed([workers.submit(episode,job) for job in jobs]):
            row = future.result()
            arm = row['arm']
            run.ref.save(dest/arm/(row['room']+'.json'),row)
            completed[arm] += 1
            progress()
    summarize(panel,dest)


if __name__ == '__main__':
    main()
