"""Paired actor advantage under robot translations and causal public decision prefixes."""
import copy
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from experiments.planar_tasks.authoring_v2.cap_first_disagreement import legal_shift
from tools import cap_regional_cpp as cpp
from tools import cap_regional_id_ppo as ppo

base = ppo.base
hard = base.hard
study = base.study
OUT = base.OUT.parent / 'cap_regional_advantage_20261120_01'
SEED = 6820420
ID = ACTION = None


def init():
    global ID, ACTION
    torch.set_num_threads(1)
    ID = base.ExpertActor()
    ACTION = hard.actor()
    for model, path in [(ID, base.OUT/'fixed.pt'), (ACTION, hard.OUT/'actor.pt')]:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
    cpp.load()


def freeze():
    frozen = study.read(study.OUT/'frozen.json')
    for source, path in [('old', study.ext.old.OUT/'regional'), ('new', study.ext.CAM)]:
        # Later Python imports wrote bytecode next to frozen CAPs. Verify the original
        # complete inventory in a temporary view and leave the archived campaign intact.
        manifest = study.read(path/'manifest.json')
        files = {str(f.relative_to(path)) for f in path.rglob('*') if f.is_file()
                 and f.name not in ('manifest.json','.lock')}
        extra = files - manifest.keys()
        assert not (manifest.keys() - files)
        assert all('__pycache__' in Path(f).parts and f.endswith('.pyc') for f in extra)
        with tempfile.TemporaryDirectory(dir=OUT.parent) as tmp:
            view = Path(tmp)
            for rel in [*manifest,'manifest.json']:
                (view/rel).parent.mkdir(parents=True,exist_ok=True)
                os.link(path/rel,view/rel)
            assert study.ext.old.host.RoomCampaign(view).verify_frozen() == frozen['seals'][source]
    known = set()
    for file in [study.OUT/study.ext.DEV, study.OUT/'held.json', study.ext.old.OUT/'held.json']:
        known.update(study.ext.input_hash(f) for f in study.read(file).values())
    known.update(study.read(base.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for f in (base.OUT.parent/'cap_regional_once_20261120_01/cases.json',):
        known.update(r['hash'] for r in study.read(f).values())
    for p in (base.OUT.parent/'cap_regional_actor_iteration_20261120_01/panels').glob('*.json'):
        known.update(study.ext.input_hash(r['fixture']) for r in study.read(p))
    parents = []
    counts = {k: 0 for k in range(2, 6)}
    directions = [('x',1),('x',-1),('y',1),('y',-1)]
    for row in study.ext.old.rooms.generate(SEED)['cases']:
        n = row['movable_count']
        ordinal = counts[n]
        if ordinal == 24:
            continue
        axis, sign = directions[ordinal % 4]
        fixtures = []
        for cm in (0,1,2,5):
            f = copy.deepcopy(row['fixture'])
            f['robots'][0][axis] += sign * cm / 100
            fixtures.append(f)
        if not all(map(legal_shift, fixtures)):
            continue
        hashes = [study.ext.input_hash(f) for f in fixtures]
        assert len(set(hashes)) == 4 and not known.intersection(hashes)
        known.update(hashes)
        parents.append(dict(parent=row['id'], objects=n, axis=axis, sign=sign, fold=ordinal//4,
                            repeat=ordinal in (0,12), fixtures=fixtures, hashes=hashes))
        counts[n] += 1
    assert len(parents) == 96
    protocol = dict(seed=SEED, panel=study.ext.old.digest(parents), parents=96,
                    radius_cm=[0,1,2,5], repeats=8, workers=16, raw_actor_seed=17,
                    id_sha=study.sha(base.OUT/'fixed.pt'), action_sha=study.sha(hard.OUT/'actor.pt'),
                    binary=cpp.load()[1]['binary'], source=study.sha(__file__),
                    stages=['initial','first physical proposal disagreement','first publicly observed object contact along ID prefix'],
                    target='action actor continuation terminal success minus ID actor continuation terminal success at same prefix',
                    probe='6 folds held by parent, all radii; ridge lambda=number training rows after train-only scaling, public features only; predict action for positive score; assess on all and discordant, compare training-fold constant',
                    attribution='at contact and first disagreement also compare opening-only features on identical reached cases and suffix targets')
    assert protocol['id_sha'] == study.read(ppo.OUT/'protocol.json')['parent']
    assert protocol['action_sha'] == study.read(base.OUT/'protocol.json')['hard_model']
    assert protocol['binary'] == study.read(ppo.OUT/'cpp_backend.json')['binary']
    for name, value in [('panel',parents),('protocol',protocol)]:
        dest = OUT/(name+'.json')
        if dest.exists():
            assert study.read(dest) == value
        else:
            study.save(dest,value)
    return parents


def choices(panel):
    x,c,s,v,pool,raw = panel
    t = torch.from_numpy(x[None])
    logits = ID(t)[0].numpy()
    i = pool.index(raw[int(logits.argmax())]) if np.isfinite(logits).any() else 0
    scores = ACTION.values(t,torch.from_numpy(c[None]),torch.from_numpy(s[None]))[0].numpy()[:len(pool)]
    return i, int(scores.argmax()), logits, scores


def snapshot(env, lib, panel, decision, step):
    _, c, s, _, pool, _ = panel
    i,j,logits,scores = decision
    obs = study.dq.current_event(env.history)['observation']
    geom = lib.geometry
    g = geom.rows.get(geom.goal_id,np.full(12,np.nan))
    x = [step,obs['tick'],len(pool),float(i==j),*g,*np.ravel(geom.robot_xy)]
    for robot in obs['robots']:
        x.extend([robot['contacting_object'],robot['contacting_static'],robot['contacting_teammate'],
                  robot['own']['vx'],robot['own']['vy'],robot['own']['omega']])
    for idx in [i,j]:
        x.extend(c[idx])
        x.extend(s[idx])
    # Retain only observable quantities. Missing CAP logits get a finite sentinel plus mask in analysis.
    x.extend(np.where(np.isfinite(logits),logits,np.nan))
    x.extend([scores[i],scores[j]])
    return dict(step=step,tick=obs['tick'],prefix=env.history.sha256,
                features=[round(float(z),6) if np.isfinite(z) else None for z in x])


@torch.inference_mode()
def episode(f, switch=None, inspect=False, check=None):
    env = ppo.environment(f)
    lib = cpp.Proposals()
    stages = {}
    try:
        for step in range(300):
            panel = lib.panel(env.history)
            if check and step in check:
                assert env.history.sha256 == check[step]
            pool = panel[4]
            if inspect and (len(stages)<3):
                d = choices(panel)
                i,j = d[:2]
                obs = study.dq.current_event(env.history)['observation']
                names = []
                if step == 0:
                    names.append('initial')
                if 'divergence' not in stages and i != j:
                    names.append('divergence')
                if 'contact' not in stages and any(r['contacting_object'] for r in obs['robots']):
                    names.append('contact')
                if names:
                    rec = snapshot(env,lib,panel,d,step)
                    stages.update({name:rec for name in names})
                choice = i
            elif switch is not None and step >= switch:
                x,c,s,_,_,_ = panel
                scores = ACTION.values(*(torch.from_numpy(a[None]) for a in (x,c,s)))[0,:len(pool)]
                choice = int(scores.argmax())
            else:
                logits = ID(torch.from_numpy(panel[0][None]))[0]
                choice = pool.index(panel[5][int(logits.argmax())]) if logits.isfinite().any() else 0
            env.step(pool[choice],remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        if check:
            assert step >= max(check)
        return dict(success=int(env.env.success),ticks=env.env.total_periods,history=env.history.sha256,
                    steps=step+1,reason=env.env.termination or 'decision_budget'),stages
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def run_case(job):
    parent, idx = job
    cm = (0,1,2,5)[idx]
    path = OUT/'cases'/f"{parent['parent']}_{cm}.json"
    if path.exists():
        return study.read(path)['runs']
    f = parent['fixtures'][idx]
    baseline, stages = episode(f,inspect=True)
    divergence = stages.get('divergence')
    contact = stages.get('contact')
    runs = 1
    if divergence:
        action,_ = episode(f,switch=0,check={divergence['step']:divergence['prefix']})
        runs += 1
    else:
        action = baseline
    labels = dict(initial=action)
    if divergence:
        labels['divergence'] = action
    if contact:
        if not divergence or contact['step']<=divergence['step']:
            labels['contact'] = action
        else:
            labels['contact'],_ = episode(f,switch=contact['step'],check={contact['step']:contact['prefix']})
            runs += 1
    if parent['repeat'] and cm==0:
        both = [episode(f)[0],episode(f,switch=0)[0]]
        assert both == [baseline,action]
        runs += 2
    study.save(path,dict(parent=parent['parent'],cm=cm,objects=parent['objects'],fold=parent['fold'],
                         direction=[parent['axis'],parent['sign']],baseline=baseline,labels=labels,
                         stages=stages,runs=runs))
    return runs


def main():
    init()
    parents = freeze()
    jobs = [(p,i) for p in parents for i in range(4)]
    done = runs = 0
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        for future in as_completed([pool.submit(run_case,job) for job in jobs]):
            runs += future.result()
            done += 1
            study.save(OUT/'progress.json',dict(fixtures=done,total=len(jobs),rollouts=runs))
            if done%32==0:
                print(done,len(jobs),runs,flush=True)


if __name__=='__main__':
    main()
