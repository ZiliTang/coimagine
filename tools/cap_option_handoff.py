"""One untrimmed CAP proposal per edge, paired at identical histories."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp

from tools import cap_value_dag_data as previous
from tools import cap_topology_graph_report as graph

g = previous.g
s = g.study
OUT = g.OUT.parent / 'cap_option_handoff_20261123_01'
DOC = g.DOC.parent / '2026-11-23_cap_option_handoff'
SEED = 662316037
ARMS = ('AA', 'AB', 'BA', 'BB')
read, save = s.read, s.save


def panel(fold):
    old = read(previous.OUT / 'panels.json')
    if fold == 'dev':
        rooms = read(s.OUT / 'development.json')
        rooms = {k: rooms[k] for k in old['rooms']['dev']}
        assert {k:s.phase.input_hash(f) for k,f in rooms.items()} == old['rooms']['dev']
    else:
        rooms = {r['id']:r['fixture'] for r in s.phase.cap.rooms.generate(SEED)['cases']}
        hashes = {s.phase.input_hash(f) for f in rooms.values()}
        seen = s.collect_identities()
        for values in old['rooms'].values():
            seen.update(values.values())
        for name in ('development', 'held'):
            seen.update(s.phase.input_hash(f) for f in read(s.OUT / (name+'.json')).values())
        for seed in (662315862, 662315879, g.prep.SEED):
            seen.update(s.phase.input_hash(r['fixture']) for r in s.phase.cap.rooms.generate(seed)['cases'])
        assert len(hashes) == 200 and not hashes & seen
    return rooms, old['pair']



def public_flags(features):
    conditions = {
        'touch': any(features[6:8]),
        'close': features[9] and features[10] <= .45,
        'progress': features[23] and features[24] >= .02,
        'clear': features[9] and features[11] >= .16,
    }
    return [name for name,active in conditions.items() if active]


def state(w, obs, hist):
    physical, gap, _ = g.topology(w, obs)
    public = g.public_vector(obs, hist)
    _, node, _ = PREDICT([dict(x=public)])
    votes = public_flags(public)
    return dict(h=w.history.sha256, tick=obs['tick'], decision=w.decision_count,
                flags=votes, node=int(node[0]), bits=physical, gap=gap)


def run_arm(fixture, pair, depth, arm):
    policies = [s.phase.FrozenCAP(s.MODULES[k]) for k in pair]
    counts = Counter()
    hist = deque(maxlen=4)
    fork = None
    edge = []
    mid = root = None
    with s.phase.world(fixture) as (w,_):
        def update(history):
            for p in policies:
                p.update(history)
        update(w.history)
        n = 0
        while not s.ended(w):
            obs = s.observe(w)
            search = arm != 'singleB' and fork is None and n >= depth
            step = n-fork if fork is not None else None
            if arm != 'singleB' and (fork is None or step == 1):
                hist.append(g.live_frame(obs))
            idx = int(arm == 'singleB')
            if step is not None and step < 2:
                idx = int(arm[step]=='B')
            proposal = g.prep.prior.try_action(policies[idx],obs)
            other = None
            if search:
                other = g.prep.prior.try_action(policies[1],obs)
                if proposal is not None and other is not None and proposal != other:
                    fork, step = n, 0
                    root = state(w,obs,hist)
                    idx = int(arm[0]=='B')
                    proposal = other if idx else proposal
            if step == 1:
                mid = state(w,obs,hist)
                other = g.prep.prior.try_action(policies[1-idx],obs)
                mid['distinct'] = proposal is not None and other is not None and proposal != other
            valid = proposal is not None
            if not valid:
                proposal = g.prep.prior.try_action(policies[1-idx],obs) or g.prep.BRAKE
            if step in (0,1):
                before = w.env.total_periods
                target = w.env.primary_id
                goal = obs['goal']['xy']
                gap = g.prep.goal_gap(w,target,goal)
            s.tick_action(w,proposal,g.prep.prior.EXECUTOR,update,counts)
            if step in (0,1):
                edge.append(dict(h=w.history.sha256, action=proposal, legal=valid,
                                 ticks=w.env.total_periods-before,
                                 gain=gap-g.prep.goal_gap(w,target,goal)))
            n += 1
        result = s.output(w,counts)
    result.pop('counts')
    return dict(root=root,mid=mid,edges=edge,result=result)


def room(job):
    fold, key, fixture, pair, depth = job
    path = OUT / fold / (key+'.json')
    if path.exists():
        return
    arms = {}
    for tag in ARMS:
        ref = arms.get(tag[0]+'A')
        if ref is not None and ref['mid'] is None:
            arms[tag] = ref
            continue
        arms[tag] = run_arm(fixture,pair,depth,tag)
        if arms[tag]['root'] is None:
            arms = dict.fromkeys(ARMS,arms[tag])
            break
    assert all(a['root']==arms['AA']['root'] for a in arms.values())
    for letter in 'AB':
        a,b = arms[letter+'A'],arms[letter+'B']
        assert a['mid']==b['mid'] and a['edges'][:1]==b['edges'][:1]
        if a['mid'] and a['edges'][1]['action']==b['edges'][1]['action']:
            assert a['result']['history']==b['result']['history']
    single_b = run_arm(fixture,pair,depth,'singleB')['result']
    save(path,dict(key=key,identity=s.phase.input_hash(fixture),depth=depth,arms=arms,single_b=single_b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fold',choices=('dev','fresh'))
    p.add_argument('--workers',type=int,default=18)
    args = p.parse_args()
    rooms,pair = panel(args.fold)
    if args.fold == 'fresh':
        assert (OUT / 'frozen.json').is_file()
    keys = sorted(rooms,key=lambda k:hashlib.sha256(('option-chain/'+k).encode()).digest())
    save(OUT/(args.fold+'_panel.json'),{k:s.phase.input_hash(rooms[k]) for k in keys})
    s.initialize()
    global PREDICT
    _, PREDICT = graph.frozen_model()
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('fork')) as pool:
        jobs = [pool.submit(room,(args.fold,k,rooms[k],pair,(0,8,24)[i%3])) for i,k in enumerate(keys)]
        for i,job in enumerate(as_completed(jobs),1):
            job.result()
            if i%20==0:
                print(args.fold,i,flush=True)


if __name__=='__main__':
    main()
