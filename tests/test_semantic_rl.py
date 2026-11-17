"""Semantic RL contracts: coverage, causal features, exact logp and real timing."""
from copy import deepcopy
import math
import pytest
import torch
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import action_space, canonical, History
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig, Detector
from experiments.planar_tasks.semantic_rl.learning import advantages, ppo_update, demonstration_traces

SOURCE = '''NAMES=['duration','recommended']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):return dict(duration=a['ticks'],recommended=float(a in p))
'''


def history():
    env = Environment(X.seed_fixture(), EventConfig(mode='time'))
    return env.history


def test_full_public_action_support_without_joint_scoring():
    h = history(); o = h.events[-1]['observation']; tree = ActionTree(o)
    leaves = set()
    for p in tree.choices({}):
        for q in tree.choices(p):
            for r in tree.choices(q):
                for a in tree.choices(r):
                    leaves.add(canonical(a))
    expected = {canonical(a) for a in action_space(o)}
    assert leaves == expected and len(leaves) == tree.count
    assert tree.count > 48
    # Source changes/teacher recommendations have no role in this tree.


def test_prefix_features_are_sandboxed_and_full_actions_still_legal():
    h = history(); h.events[-1]['observation']['secret_mass'] = 999
    source = SOURCE+'''
def prefix_semantic_features(m,o,p,recs):
 return dict(duration=len(p),recommended=o.get('secret_mass',0))
'''
    f = Features(source); tree = ActionTree(h.events[-1]['observation'])
    panel = f.panel(h, tree, {})
    assert all(r[:2] == [1, 0] for r in panel['semantic'])
    assert len(panel['choices']) == len(tree.menus)
    assert all('ticks' not in p for p in panel['choices'])


def test_conditional_logp_teacher_recompute_and_checkpoint(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(3)
    h = history(); f = Features(SOURCE); root = f.panel(h, ActionTree(h.events[-1]['observation']), {})
    model = Policy([root], f.sha256)
    teacher = dict(target_id=None,a0=44,a1=44,ticks=17)  # not recommended
    trace = model.decide(h, f, teacher=teacher)
    assert trace['action'] == teacher and not trace['recommended']
    assert trace['scored_prefixes'] < trace['legal_action_count']
    lp = sum(model.distribution(p)[0].log_prob(torch.tensor(i)) for p,i in zip(trace['panels'],trace['indices']))
    assert float(lp.detach()) == pytest.approx(trace['logp'])
    model.save(tmp_path/'policy.pt', {'test':True})
    restored, meta = Policy.load(tmp_path/'policy.pt', f.sha256)
    assert meta['test'] and float(restored.evaluate(trace)[0].detach()) == pytest.approx(trace['logp'])
    with pytest.raises(ValueError, match='incompatible'):
        Policy.load(tmp_path/'policy.pt', 'wrong')
    trace.update(periods=17,reward=-.85,terminal=False)
    update = ppo_update(model,torch.optim.Adam(model.parameters(),lr=1e-4),[trace],bootstrap=0.,epochs=1)
    assert update['logp_recompute_error'] < 2e-5 and update['updates'] == 1
    trace['logp'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        ppo_update(model,torch.optim.Adam(model.parameters()),[trace])


def test_variable_duration_gae_terminal_and_collection_cut():
    steps = [dict(value=2.,reward=-1.,periods=20,terminal=False),
             dict(value=3.,reward=-2.,periods=40,terminal=False)]
    targets, adv = advantages(steps, bootstrap=4.,trace_seconds=10.)
    assert adv[1] == pytest.approx(-1.)
    assert adv[0] == pytest.approx(-math.exp(-.1))
    steps[-1]['terminal'] = True
    _, terminal = advantages(steps, bootstrap=400.)
    assert terminal[-1] == pytest.approx(-5.)


def test_tempered_sampling_and_ppo_use_same_conditional_distribution():
    torch.set_num_threads(1); torch.manual_seed(3)
    h=history();f=Features(SOURCE);root=f.panel(h,ActionTree(h.events[-1]['observation']),{})
    model=Policy([root],f.sha256)
    trace=model.decide(h,f,temperature=2.,generator=torch.Generator().manual_seed(19))
    expected=sum(model.distribution(p,2.)[0].log_prob(torch.tensor(i)) for p,i in zip(trace['panels'],trace['indices']))
    assert trace['logp']==pytest.approx(float(expected.detach()))
    for p in trace['panels']:
        cold=model.distribution(p)[0];hot=model.distribution(p,2.)[0]
        assert hot.probs.sum().item()==pytest.approx(1.)
        assert hot.entropy().item()>=cold.entropy().item()-1e-6
        assert hot.logits.argmax()==cold.logits.argmax()
    trace.update(periods=10,reward=-.5,terminal=True)
    update=ppo_update(model,torch.optim.Adam(model.parameters(),lr=1e-4),[trace],epochs=1)
    assert update['logp_recompute_error']<2e-5 and update['updates']==1
    with pytest.raises(ValueError,match='temperature'):model.distribution(root,0.)


def test_time_adapter_preserves_existing_replay():
    commands = [dict(target_id=None,a0=44,a1=44,ticks=4)]*2
    expected = X.execute(X.seed_fixture(), commands)
    env = Environment(X.seed_fixture(),EventConfig(mode='time'))
    actual = [env.history.sha256]
    for a in commands:
        event = env.step(a); actual.append(env.history.sha256)
        assert event['periods'] == 4
    assert actual == [r['history_hash'] for r in expected['public_rows']]


def test_progress_handoff_and_elapsed_ticks():
    env = Environment(X.seed_fixture())
    action = dict(target_id='entity-b',a0=44,a1=10,ticks=40)
    first = env.step(action)
    assert first['reason'] == 'target_handoff' and first['periods'] == 1
    second = env.step(action, remaining_ticks=5)
    assert second['handoff']['commit'] and second['periods'] <= 5
    assert env.history.events[-1]['observation']['tick'] == first['periods']+second['periods']
    assert env.history.events[-1]['action'] == action


def test_progress_detector_uses_signed_public_odometry_and_latches():
    o = deepcopy(history().events[-1]['observation'])
    for r in o['robots']:
        r['contacting_object'] = True
        r['execution'] = dict(direction_valid=True,stage='push',done=False)
    d = Detector(EventConfig(contact_ticks=2,distance_m=.2))
    a = dict(target_id='entity-a',a0=2,a1=2,ticks=40)
    d.begin(a,o)
    assert d.update(deepcopy(o)) is None
    assert d.update(deepcopy(o)) == 'contact'
    d.begin(a,o)
    assert d.update(deepcopy(o)) is None  # no repeated contact event each decision
    n = deepcopy(o); n['robots'][0]['own']['x'] += .21
    assert d.update(n) == 'progress'
    d.begin(a,n)
    n = deepcopy(n)
    n['robots'][0]['own']['x'] -= .3
    assert d.update(deepcopy(n)) is None  # travel against commanded direction is not progress


def test_demo_rejects_failed_test_and_cross_timing():
    f = Features(SOURCE)
    with pytest.raises(ValueError, match='successful'):
        demonstration_traces({'summary':{'status':'task_failure'}},f,split='train',mode='time')
    with pytest.raises(ValueError, match='successful'):
        demonstration_traces({'summary':{'status':'success'}},f,split='test',mode='time')
    with pytest.raises(ValueError, match='mismatch'):
        demonstration_traces({'summary':{'status':'success'}},f,split='train',mode='progress')
    with pytest.raises(ValueError, match='successful'):
        demonstration_traces({'summary':{'status':'success'}, 'split':'test'},f,split='train',mode='time')
    with pytest.raises(ValueError, match='configuration mismatch'):
        demonstration_traces({'summary':{'status':'success'}, 'execution_mode':'progress',
                              'event_config':{'distance_m':999}},f,split='train',mode='progress')


def test_conditional_distribution_sums_to_one_and_permits_nonrecommendations():
    # Small public menu, exhaustively sum learned conditional joint probabilities.
    h = history(); o = h.events[-1]['observation']
    o['entities'] = []; o['entity_legal_pairs'] = {}
    o['legal_pairs'] = o['global_legal_pairs'] = [[44,44],[47,44],[47,47]]
    tree = ActionTree(o); f = Features(SOURCE)
    root = f.panel(h,tree,{})
    torch.set_num_threads(1); torch.manual_seed(3)
    model = Policy([root],f.sha256)
    probs = []
    def walk(prefix, probability):
        if len(prefix)==4:
            probs.append((prefix,probability)); return
        panel=f.panel(h,tree,prefix)
        with torch.no_grad(): dist,_=model.distribution(panel)
        for choice,p in zip(panel['choices'],dist.probs.tolist()):
            walk(choice,probability*p)
    walk({},1.)
    assert len(probs)==tree.count==120
    assert sum(p for _,p in probs)==pytest.approx(1.,abs=1e-6)
    assert all(p>0 for _,p in probs)
    assert any(a['a0']==47 and a['a1']==47 for a,_ in probs)
