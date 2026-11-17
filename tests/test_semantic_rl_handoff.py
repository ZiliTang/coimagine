from copy import deepcopy
from test_semantic_rl_stream import SOURCE,COUNT
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture


def run(source,**kwargs):
 f=Features(source,action_protocol=protocol(EventConfig(mode='duration')))
 try:
  result,_=rollout(seed_fixture(),source,features=f,config=EventConfig(mode='duration'),max_ticks=80,max_decisions=4,fast_feedback=True,**kwargs)
  state=deepcopy(f.worker.memory)
  return result,state
 finally:f.close()


def test_prefix_identity_and_full_horizon_accounting():
 a,_=run(COUNT);prefix=[r['action'] for r in a['public_rows'][1:3]];b,m=run(COUNT,prefix=prefix)
 assert a['summary']['error'] is None and b['summary']['error'] is None
 assert a['public_rows']==b['public_rows'] and a['replay']['frames']==b['replay']['frames']
 assert b['summary']['ticks']==80 and b['prefix_replay']['ticks']==40
 assert b['prefix_replay']['suffix_decisions']==2 and m['count']==4


def test_receiver_ingests_actual_foreign_actions():
 foreign=[dict(target_id=None,a0=44,a1=44,ticks=7)]*2
 r,m=run(COUNT,prefix=foreign)
 assert r['summary']['error'] is None
 assert r['summary']['ticks']==54 and m['ticks']==34 and m['count']==4
 assert r['public_rows'][1]['action']['ticks']==7 and r['public_rows'][3]['action']['ticks']==20


def test_illegal_prefix_preserved_as_failure():
 r,_=run(SOURCE,prefix=[dict(target_id='unobserved_entity',a0=44,a1=44,ticks=20)])
 assert r['summary']['error'] is not None
 assert r['summary']['ticks']==0 and not r['prefix_replay']['complete']


def test_replay_comparison_uses_serialized_representation():
 from experiments.planar_tasks.authoring_v2.handoff_entry_study import serialized_equal
 assert serialized_equal([{'xy':(1.,2.)}],[{'xy':[1.,2.]}])
 assert not serialized_equal([{'xy':(1.,2.)}],[{'xy':[1.,2.1]}])
