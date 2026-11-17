from copy import deepcopy
from itertools import islice
import pytest
from experiments.planar_tasks.relational_search.experience_candidates import ExperienceCandidates,descriptor
from experiments.planar_tasks.relational_search.contract import History,canonical,action_space
from experiments.planar_tasks.relational_search import live_search as L
from tests.test_relational_live import history,Model,BRAKE


def library(h,action):
 return ExperienceCandidates([dict(descriptor=descriptor(h).tolist(),action=action,provenance=dict(trajectory='test',boundary=999))])


def test_descriptor_rejects_truth_and_does_not_use_clock_or_answer_cursor():
 h=history();a=descriptor(h);other=deepcopy(h)
 for e in other.events:
  o=e['observation'];o['tick']+=100;o['decision']=987;o['truth_mass']=123
  for x in o['entities']:x['last_seen_tick']+=100
  for r in o['robots']:
   for key in ('tick','time','elapsed_seconds'):r[key]=999
   r['remaining_seconds']=1;r['scan']['tick']=999;r['scan']['sample_time']=999
 assert descriptor(other)==pytest.approx(a)
 db=library(h,BRAKE);assert db.retrieve(other)[0]['action']==BRAKE
 assert db.retrieve(other)[0]['distance']==pytest.approx(0)


def test_invalid_experience_does_not_override_public_menu():
 h=history();db=library(h,dict(BRAKE,target_id='not-visible'))
 assert db.retrieve(h)==[]


def test_experience_scored_and_expanded_without_changing_scores_or_budget():
 h=history();a=dict(BRAKE,ticks=7);db=library(h,a)
 b=L.Budget(calls=2,prediction_ticks=80,batch=72)
 result=L.choose(h,Model(h),lambda h,aa:[float(x['ticks']) for x in aa],b,experience=db,reserved_root_calls=1)
 assert result['scoring'][0]['actions'][0]==a
 assert result['scoring'][0]['scores'][0]==7
 assert result['trace'][0]['action']==a and result['trace'][0]['reserved_experience']
 assert result['calls']==2 and result['prediction_ticks']<=80
 assert result['trace'][0]['actual_ticks']==7
