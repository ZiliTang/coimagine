"""Live metrics, incomplete validation and conservative unattended stop decisions."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from experiments.option_rl_v10 import train
from experiments.option_rl_v10.live import atomic_json,read_json,status
from experiments.option_rl_v10.monitor import CHARTS,write_dashboard
from experiments.option_rl_v10.supervise import fault


def test_dashboard_contains_all_key_metrics_and_refreshes_atomically(tmp_path):
    status(tmp_path,phase='validation',iteration=2,eval_done=3,eval_total=64)
    keys={k for _,series in CHARTS for k,_,_ in series}
    required={'explore_return','success_rate','explore_progress','mean_progress','policy_loss','value_loss',
              'entropy','train_anomaly_rate','eval_anomaly_rate','actor_trunk_grad_norm','critic_trunk_grad_norm',
              'action_reposition','action_translate','initial_logp_max_error'}
    assert required<=keys
    history=[dict(iteration=1,explore_return=-3.,success_rate=None)]
    path=write_dashboard(history,tmp_path/'dashboard.html',subtitle='<unsafe>')
    text=path.read_text()
    assert 'http-equiv="refresh" content="10"' in text and '验证 3/64' in text
    assert '&lt;unsafe&gt;' in text and '<unsafe>' not in text
    assert 'ctx.arc(x,y,2.7' in text # first real validation point is visible
    history.append(dict(iteration=2,explore_return=-2.,success_rate=.1));write_dashboard(history,path)
    assert '"iteration": 2' in path.read_text()
    assert not list(tmp_path.glob('*.tmp'))


def test_zero_early_success_does_not_stop_a_healthy_run():
    rows=[dict(iteration=i,success_rate=0.,mean_progress=-.2,entropy=4.,train_anomaly_rate=0.,eval_anomaly_rate=0.) for i in range(15)]
    assert fault(rows,{'updated_unix':time.time()}) is None


def test_fault_stops_have_actual_evidence():
    assert fault([{'value_loss':float('nan')}],{})=='nonfinite_history'
    assert fault([],{'updated_unix':0},now=301)=='heartbeat_stale_over_300s'
    assert fault([{'train_anomaly_rate':.3}]*3,{})=='train_anomaly_rate_ge_25pct_three_iterations'
    assert fault([{'success_rate':0.,'eval_anomaly_rate':.2}]*2,{})=='validation_anomaly_rate_ge_10pct_twice'
    assert fault([{'approx_kl':.8}]*3,{})=='ppo_kl_over_0_5_three_iterations'


def test_partial_validation_is_not_reported_as_complete_success(monkeypatch):
    calls=[]
    def rollout(*args,**kwargs):
        calls.append(1)
        return [],dict(success=True,progress=.5,return_=1.,initial_distance=1.,eligible_for_ppo=True,**{'return':1.})
    monkeypatch.setattr(train,'run_episode',rollout)
    result=train.evaluate(None,None,[1,2,3],should_stop=lambda:len(calls)>=1)
    assert not result['complete'] and result['completed_roots']==1 and result['expected_roots']==3
    assert result['success_rate'] is result['mean_progress'] is None


def test_requested_episode_count_is_not_maximum_retry_count(tmp_path,monkeypatch):
    calls=[]
    def fake_episode(policy,instance,normalizer,**kwargs):
        calls.append(instance.index)
        s=dict(name=instance.name,initial_distance=1.,success=False,progress=0.,eligible_for_ppo=True,
               components=dict(progress=0.,distance_cost=-.5,time_cost=-.5,success_bonus=0.,failure_penalty=-2.),
               skill_exits={'wait_complete':2},actions=[dict(a0=60,a1=60,time=90.)],**{'return':-3.})
        return [None],s
    monkeypatch.setattr(train,'run_episode',fake_episode)
    monkeypatch.setattr(train,'ppo_update',lambda *a,**k:{'updates':0})
    result=train.train(out=tmp_path/'run',iterations=1,episodes=2,eval_roots=0,epochs=1,hours=.01,hidden=8)
    history=read_json(tmp_path/'run/history.json')
    assert calls==[0,1] and result['next_scene_ordinal']==2
    assert history[0]['attempted_episodes']==history[0]['valid_episodes']==2
