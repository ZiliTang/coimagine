from dataclasses import replace
import math
import numpy as np
import pytest
from experiments.option_rl_v12 import controller as C
from experiments.option_rl_v12.mode_env import ContinuousModeEnv
from experiments.option_testbench.effect_push_eval import make_case
from experiments.option_testbench.follow_peer_eval import instance_from_dict

def instance():return instance_from_dict(make_case('continuous_single_N',distance=.3,offset=.18)['instance'])
def push(v=.16):return next(a.index for a in C.ACTIONS if a.kind=='push' and a.type_index==2 and a.speed==v)

def test_repeat_and_speed_update_keep_controller_memory_and_fixed_ticks():
 w=ContinuousModeEnv(instance());a=push()
 w.joint_step(a,C.STOP);s=w.slots[0].skill
 for _ in range(8):
  b=w.joint_step(a,C.STOP);assert b['periods']==4 and w.slots[0].skill is s
 tracker=s.endpoint_tracker;ref=s.reference.value
 w.joint_step(push(.32),C.STOP)
 assert w.slots[0].skill is s and s.endpoint_tracker is tracker and s.skill_time>1.8 and s.speed_command==.32
 assert math.hypot(*ref)>0

def test_decision_frequency_and_recording_do_not_change_constant_command_physics():
 a=ContinuousModeEnv(instance(),decision_hz=5);b=ContinuousModeEnv(instance(),decision_hz=10,collect_dense=True)
 for _ in range(20):
  a.joint_step(push(),C.STOP,record=False)
  for _ in range(2):b.joint_step(push(),C.STOP,record=True)
 assert a.object_state()==b.object_state() and a.robot_states()==b.robot_states()
 assert a.reward_total==pytest.approx(b.reward_total,abs=1e-10)
 np.testing.assert_array_equal(a.boundary_vector(),b.boundary_vector())
 assert len(b.dense_samples)==40 and all(x['dt']==.1 for x in b.dense_samples)
 assert len(a.boundary_vector())==len(a.observation_fields())==822

def test_stop_brakes_without_resetting_velocity_or_exceeding_reference_acceleration():
 w=ContinuousModeEnv(instance());w.joint_step(push(.32),C.STOP,record=True)
 for _ in range(24):w.joint_step(push(.32),C.STOP,record=True)
 speed=math.hypot(w.robot_states()[0].vx,w.robot_states()[0].vy);assert speed>.03
 before=w.robot_states()[0]
 w.joint_step(C.STOP,C.STOP,record=True)
 assert math.dist(before.as_tuple()[:2],w.robot_states()[0].as_tuple()[:2])>0
 for _ in range(10):w.joint_step(C.STOP,C.STOP,record=True)
 assert math.hypot(w.robot_states()[0].vx,w.robot_states()[0].vy)<.03
 refs=[r['r0']['reference_velocity'] for r in w.rows]
 assert max(math.dist(a,b) for a,b in zip([(0,0)]+refs,refs))<=.5*.05+1e-8
 assert not any(r['penetration']>.002 for r in w.rows)

def test_full_budget_failure_cost_and_follow_legality():
 w=ContinuousModeEnv(instance());bad=C.FOLLOW
 with pytest.raises(ValueError):w.joint_step(bad,C.STOP)
 assert w.total_periods==0
 # Same physical cost over a short diagnostic horizon; no option termination bonus.
 w.config=replace(w.config,time_budget=.4)
 w.joint_step(C.STOP,C.STOP);b=w.joint_step(C.STOP,C.STOP)
 assert b['termination']=='task_budget_exhausted'
 assert w.reward_components['time_cost']==pytest.approx(-.016)
 assert w.reward_components['failure_penalty']==-2

def test_continuous_reposition_has_no_distance_goal_and_no_fake_branch_snapshot():
 w=ContinuousModeEnv(instance());rp=next(a.index for a in C.ACTIONS if a.kind=='reposition' and a.direction=='CCW' and a.speed==.12)
 w.joint_step(rp,C.STOP);s=w.slots[0].skill
 for _ in range(30):w.joint_step(rp,C.STOP)
 assert w.slots[0].skill is s and s.exit_reason!='reposition_complete' and s.action.scale_kind=='tangent_speed'
 with pytest.raises(NotImplementedError):w.export_branch_state()

def test_invisible_object_still_allows_brake_and_updates_are_causal():
 from dataclasses import replace
 w=ContinuousModeEnv(instance(),collect_dense=True);o=w.boundary_observations()[0];blind=replace(o,patch=None,patches=())
 mask=C.mask(blind);assert mask[C.STOP] and mask[C.FOLLOW] and not mask[:38].any()
 before=w.boundary_vector().copy();w.joint_step(push(),C.STOP)
 np.testing.assert_array_equal(w.dense_samples[0]['observation'],before)
