from copy import deepcopy

from tools.cap_symbolic_general_evaluate import gate_action, inside, metric, same_public
from experiments.planar_tasks.authoring_v2.symbolic_general import scaffold, frontend, simulation
from tools.cap_symbolic_general_runtime import Controller
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture


def test_fixed_candidate_gate_matches_runtime_first_selected_action():
    world,memory,_=simulation.replay_boundary(seed_fixture(),[])
    observation=simulation.observation(world);z,commands=frontend.semantic_packet(memory,observation)
    policy=scaffold.verify_source(scaffold.source(scaffold.starter()))
    action=policy.select(z)['option'];assert action!='brake'
    class Model:
        sha256='test'
        def predict(self,*args):
            return dict(samples=[],supported=False,calibrated=False,local_rooms=0)
        def interval_outcomes(self,*args):return []
    class Rollout:
        def __init__(self):self.ledger=dict(branches=0,ticks=0,models=0,incompatible=0)
        def run(self,*args):
            self.ledger['branches']+=2
            return dict(samples=[dict(values=deepcopy(z),path_safe=True)]*2)
    direct=gate_action(policy,Model(),Rollout(),memory,observation,z,commands,action,2)
    controller=Controller(Model(),rollout=Rollout())
    runtime=controller.decide(memory,observation,z,commands,policy)
    assert direct['action']==runtime['checks'][0]['action']
    assert direct['allowed']==runtime['permissions'][action]['permit'] is True
    assert direct['source']==runtime['permissions'][action]['source']
    assert direct['rollout_delta']['branches']==controller.rollout.ledger['branches']==2


def test_joint_coverage_does_not_mix_distinct_modes():
    outcomes=[dict(updates=dict(visible=True,distance=[0,1])),
              dict(updates=dict(visible=False,distance=[2,3]))]
    assert inside(dict(visible=True,distance=.5),outcomes)
    assert not inside(dict(visible=True,distance=2.5),outcomes)
    outcomes[0]['path_safe']=True
    assert not inside(dict(visible=True,distance=.5),outcomes,path_safe=False)


def test_nominal_rollout_cannot_erase_unsafe_predicted_mode():
    world,memory,_=simulation.replay_boundary(seed_fixture(),[])
    observation=simulation.observation(world);z,commands=frontend.semantic_packet(memory,observation)
    policy=scaffold.verify_source(scaffold.source(scaffold.starter()));action=policy.select(z)['option']
    class Model:
        def predict(self,*args):
            return dict(samples=[dict(values=z,path_safe=False)],supported=False,calibrated=False,local_rooms=1)
        def interval_outcomes(self,*args):
            return [dict(updates={k:v if type(v)is bool else [v,v]for k,v in z.items()},path_safe=False)]
    class Rollout:
        ledger=dict(branches=0,ticks=0,models=0,incompatible=0)
        def run(self,*args):raise AssertionError('an extra nominal safe hypothesis cannot disprove this unsafe mode')
    result=gate_action(policy,Model(),Rollout(),memory,observation,z,commands,action,2)
    assert result['source']=='unresolved_predicted_mode'
    assert not result['allowed']and result['queries']==0


def test_measured_successor_uses_runtime_permissions_not_ungated_policy():
    fixture=seed_fixture();world,memory,_=simulation.replay_boundary(fixture,[])
    z,_=frontend.semantic_packet(memory,simulation.observation(world))
    source=scaffold.source(scaffold.starter());policy=scaffold.verify_source(source)
    action=policy.select(z)['option'];assert action!='brake'
    class StopController:
        def decide(self,memory,observation,z,commands,policy):
            return dict(permissions={name:dict(permit=name=='brake')for name in commands})
    measured=simulation.branch(fixture,[],action,source,controller=StopController(),retain_root_public=True)
    assert measured['successor']['option']=='brake'
    assert measured['successor_completed']is False
    assert measured['next_semantic']['permit:'+action]is True
    assert measured['successor_semantic']['permit:'+action]is False
    assert same_public(simulation.observation(world),measured['root_public'])


def test_coverage_failure_and_safety_have_distinct_denominators():
    template=dict(error=None,joint_set_covered=None,online_branches=0,online_ticks=0,
                  source='model',root_static=False,actual_contract_valid=True)
    rows=[dict(template,admitted=True,actual_completed=True,safe_effective=False,full_static=True),
          dict(template,admitted=False,actual_completed=True,safe_effective=True,full_static=False),
          dict(template,admitted=True,actual_completed=False,safe_effective=False,full_static=False)]
    result=metric(rows,1)
    assert result['C']==0 and result['F']==.5
    assert result['admitted_static']==1 and result['rejected_effective']==1
    assert metric([rows[1]],1)['F']is None
