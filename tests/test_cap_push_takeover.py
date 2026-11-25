from types import SimpleNamespace

from experiments.planar_tasks.semantic_rl.push_takeover import push_checkpoints
from tools.cap_cluster_module import BranchMember


def test_joint_push_checkpoints_merge_simultaneous_progress():
    obs = {'tick': 0}
    world = SimpleNamespace(env=SimpleNamespace(public=lambda: obs, active_target='box'),decision_count=1)
    reviews = []
    def decline(observation, action, left, reason):
        reviews.append((observation['tick'],reason))
    action = dict(target_id='box', a0=3, a1=3, ticks=30)
    review = push_checkpoints(world,action,decline,progress_m=.1,max_decisions=300)
    def move(tick, x, touch=True):
        obs['tick']=tick
        r=dict(own=dict(x=x,y=0.),contacting_object=touch,execution=dict(direction_valid=True,stage='push'))
        assert review(dict(robots=[r,r]),30-tick) is None
    move(1,0.)
    move(2,.06)
    move(3,.11)
    move(4,.13,False)
    move(5,.9)
    world.decision_count=299
    move(6,1.2)
    assert reviews==[(1,'contact'),(3,'progress'),(5,'contact')]


def test_module_preview_can_defer_with_different_parent_proposal():
    replies=[['parent','fallback'], ['module','fallback','parent']]
    events=[]
    def request(packet):
        events.append(packet['event'])
        return dict(proposals=replies.pop(0),memory={'changed':True})
    m=BranchMember(SimpleNamespace(request=request),'full',None)
    m.memory={'original':True}
    m.input_event={'observation':{'tick':10}}
    for expected in (None,'module'):
        assert m.push_checkpoint({'tick':14},{'ticks':10},6,'contact')==expected
    assert m.memory=={'original':True}
    assert events[-1]['executed_ticks']==4
    assert events[-1]['push_checkpoint']=={'reason':'contact','remaining_ticks':6}
