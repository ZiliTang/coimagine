from copy import deepcopy
import pytest
from experiments.planar_tasks.authoring_v2.da_joint_study import audit_prefix


def reference():
    return dict(public_rows=[dict(action=None,observation=dict(tick=0)),
        dict(action={'ticks':4},observation=dict(tick=4)),
        dict(action={'ticks':4},observation=dict(tick=8))],
        replay=dict(frames=[dict(tick=t,x=t) for t in range(9)]))


def test_branch_audit_checks_physics_and_public_prefix_only():
    a=reference();b=deepcopy(a);b['prefix_replay']=dict(executed=2)
    b['public_rows'][2]['action']={'ticks':5};b['replay']['frames'][-1]['x']=99
    assert audit_prefix(b,a,1)['physics_equal']
    b['replay']['frames'][3]['x']=99
    with pytest.raises(ValueError,match='physics'):audit_prefix(b,a,1)


def test_branch_audit_rejects_incomplete_and_changed_observation():
    a=reference();b=deepcopy(a);b['prefix_replay']=dict(executed=0)
    with pytest.raises(ValueError,match='incomplete'):audit_prefix(b,a,1)
    b['prefix_replay']['executed']=1;b['public_rows'][1]['observation']['tick']=3
    with pytest.raises(ValueError,match='public'):audit_prefix(b,a,1)
