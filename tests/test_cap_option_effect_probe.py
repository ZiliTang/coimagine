import numpy as np

from experiments.planar_tasks.relational_search.motion import move_id
from tools.cap_option_effect_probe import effects, order


def measurement():
    return dict(target0=[0.,0.],target1=[.04,-.01],observed0=None,observed1=None,
                robots_delta=[[.05,.3],[-.10,-.2]])


def test_measures_motion_along_each_command_and_handles_opposing_pushes():
    info=dict(push_goal_progress=.3)
    cmd=dict(a0=move_id('E'),a1=move_id('S'),target_id=None,ticks=25)
    move=effects(cmd,'target',[1.,0.],measurement(),info)
    assert np.isclose(move['move'][0],.125)
    assert 'push' not in move
    # Equal east and west translation commands have no joint push axis.
    cmd.update(target_id='target',a0=3,a1=19)
    assert set(effects(cmd,'target',[1.,0.],measurement(),info))=={'goal'}
    cmd['a1']=44
    vals=effects(cmd,'target',[1.,0.],measurement(),info)
    assert np.isclose(vals['push'][0],.04) and vals['push'][2] is None
    assert np.isclose(vals['goal'][0],1-np.hypot(.96,.01))


def test_projection_ties_do_not_depend_on_action_order():
    assert order(1.,0.)==order(0.,-1.)==.5
    assert order(-1.,-.03)==1.
