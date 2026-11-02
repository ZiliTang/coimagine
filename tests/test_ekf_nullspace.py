import numpy as np
from experiments.option_rl_v19.unknown_shape import RigidMotionEKF


def test_single_face_zeros_tangent():
    e=RigidMotionEKF()
    e.x=np.array([.20,.30,.01])
    e.P=np.eye(3)*.1
    e.apply_nullspace_prior([[1.,0.,0.],[0.,0.,1.]])
    assert abs(e.x[1]) < 1e-9
    assert abs(e.x[0]-.20) < 1e-9


def test_nonparallel_faces_keep_both_components():
    e=RigidMotionEKF()
    e.x=np.array([.20,.30,.01])
    before=e.x.copy()
    e.apply_nullspace_prior([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    assert np.allclose(e.x,before)
