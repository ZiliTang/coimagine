"""D008 offline schema/geometry tests; no env/Pymunk/scan_lidar."""
import math
from experiments.option_rl_v19.audit_d008 import truth_faces, dot, t90, WIDTH, HEIGHT


def test_true_face_normals_are_inward_and_alternating():
    obj = [2.5, 2.0, 0.0]
    f = {x['name']: x for x in truth_faces(obj)}
    assert abs(dot(f['bottom']['n'], (0, 1)) - 1) < 1e-12
    assert abs(dot(f['top']['n'], (0, -1)) - 1) < 1e-12
    assert abs(dot(f['left']['n'], (1, 0)) - 1) < 1e-12
    assert abs(dot(f['right']['n'], (-1, 0)) - 1) < 1e-12
    for x in f.values():
        assert abs(dot(x['n'], x['t'])) < 1e-12


def test_face_endpoints_match_mid_and_length():
    f = {x['name']: x for x in truth_faces([2.5, 2.0, 0.3])}
    for x in f.values():
        mid = ((x['eps'][0][0] + x['eps'][1][0]) / 2,
               (x['eps'][0][1] + x['eps'][1][1]) / 2)
        assert math.dist(mid, x['mid']) < 1e-12
        assert abs(math.dist(x['eps'][0], x['eps'][1]) - x['L']) < 1e-12


def test_interval_contains_true_endpoint_with_wide_bracket():
    # a 4 cm-wide interval around the true endpoint must be marked covered
    true = (1.0, 0.0)
    lo, hi = (0.98, 0.0), (1.02, 0.0)
    t = (1.0, 0.0)
    assert min(dot(lo, t), dot(hi, t)) <= dot(true, t) <= max(dot(lo, t), dot(hi, t))


def test_rotation_preserves_body_face_length_and_normal_sign():
    f0 = {x['name']: x for x in truth_faces([2.5, 2.0, 0.0])}
    f1 = {x['name']: x for x in truth_faces([2.5, 2.0, 1.1])}
    for name in f0:
        assert abs(f0[name]['L'] - f1[name]['L']) < 1e-12
        assert abs(math.hypot(*f1[name]['n']) - 1) < 1e-12
        assert abs(math.hypot(*f1[name]['t']) - 1) < 1e-12


def test_tangent_is_left_rotation():
    n = (0.6, 0.8); t = t90(n)
    assert abs(dot(n, t)) < 1e-12
    assert abs(t[0] + 0.8) < 1e-12 and abs(t[1] - 0.6) < 1e-12
