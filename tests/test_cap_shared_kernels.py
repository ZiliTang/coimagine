import math

import pytest

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.pose_kernel.shared import kernels


def test_clearances_distinguishes_target_and_robot_points():
    robots = [{'own': {'x': x, 'y': 1.}} for x in (1., 4.)]
    geom = (1., 1., 1., 0.)
    target = 'box'
    point = (1.5, 1., target, 2)
    push, moves = kernels().clearances(robots, geom, target, [point])
    assert push == [2.] * 8
    assert moves[0] == .5 - .2
    assert len(moves) == 16
    push, moves = kernels().clearances(robots, geom, target, [(1.5, 1., 'peer', 3)])
    assert push[0] == 0.
    assert moves[0] == .5 - .38


def plan(pos, goal, points=(), extent=2.4):
    size = int(extent/.12)
    return kernels().plan((extent, extent), size, size, pos, (8., 8.),
                          (8., 8., 1., 0.), points, goal)


def test_navigate_straight_path_and_exact_goal_tie():
    extent = 2.4
    ny = int(extent/.12)
    start = (4.5*.12, 4.5*.12)
    # The midpoint ties exactly in binary64 for these two cell centers.
    x1, x2 = 11.5*.12, 12.5*.12
    goal_x = (x1+x2)/2
    assert goal_x-x1 == x2-goal_x
    _, end, found, parents, _ = plan(start, (goal_x, start[1]))
    assert found == end == 11*ny+4
    assert parents == {i*ny+4: (i-1)*ny+4 for i in range(5, 12)}


def test_navigate_selects_end_before_clearing_start():
    pos = (6.5*.12, 6.5*.12)
    start = 6*20+6
    blocked, end, found, _, _ = plan(pos, pos, [pos])
    assert start not in blocked
    assert end != start and found == end


def test_navigate_retains_strict_obstacle_boundary():
    # Use nextafter to place points on opposite sides of the exact .205 gate.
    cell = (5.5*.12, 5.5*.12)
    pos = (1.6, 1.6)
    x = cell[0] + .205
    for measured_x in (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf)):
        blocked, *_ = plan(pos, pos, [(measured_x, cell[1])])
        assert (5*20+5 in blocked) == (math.hypot(measured_x-cell[0], 0.) < .205)


def test_navigate_exhausted_and_no_free_cells():
    start, goal = (.5, 1.2), (1.85, 1.2)
    wall = [(1.2, j*.12) for j in range(21)]
    _, _, found, parents, stats = plan(start, goal, wall)
    assert found is None and parents == {} and stats[2] == 0
    with pytest.raises(ValueError, match='empty sequence'):
        plan(start, goal, extent=.24)
