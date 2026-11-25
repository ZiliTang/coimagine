from experiments.planar_tasks.authoring_v2.four_stage.physical import command_and_stop, process_bounds
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary
from tools.cap_unknown_safety import fixture


def test_actual_tick_positions_and_full_braking_are_audited():
    world, _, _ = replay_boundary(fixture(False), [])
    result = command_and_stop(world, dict(target_id=None, a0=48, a1=44, ticks=4))
    root = result["root"]
    assert len(root["samples"]) == root["event"]["periods"]
    assert root["process"]["r0"][1] > 0
    assert len({tuple(s["robots"][0]) for s in root["samples"]}) > 1
    assert result["stop_ticks"] == 40
    assert result["process"]["r0"][1] >= root["process"]["r0"][1]
    assert result["stopped"]
    assert not result["full_target_static"]
    assert not result["full_robot_static"]


def test_turning_target_boundary_is_not_replaced_by_centre_displacement():
    start = dict(robots=[(0, 0), (1, 1)], target_corners=[(0, 0), (1, 0)])
    turned = dict(robots=[(0, 0), (1, 1)], target_corners=[(0, 0), (0, 1)])
    assert process_bounds(start, [turned])["target_boundary"] > 1.4
