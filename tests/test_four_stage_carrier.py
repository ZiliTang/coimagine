from pathlib import Path

import pytest

from experiments.planar_tasks.authoring_v2.four_stage import pilot_policy
from experiments.planar_tasks.authoring_v2.four_stage.carrier import build, verify, checked_boundary
from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary, observation
from experiments.planar_tasks.semantic_rl.typed_compiler import validate_source
from tools.cap_unknown_safety import fixture


def test_checked_carrier_contains_exact_decision_function_and_fits_limits():
    source = build(Path(pilot_policy.__file__).read_text())
    validate_source(source)
    program = verify(source)
    world, _, _ = replay_boundary(fixture(False), [])
    p = PublicFrontend().prepare(observation(world))
    boundary = checked_boundary(source, p, {"brake": True})
    assert boundary["memory"]["four_stage_values"]
    assert program.select(p["symbols"], lambda a: dict(allowed=a=="brake"))["selected"] == "brake"


def test_carrier_cannot_silently_bypass_gateway_permissions():
    source = build(Path(pilot_policy.__file__).read_text())
    changed = source.replace('permissions.get(name, False)', 'True')
    with pytest.raises(ValueError, match="carrier differs"):
        verify(changed)
