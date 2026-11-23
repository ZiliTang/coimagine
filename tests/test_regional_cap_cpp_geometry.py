"""Differential tests of native helpers against the frozen CAP source helpers.

These tests do not execute environments, CAP policy contexts, or training.
"""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import random
import shutil
import subprocess
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIVE = ROOT / "outputs/semantic_rl/cap_regional_five_20261119_01/regional/semantics"
NEXT = ROOT / "outputs/semantic_rl/cap_regional_extension_20261119_01/author/semantics"
SOURCES = [FIVE / f"s{i:03}.py" for i in (4, 8, 11, 13, 19)] + [
    NEXT / f"s{i:03}.py" for i in (0, 1, 2, 5, 9)
]


@pytest.fixture(scope="module")
def native_geometry(tmp_path_factory):
    torch = pytest.importorskip("torch")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("regional_geometry")
    source = directory / "geometry_test.cpp"
    source.write_text(
        '#include "' + str(ROOT / "tools/regional_cap_cpp/geometry.hpp") + '"\n'
        'PYBIND11_MODULE(_regional_geometry_tests,m){'
        'm.def("norm",&rc::norm);m.def("estimate",&rc::estimate);'
        'm.def("obstacles",&rc::obstacles);m.def("route",&rc::route);'
        'm.def("move_action",&rc::move_action);m.def("round",&rc::geometry_round);'
        'm.def("hypot",&rc::geometry_hypot);}\n'
    )
    library = directory / ("_regional_geometry_tests" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(
        [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off",
         "-fno-builtin-sin", "-fno-builtin-cos", "-I" + sysconfig.get_path("include"),
         "-I" + str(Path(torch.__file__).parent / "include"), str(source), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    spec = importlib.util.spec_from_file_location("_regional_geometry_tests", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen_helpers():
    if not all(path.exists() for path in SOURCES):
        pytest.skip("frozen regional CAP source artifacts unavailable")
    result = []
    names = {"norm", "estimate", "obstacles", "route", "move_action", "hpush", "hpop", "initial_memory"}
    for path in SOURCES:
        tree = ast.parse(path.read_text())
        tree.body = [ast.Import(names=[ast.alias(name="math")])] + [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        ast.fix_missing_locations(tree)
        namespace = {}
        exec(compile(tree, str(path), "exec"), namespace)
        result.append(namespace)
    return result


def test_python_round_and_hypot(native_geometry):
    rng = random.Random(4320)
    for value in [i / 2 for i in range(-101, 102)]:
        assert native_geometry.round(value) == round(value)
    pairs = [(0., 0.), (-0., 0.), (1e-310, 2e-310), (1e308, 1e308), (3., 4.)]
    pairs += [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(20000)]
    for x, y in pairs:
        assert native_geometry.hypot(x, y) == math.hypot(x, y)
        denominator = max(math.hypot(x, y), 1e-9)
        assert native_geometry.norm(x, y) == [x / denominator, y / denominator]


def _observation(rng):
    robots = []
    for _ in range(2):
        origin = [rng.uniform(.3, 5.7), rng.uniform(.3, 5.7)]
        angles = [rng.uniform(-math.pi, math.pi) for _ in range(72)]
        patches = [dict(valid=rng.choice([True, False]), span_m=rng.uniform(0, 2),
                        tangent_world=[rng.uniform(-1, 1), rng.uniform(-1, 1)],
                        normal_world=[rng.uniform(-1, 1), rng.uniform(-1, 1)]) for _ in range(4)]
        scan = dict(origin=origin, ray_angles=angles, distances=[rng.uniform(0, 5) for _ in angles],
                    valid=[rng.choice([True, False]) for _ in angles],
                    category=[rng.choice([0, 1, 2, 3]) for _ in angles],
                    entity_ids=[rng.choice(["target", "other", None]) for _ in angles])
        robots.append(dict(own=dict(x=origin[0], y=origin[1]), scan=scan, patches=patches,
                           execution=dict(stage=rng.choice(["move_blocked", "move", "wait"]))))
    return dict(robots=robots, entities=[dict(id="target", points=[
        [rng.uniform(1, 3), rng.uniform(1, 3)] for _ in range(rng.choice([0, 2, 6, 10]))])],
        goal=dict(target_id="target"), active_target=rng.choice(["target", None]))


def test_all_helper_variants_exact(native_geometry, frozen_helpers):
    rng = random.Random(8403)
    for case in range(30):
        observation = _observation(rng)
        for cap_id, reference in enumerate(frozen_helpers):
            memory = reference["initial_memory"]()
            memory.update(ang=rng.uniform(0, math.pi / 2), dims=[rng.random(), rng.random()],
                          long_size=rng.random() * 2, short_size=rng.random() * .4)
            other_memory = copy.deepcopy(memory)
            assert native_geometry.estimate(cap_id, other_memory, observation) == reference["estimate"](memory, observation)
            assert other_memory == memory
            for exclude in [None, "target"]:
                expected = reference["obstacles"](observation, exclude)
                assert native_geometry.obstacles(cap_id, observation, exclude) == [list(p) for p in expected]
            points = reference["obstacles"](observation, None)
            start, goal = [[rng.uniform(0, 6), rng.uniform(0, 6)] for _ in range(2)]
            assert native_geometry.route(cap_id, start, goal, points, .3, [6, 6]) == list(reference["route"](start, goal, points, .3, [6, 6]))
            robot = observation["robots"][0]
            assert native_geometry.move_action(cap_id, robot, goal, points, [6, 6]) == reference["move_action"](robot, goal, points, [6, 6])


def test_route_lexicographic_ties_and_destination_relocation(native_geometry, frozen_helpers):
    cases = [
        ([.9, 2.7], [4.5, 2.7], [(2.7, y * .18) for y in range(10, 21)]),
        ([.9, .9], [4.5, 4.5], [(4.5, 4.5)]),
        ([.45, .81], [5.13, 4.95], [(2.7, 2.7), (2.7, 2.88)]),
        ([.9, .9], [4.5, 4.5], [(2.7, y * .18) for y in range(34)]),
    ]
    for cap_id, reference in enumerate(frozen_helpers):
        for start, goal, points in cases:
            assert native_geometry.route(cap_id, start, goal, points, .3, [6, 6]) == list(reference["route"](start, goal, points, .3, [6, 6]))
