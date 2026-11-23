import json
import math
from pathlib import Path
import subprocess

import pytest

from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import pose_kernel as pk


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    output = tmp_path_factory.mktemp('pose') / 'pose.so'
    subprocess.run(['gcc', '-std=c99', '-O2', '-shared', '-fPIC', '-fno-fast-math',
                    '-ffp-contract=off', str(Path(pk.__file__).with_name('pose_route.c')),
                    '-lm', '-o', str(output)], check=True)
    return pk.PoseKernel(output)


def test_start_already_at_goal_and_empty_points(kernel):
    start = 5*15 + 5
    stats, parents = kernel.search([3.6, 3.6], 15, 15, 0., [], start, 5., 5.)
    assert stats.tolist() == [start, start, 0, 0, -1]
    assert not parents


def test_exhausted_frontier_keeps_best_reached_node(kernel):
    # Footprint cannot fit in this arena at any orientation. The initial cell
    # alone is opened by the reference search.
    stats, parents = kernel.search([.48,.48], 2, 2, math.pi/7, [], 0, 1., 1.)
    assert stats[0] == -1 and stats[1] == 0
    assert stats[3] == 0 and stats[4] == -1
    assert not parents


def test_original_direction_and_parent_order(kernel):
    start = 5*15 + 5
    stats, parents = kernel.search([3.6, 3.6], 15, 15, 0., [], start, 8., 5.)
    assert stats[0] == 8*15+5
    assert parents == {8*15+5: 7*15+5, 7*15+5: 6*15+5, 6*15+5: start}
    assert stats[4] > 0


def test_reject_other_cap_before_loading_native_code(kernel):
    with pytest.raises(ValueError, match='exact CAP009'):
        pk.make_route('other source', kernel)


def test_incomplete_point_cannot_enter_native_kernel(kernel):
    with pytest.raises(ValueError, match='two coordinates'):
        kernel.search([3.6, 3.6], 15, 15, 0., [[1.0]], 0, 1., 1.)


def test_default_kernel_cache_reloads_and_checks_binary(tmp_path, monkeypatch):
    from experiments.planar_tasks.semantic_rl.program_trace_value_v1.pose_kernel import build

    monkeypatch.setattr(build, 'CACHE', tmp_path)
    build.default_kernel.cache_clear()
    try:
        build.default_kernel()
        binary, = tmp_path.glob('*/pose.so')
        # Change the expected hash in metadata, leaving the mapped binary intact.
        manifest = binary.with_name('manifest.json')
        metadata = json.loads(manifest.read_text())
        metadata['binary_sha256'] = ''
        manifest.write_text(json.dumps(metadata))
        build.default_kernel.cache_clear()
        with pytest.raises(ValueError, match='cache integrity'):
            build.default_kernel()
    finally:
        build.default_kernel.cache_clear()


def test_native_default_and_reference_switch(monkeypatch):
    from types import SimpleNamespace
    from hashlib import sha256
    from experiments.planar_tasks.semantic_rl.program_trace_value_v1.pose_kernel import build

    source = 'test pinned CAP'
    monkeypatch.setattr(pk, 'SOURCE_SHA', sha256(source.encode()).hexdigest())
    old, new = object(), object()
    module = SimpleNamespace(pose_route=old)
    monkeypatch.setattr(build, 'default_kernel', lambda: object())
    monkeypatch.setattr(pk, '_replacement', lambda *args: new)
    monkeypatch.delenv('COIMAGINE_CAP009_POSE', raising=False)
    pk.configure(module, 'different source')
    assert module.pose_route is old
    pk.configure(module, source)
    pk.configure(module, source)
    assert module.pose_route is new and module._cap009_reference_route is old
    monkeypatch.setenv('COIMAGINE_CAP009_POSE', 'reference')
    pk.configure(module, source)
    assert module.pose_route is old
