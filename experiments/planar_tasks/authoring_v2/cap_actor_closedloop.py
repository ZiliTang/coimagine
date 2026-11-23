"""Greedy same-panel evaluation of the frozen 64-wide single-teacher GA actors."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import cap_actor_distillation as actor
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save

BASE = actor.previous.BASE
CAPACITY = BASE / 'cap_actor_capacity_20261119_01'
EARLY = BASE / 'cap_actor_distillation_20261119_01'
OUT = BASE / 'cap_actor_closedloop_20261119_01'


def run():
    panel = actor.ARCHIVE / 'panel.json'
    cases = read(panel)['cases']
    ref = read(actor.PARENT / 'inputs.json')
    baseline = read(EARLY / 'protocol.json')
    assert sha(panel) == baseline['panel_sha256']
    assert sha(actor.PARENT / 'normalizers.npz') == baseline['normalizers_sha256']
    assert len(cases) == len({c['case_id'] for c in cases}) == 200
    frozen = read(CAPACITY / 'FROZEN.json')['files']
    jobs, weights = [], {}
    for seed in actor.SEEDS:
        name = f'single_s{seed}'
        checkpoint = CAPACITY / f'w64_s{seed}_u24000.pt'
        weights[checkpoint.name] = sha(checkpoint)
        assert weights[checkpoint.name] == frozen[checkpoint.name]
        jobs.extend(dict(condition=name, case=case, checkpoint=str(checkpoint),
                         directory=str(OUT / 'evaluation' / name / case['case_id']))
                    for case in cases)
    protocol = dict(panel_sha256=sha(panel), weights=weights, early_per_scene_sha256=sha(EARLY/'per_scene.csv'),
                    seeds=list(actor.SEEDS), workers=12, actor='GA single priority teacher 64x64',
                    updates=24000, epsilon=0, max_ticks=1800, max_decisions=300, wall_seconds=1800,
                    primary='paired 24k minus 6k success rate inside the seven-CAP union',
                    secondary='full panel, unique-CAP and outside-union success; paired scene changes')
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        assert read(OUT / 'protocol.json') == protocol
    save(OUT / 'protocol.json', protocol)
    progress = dict(completed=0, total=len(jobs))
    save(OUT / 'progress.json', progress)
    with ProcessPoolExecutor(12, mp_context=mp.get_context('spawn'), initializer=actor.fast_runtime.initialize,
                             initargs=(ref['bundle'], ref['priority_bundle'])) as pool:
        futures = [pool.submit(actor.evaluation_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0:
                progress['completed'] = done
                save(OUT / 'progress.json', progress)
                print(f'{done}/{len(jobs)} complete', flush=True)


if __name__ == '__main__':
    run()
