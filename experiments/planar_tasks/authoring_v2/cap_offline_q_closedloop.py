"""Evaluate frozen offline GA-Q on the single-teacher actor retention panel."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import cap_actor_closedloop as baseline
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save

PARENT = baseline.actor.PARENT
OUT = baseline.BASE / 'cap_offline_q_closedloop_20261119_01'
PANEL = baseline.actor.ARCHIVE / 'panel.json'
SEEDS = baseline.actor.SEEDS


def checkpoint(seed):
    return PARENT / 'models' / f'GA_s{seed}' / 'offline.pt'


def run():
    prior = read(baseline.OUT / 'FINISHED.json')
    cases = read(PANEL)['cases']
    assert len(cases) == len({c['case_id'] for c in cases}) == 200
    assert sha(PANEL) == read(baseline.OUT / 'protocol.json')['panel_sha256']
    actor_hash = sha(baseline.OUT / 'per_scene.csv')
    assert prior['complete'] and actor_hash == prior['files']['per_scene.csv']
    frozen = read(PARENT / 'ALL_MODELS_FROZEN.json')['files']
    weights = {}
    jobs = []
    for seed in SEEDS:
        path = checkpoint(seed)
        name = f'offline_s{seed}'
        digest = sha(path)
        assert frozen[str(path.relative_to(PARENT))] == digest
        metadata = read(path.parent / 'offline_complete.json')
        assert metadata['complete'] and metadata['arm'] == 'GA' and metadata['seed'] == seed
        weights[str(path)] = digest
        jobs.extend(dict(condition=name, case=case, checkpoint=str(path), epsilon=0.,
                         directory=str(OUT/'evaluation'/name/case['case_id'])) for case in cases)
    protocol = dict(actor_per_scene_sha256=actor_hash, weights=weights, panel_sha256=sha(PANEL),
                    seeds=SEEDS, workers=12, max_ticks=1800, max_decisions=300, wall_seconds=1800, epsilon=0,
                    primary='Offline Q minus 24k actor retention in seven-CAP success union',
                    secondary=['all', 'one successful CAP', 'multiple successful CAPs', 'outside union'],
                    ci='10,000 paired scene bootstrap draws stratified by movable count, both fixed seeds kept together')
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json', protocol)
    save(OUT/'progress.json', dict(completed=0, total=len(jobs)))
    inputs = read(PARENT/'inputs.json')
    with ProcessPoolExecutor(12, mp_context=mp.get_context('spawn'), initializer=fast_runtime.initialize,
                             initargs=(inputs['bundle'], inputs['priority_bundle'])) as pool:
        pending = [pool.submit(fast_runtime.run, job) for job in jobs]
        for n, future in enumerate(as_completed(pending), 1):
            result = future.result()
            if not result['complete']:
                raise RuntimeError(result)
            if n % 10 == 0:
                save(OUT/'progress.json', dict(completed=n, total=len(jobs)))
                print(f'{n}/{len(jobs)} complete', flush=True)


if __name__ == '__main__':
    run()
