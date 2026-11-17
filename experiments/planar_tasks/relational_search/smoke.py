"""One bounded M0 integration + M1 physics probe, with honest evidence labels.

No LLM, controller training, full policy evaluation or knowledge discovery here.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .contract import History, digest, public_observation
from .evidence import Ledger
from .fit import fit
from .physics_probe import Fixture, PhysicsProbe


def mechanism_fixture():
    """Handwritten synthetic history-aliasing fixture. NOT a physical dataset."""
    def packet(tick, past_x):
        return dict(tick=tick, decision=tick, protocol='synthetic-mechanism', robots=[],
                    goal=dict(target_id='object', xy=[3., 0.], radius=.1),
                    legal_pairs=[[0, 0], [44, 44]],
                    entities=[dict(id='object', points=[[past_x, 0.]], last_seen_tick=tick)])
    def features(h, a, revised):
        command = 1. if a['a0'] == 0 else -1.
        values = [command]
        if revised:
            values.append(command * h.events[0]['observation']['entities'][0]['points'][0][0])
        return values
    def dataset(magnitudes):
        data = []
        for m in magnitudes:
            for sign in (-1, 1):
                h = History(); h.append(packet(0, sign*m))
                prior = dict(target_id='object', a0=44, a1=44, ticks=20)
                h.append(packet(20, 0.), prior)
                good = dict(target_id='object', a0=0 if sign > 0 else 44,
                            a1=0 if sign > 0 else 44, ticks=20)
                bad = dict(good, a0=44 if sign > 0 else 0, a1=44 if sign > 0 else 0)
                data.append((h, good, bad))
        return data
    train, held = dataset([.2, .4, .6]), dataset([.3, .5])
    result = dict(source='handwritten synthetic diagnostic, not LLM/physics discovery', models={})
    for revised in (False, True):
        pos = [features(h, a, revised) for h, a, _ in train]
        neg = [features(h, b, revised) for h, _, b in train]
        model, losses = fit(pos, neg)
        margins = model.score([features(h, a, revised) for h, a, _ in held]) - model.score([features(h, b, revised) for h, _, b in held])
        result['models']['revised' if revised else 'initial'] = dict(
            checkpoint=model.record(), loss_before=losses[0], loss_after=losses[-1],
            held_pairs=len(held), strict_correct=int(np.sum(margins > 1e-8)),
            ties=int(np.sum(np.abs(margins) <= 1e-8)))
    return result


def run(destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ledger = Ledger(dict(execution_ticks=4000, prediction_ticks=24000))
    summary = dict(status='running', phase='M0 plus M1 low-level probe', llm_calls=0,
                   task_success_evaluations=0, mechanism=mechanism_fixture(), physics=[])
    try:
        for name, fixture in [('free', Fixture(blocker=False)), ('coupled', Fixture()),
                              ('heavy', Fixture(blocker_mass=30.))]:
            p = PhysicsProbe(fixture)
            initial = p.audit_state()
            public = public_observation(p.observation())
            with (out / (name + '.jsonl')).open('w') as log:
                log.write(json.dumps(dict(observation=public, offline_audit=initial)) + '\n')
                for _ in range(160):
                    if time.monotonic() - started > 900:
                        raise RuntimeError('wall budget exhausted')
                    ledger.charge(execution_ticks=1)
                    p.step_forces(((10., 0.), (10., 0.)))
                    log.write(json.dumps(dict(observation=public_observation(p.observation()),
                                              offline_audit=p.audit_state())) + '\n')
            # Replay is explicit additional physical work, never a snapshot clone.
            ledger.charge(execution_ticks=len(p.prefix))
            replay = p.replay()
            final = p.audit_state()
            assert replay.audit_state() == final
            summary['physics'].append(dict(name=name, configuration=p.configuration(),
                executed_ticks=160, replay_ticks=160, exact_replay=True,
                target_dx=final['objects'][0][0]-initial['objects'][0][0],
                blocker_dx=None if not fixture.blocker else final['objects'][1][0]-initial['objects'][1][0]))
        assert summary['physics'][1]['blocker_dx'] > .01, 'coupled body did not move'
        assert summary['physics'][1]['target_dx'] < summary['physics'][0]['target_dx'], 'coupling effect absent'
        summary['status'] = 'complete'
    except Exception as exc:
        summary.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['ledger'] = ledger.record()
        summary['wall_seconds'] = time.monotonic() - started
        summary['limitations'] = ['No Option target switching yet', 'No deployable multi-object observation adapter',
                                  'No learned history encoder/LLM revision', 'Synthetic fitting fixture is not task performance']
        (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        import hashlib
        files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.rglob('*') if p.is_file()}
        sources = list(Path(__file__).parent.glob('*.py')) + [Path('tests/test_relational_search.py')]
        frozen = out/'source'; frozen.mkdir()
        for p in sources:
            target = frozen/p.name
            target.write_bytes(p.read_bytes())
            files[str(target.relative_to(out))] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = dict(files=files, summary_sha256=digest(summary))
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    args = parser.parse_args()
    result = run(args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
