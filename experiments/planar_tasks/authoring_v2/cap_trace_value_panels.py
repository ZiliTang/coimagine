"""Freeze new final-test and online-training roots before any fitting."""
import hashlib
from pathlib import Path

from .random_room_tasks import generate
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save


def freeze(bundle, out):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seen = {r['content_sha256'] for r in read(bundle/'used_scene_identities.json')}
    seen_ids = {r['case_id'] for r in read(bundle/'used_scene_identities.json')}
    previous_count = len(seen)
    panels = {}
    for name, seed in [('final_test', 2026111802), ('online_training', 2026111804)]:
        rows = []
        skipped = []
        for attempt in range(100):
            source = generate(seed+1000*attempt)
            for row in source['cases']:
                count = row['movable_count']
                if sum(r['movable_count']==count for r in rows) >= 50:
                    continue
                fixture = row['fixture']
                content = digest({k:v for k,v in fixture.items() if k != 'scene_id'})
                if content in seen:
                    skipped.append(content)
                    continue
                case_id = f'{name}_{count}_{sum(r["movable_count"]==count for r in rows):03d}'
                assert case_id not in seen_ids
                seen.add(content)
                seen_ids.add(case_id)
                rows.append(dict(case_id=case_id, fixture=fixture, fixture_sha256=digest(fixture),
                                 content_sha256=content, movable_count=count, generation_seed=seed+1000*attempt))
            if len(rows) == 200:
                break
        assert len(rows) == 200
        panels[name] = dict(seed=seed, cases=rows, duplicate_content_rejections=skipped)
        save(out/(name+'.json'), panels[name])
    source_path = Path(__file__).with_name('random_room_tasks.py')
    record = dict(frozen=True, previous_unique_contents=previous_count, new_test=200, new_online=200,
                  cross_panel_overlap=0, generator_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  no_policy_executions=True, no_outcome_filtering=True,
                  files={name+'.json':hashlib.sha256((out/(name+'.json')).read_bytes()).hexdigest() for name in panels})
    save(out/'manifest.json', record)
    print(record, flush=True)
