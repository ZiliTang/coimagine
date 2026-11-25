"""Paired development diagnostic: exchange the pusher and assistant at a PUSH endpoint."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tools.guarded_calibration_v2 import cap_guarded_calibration_forks as study


def compare(room):
    packet = study.archive.read(study.OUT/'forks/dev'/(room+'.json'))
    if packet['collector_sha256'] != study.frozen_hash():
        raise ValueError('collector differs')
    anchors = {a['anchor_id']: a for a in packet['anchors']}
    regular = study.followup
    def swapped(world, action, *, execute):
        return regular(world, dict(action, a0=action['a1'], a1=action['a0']), execute=execute)
    rows = []
    for row in packet['rows']:
        anchor = anchors[row['anchor_id']]
        original = row['outcome']
        with patch.object(study, 'followup', swapped):
            alternate = study.evaluate(study.development_rooms()[room], study.record_for('dev',room),
                                       anchor, row['progress_m'], execute_followup=True)
        if any(alternate[k] != original[k] for k in original if k not in (
            'followup_ticks','followup_reason','followup_completed','endpoint_followup_available',
            'orbit_ticks','approach_ticks')):
            raise ValueError('pre-handoff physical result differs')
        rows.append(dict(room=room, anchor_id=anchor['anchor_id'], tick=anchor['tick'], public_history=anchor['public_history'],
                         progress_m=row['progress_m'], original_action=anchor['original_action'],
                         keep=original, swap=alternate))
    return rows


if __name__ == '__main__':
    with ProcessPoolExecutor(4) as pool:
        rows = [r for batch in pool.map(compare,sorted(study.development_rooms())) for r in batch]
    def counts(group):
        return {key:dict(available=sum(r[key]['endpoint_followup_available'] for r in group),
                        attempted=sum(r[key]['approach_ticks']>0 for r in group),
                        completed=sum(r[key]['followup_completed'] for r in group))
               for key in ('keep','swap')}
    summary = {stage:counts([r for r in rows if r['anchor_id']==stage]) for stage in ('early','late')}
    study.write_json(study.OUT/'role_swap_dev.json', dict(summary=summary, rows=rows,
                     source_sha256=study.archive.sha(Path(__file__)),collector_sha256=study.frozen_hash()))
    print(summary)
