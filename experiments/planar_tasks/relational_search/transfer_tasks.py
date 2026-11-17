"""Versioned relation-transfer development fixtures and editable testbed seeds."""
from pathlib import Path
import json
from .contract import digest

VERSION='relational-transfer-v1'
PATH=Path(__file__).resolve().parents[1]/'fixtures/relational_transfer_v1.json'
DIRECTIONS={'east':'东向','north':'北向','west':'西向','diagonal':'斜向'}


def cases():
    data=json.loads(PATH.read_text())
    if data['version']!=VERSION:raise ValueError('relation transfer suite version')
    rows=data['cases']
    if len(rows)!=8 or len({r['id'] for r in rows})!=8:raise ValueError('expected eight unique development cases')
    for row in rows:
        if row['split']!='development' or digest(row['fixture'])!=row['fixture_sha256']:
            raise ValueError('relation transfer fixture integrity')
    return rows


def preset_id(case):return 'rel_transfer_v1_'+case['id']


def title(case):
    relation='障碍挡路' if case['host_annotation']['straight_corridor_overlap'] else '通道已空'
    return f'关系迁移 v1 · {DIRECTIONS[case["layout"]]} · {relation}'


def preview_sequence():
    # An explicit harmless preview, not a purported solution or teacher demo.
    return [dict(target_id=None,a0=44,a1=44,decisions=1)]
