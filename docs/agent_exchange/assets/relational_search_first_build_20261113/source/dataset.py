"""Durable preference records with causal history and mother-group separation."""
from dataclasses import asdict
import json
from pathlib import Path
from copy import deepcopy
from .contract import History, canonical, digest, validate_action
from .evidence import Outcome, compare


def record(group, split, history, left, right):
    if not isinstance(group, str) or not group or split not in ('train', 'validation', 'test'):
        raise ValueError('explicit mother group and split required')
    if not history.events or left.origin != history.sha256 or right.origin != history.sha256:
        raise ValueError('outcomes must reference this exact legal history')
    obs = history.events[-1]['observation']
    validate_action(left.action, obs)
    validate_action(right.action, obs)
    return dict(version='relational-comparison-v1', group=group, split=split,
                history=deepcopy(history.events), left=asdict(left), right=asdict(right),
                comparison=compare(left, right))


def verify(row):
    if row['version'] != 'relational-comparison-v1':
        raise ValueError('dataset version')
    h = History()
    for e in row['history']:
        # A retained 64-event window may begin with a preceding action outside
        # the window. It is metadata, not an extra visible observation.
        h.append(e['observation'], None if not h.events else e['action'])
    if row['history']:
        h.events[0]['action'] = row['history'][0]['action']
    expected = record(row['group'], row['split'], h, Outcome(**row['left']), Outcome(**row['right']))
    if canonical(expected) != canonical(row):
        raise ValueError('comparison or public history was altered')
    return h


def write(path, records):
    """Never overwrite an evidence dataset. Unknowns/ties remain in the file."""
    rows = list(records)
    groups = {}
    for row in rows:
        verify(row)
        previous = groups.setdefault(row['group'], row['split'])
        if previous != row['split']:
            raise ValueError('mother group crosses data splits')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        for row in rows:
            f.write(canonical(row)+'\n')
    return dict(rows=len(rows), sha256=digest(rows), groups=groups)


def read_training(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    groups = {}
    selected = []
    for row in rows:
        h = verify(row)
        previous = groups.setdefault(row['group'], row['split'])
        if previous != row['split']:
            raise ValueError('mother group crosses data splits')
        if row['split'] == 'train' and row['comparison']['status'] == 'preference':
            selected.append((h, row['comparison']))
    return selected
