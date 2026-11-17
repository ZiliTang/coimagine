"""Host-only, seeded evaluation panel; no outcome-based screening or dev aliases.

The distribution is deliberately narrow: the same three task families and fixed
rectangle as development. New roots test instance transfer, not unseen dynamics,
new task families, or general manipulation. Geometry legality is not solvability.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

from coimagine.continuous2d.scenes import Box, Pose
from coimagine.random_start.layout import InstanceSpec
from coimagine.random_start import spec as R
from ..tasks import geometry_audit, task_catalog

PANEL_VERSION = 'planar-evaluation-roots-v1'
FAMILIES = ('open', 'wall', 'detour')
SEED_NAMESPACE = 2**96
MAX_DRAWS = 1000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationTask:
    family: str
    root_id: str
    instance: InstanceSpec

    def record(self):
        return {'taskset_version': PANEL_VERSION, 'family': self.family,
                'root_id': self.root_id, 'split': 'independent_evaluation',
                'description': 'Fresh sampled root; same-family instance transfer.',
                'feasibility': 'unverified', 'instance': self.instance.as_dict()}

    @property
    def sha256(self):
        return content_hash(self.record())


def _geometry_record(instance):
    result = instance.as_dict()
    for name in ('name', 'map', 'split', 'index', 'seed', 'env_version'):
        result.pop(name)
    return result


def _draw(family, index, seed, rng):
    if family == 'wall':
        wall = Box(rng.uniform(.9, 1.05), rng.uniform(1.85, 2.15),
                   rng.uniform(.12, .18), rng.uniform(1.05, 1.4))
        angle = rng.uniform(-.12, .12)
        extent = .5 * abs(math.cos(angle)) + .125 * abs(math.sin(angle))
        ox = wall.cx + wall.half_w + extent + rng.uniform(.01, .08)
        oy = rng.uniform(1.6, 2.4)
        offsets = ((rng.uniform(-.28, -.16), -rng.uniform(.43, .52)),
                   (rng.uniform(.16, .28), rng.uniform(.43, .52)))
        obstacles = (wall,)
        goal = (rng.uniform(3.65, 4.2), rng.uniform(1.35, 2.65))
    else:
        ox = rng.uniform(1.5, 2.2) if family == 'open' else rng.uniform(1.3, 1.8)
        oy = rng.uniform(1.3, 2.7)
        angle = math.pi / 2 + rng.uniform(-.25, .25)
        offsets = ((-rng.uniform(.55, .65), -rng.uniform(.28, .40)),
                   (-rng.uniform(.55, .65), rng.uniform(.28, .40)))
        obstacles = () if family == 'open' else (
            Box(rng.uniform(2.65, 2.95), rng.uniform(1.6, 2.4),
                rng.uniform(.15, .25), rng.uniform(.4, .75)),)
        goal = (rng.uniform(3.8, 4.2), rng.uniform(1.35, 2.65))
    robots = tuple(Pose(ox + dx, oy + dy,
                       math.atan2(-dy, -dx) + rng.uniform(-.15, .15))
                   for dx, dy in offsets)
    name = f'planar_eval_v1_{family}_{seed:x}'
    instance = InstanceSpec(name=name, map='planar', split='independent_evaluation',
        index=index, seed=seed, arena=(5., 4.), obstacles=obstacles,
        object_pose=Pose(ox, oy, angle), robot_poses=robots,
        goal_centre=goal, goal_radius=R.SUCCESS_DISTANCE)
    return EvaluationTask(family, name, instance)


def generate_panel(seed=2026110901, per_family=10):
    """Deterministic host generation. Does not instantiate or step physics."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if type(per_family) is not int or not 1 <= per_family <= 1000:
        raise ValueError('per_family must be an integer in [1, 1000]')
    used = {content_hash(_geometry_record(t.instance)) for t in task_catalog().values()}
    records, rejection_counts = [], {}
    for family_index, family in enumerate(FAMILIES):
        for index in range(per_family):
            case_seed = SEED_NAMESPACE + (seed << 24) + (family_index << 20) + index
            rng = random.Random(case_seed)
            for draw in range(MAX_DRAWS):
                task = _draw(family, index, case_seed, rng)
                audit = geometry_audit(task.instance)
                fingerprint = content_hash(_geometry_record(task.instance))
                if audit['legal'] and fingerprint not in used:
                    used.add(fingerprint)
                    records.append({'task': task.record(), 'task_sha256': task.sha256,
                                    'geometry': audit, 'draws': draw + 1})
                    break
            else:
                raise RuntimeError(f'geometry generation exhausted: {family}/{index}')
            rejection_counts[task.root_id] = draw
    panel = {'version': PANEL_VERSION, 'seed': seed, 'per_family': per_family,
             'split': 'independent_evaluation', 'selection': 'seeded_geometry_only',
             'feasibility': 'unverified', 'physics_evaluations_at_creation': 0,
             'scope': 'same-family instance transfer; fixed shape/dynamics, no new-family claim',
             'development_geometry_sha256': sorted(content_hash(_geometry_record(t.instance))
                                                     for t in task_catalog().values()),
             'rejection_counts': rejection_counts, 'tasks': records,
             'instances_sha256': content_hash([row['task_sha256'] for row in records])}
    return dict(panel, content_sha256=content_hash(panel))


def create_panel(path, seed=2026110901, per_family=10):
    """Write once. Keep this artifact outside the model development workspace."""
    panel = generate_panel(seed, per_family)
    panel.pop('content_sha256')
    panel['generated_unix'] = time.time()
    panel['content_sha256'] = content_hash(panel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(panel, indent=2, allow_nan=False) + '\n')
    return panel


def load_panel(path):
    """Validate hash, provenance and exact deterministic generation, then hydrate."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('panel must be a regular non-symlink file')
    panel = json.loads(path.read_text())
    if panel.get('version') != PANEL_VERSION:
        raise ValueError('unknown evaluation panel version')
    payload = {k: v for k, v in panel.items() if k != 'content_sha256'}
    if content_hash(payload) != panel.get('content_sha256'):
        raise ValueError('evaluation panel hash changed')
    generated = panel.get('generated_unix')
    if isinstance(generated, bool) or not isinstance(generated, (int, float)) or not math.isfinite(generated) or generated <= 0:
        raise ValueError('panel generation timestamp required')
    # Recomputed expected content rejects relabelled development fixtures and
    # post-hoc hand-picked records even if someone rewrites the outer hash.
    expected = generate_panel(panel['seed'], panel['per_family'])
    expected.pop('content_sha256')
    expected['generated_unix'] = generated
    expected['content_sha256'] = content_hash(expected)
    if panel != expected:
        raise ValueError('evaluation panel differs from its frozen generator')
    tasks = []
    for row in panel['tasks']:
        data = row['task']; item = data['instance']
        instance = InstanceSpec(name=item['name'], map=item['map'], split=item['split'],
            index=item['index'], seed=item['seed'], arena=tuple(item['arena']),
            obstacles=tuple(Box(**b) for b in item['obstacles']),
            object_pose=Pose(**item['object']), robot_poses=tuple(Pose(**p) for p in item['robots']),
            goal_centre=tuple(item['goal_centre']), goal_radius=item['goal_radius'],
            env_version=item['env_version'], object_polygon=None)
        tasks.append(EvaluationTask(data['family'], data['root_id'], instance))
    return panel, tuple(tasks)
