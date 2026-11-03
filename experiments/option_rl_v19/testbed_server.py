"""D006 local testbed server: static UI + thin API over D005 TestbedSession."""
from __future__ import annotations
import argparse, json, math, sys, time, uuid, os, hashlib, threading
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
STATIC = Path(__file__).resolve().parent / 'testbed'
OUT = ROOT / 'outputs/testbed'
RUNS = OUT / 'runs'
RUN_INDEX = OUT / 'run_index.json'
_RUN_INDEX_LOCK = threading.RLock()
_RUN_INDEX_CACHE = None
OVERRIDES = OUT / 'overrides'
PRESETS = OUT / 'presets'
PANEL = ROOT / 'outputs/option_rl_v14/learnability_002/panel.json'
OPTION_TABLE = ROOT / 'docs/agent_exchange/state/option_table.csv'
PARAM_TABLE = ROOT / 'docs/agent_exchange/state/param_table.csv'

from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import spec as R
from experiments.option_rl_v16.learnability import load_panel
from experiments.option_rl_v19.session import (
    CONTROLLER_SOURCE_PATH, CONTROLLER_SOURCE_SHA256,
)
from experiments.option_rl_v21.session import TestbedSession
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.review_cases import review_scenes, review_presets

SERVER_STARTED_AT = time.time()
_RUNTIME_SOURCE_FILES = (
    'docs/agent_exchange/state/testbed_panel.json',
    'experiments/option_rl_v21/spec.py',
    'experiments/option_rl_v21/system.py',
    'experiments/option_rl_v21/session.py',
    'experiments/option_rl_v21/yield_skill.py',
    'experiments/option_rl_v21/review_cases.py',
    'experiments/option_rl_v17/contact_timing_audit.py',
    'experiments/option_rl_v19/testbed_server.py',
    'experiments/option_rl_v19/session.py',
    'experiments/option_rl_v19/system.py',
    'experiments/option_rl_v19/spec.py',
    'experiments/option_rl_v13/controller.py',
    'experiments/option_rl_v19/unknown_shape.py',
    'experiments/option_rl_v13/mode_env.py',
    'experiments/option_rl_v13/push_controller.py',
    'src/coimagine/random_start/env.py',
    'src/coimagine/random_start/lidar.py',
    'src/coimagine/random_start/skills.py',
    'src/coimagine/random_start/spec.py',
    'src/coimagine/continuous2d/environment.py',
    'src/coimagine/continuous2d/backend.py',
)
_LOADED_RUNTIME_HASHES = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in _RUNTIME_SOURCE_FILES}
RUNTIME_SOURCE_SHA256 = hashlib.sha256(json.dumps(_LOADED_RUNTIME_HASHES, sort_keys=True).encode()).hexdigest()


def _server_status():
    disk_sha = hashlib.sha256(CONTROLLER_SOURCE_PATH.read_bytes()).hexdigest()
    runtime_changed = any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest
                          for name, digest in _LOADED_RUNTIME_HASHES.items())
    return dict(pid=os.getpid(), started_at=SERVER_STARTED_AT, controller_version='option-v2',
                action_version=S.VERSION, action_codebook_hash=S.CODEBOOK_HASH,
                push_speeds=S.PUSH_SPEEDS, reposition_speed=S.REPOSITION_SPEED,
                support_memory_enabled=False, q_acquisition_mode='current_visible_support',
                controller_source_sha256=CONTROLLER_SOURCE_SHA256,
                controller_disk_sha256=disk_sha,
                runtime_source_sha256=RUNTIME_SOURCE_SHA256,
                restart_required=(disk_sha != CONTROLLER_SOURCE_SHA256 or runtime_changed))


def _load_scenes():
    panel = PANEL if PANEL.exists() else ROOT/'docs/agent_exchange/state/testbed_panel.json'
    train, _ = load_panel(panel)
    out = {}
    for i in train:
        out[i.name] = i
    if train:
        base = train[0]
        shapes = {
            'poly_T': ((0., 0.), (1., 0.), (1., .30), (.65, .30),
                       (.65, .85), (.35, .85), (.35, .30), (0., .30)),
            'poly_triangle': ((0., 0.), (1., 0.), (.5, .85)),
            'poly_octagon': tuple((.5 + .5*math.cos(2*math.pi*i/8),
                                  .5 + .5*math.sin(2*math.pi*i/8)) for i in range(8)),
        }
        for name, poly in shapes.items():
            cx = sum(p[0] for p in poly)/len(poly); cy = sum(p[1] for p in poly)/len(poly)
            local = tuple((float(x-cx), float(y-cy)) for x, y in poly)
            out[name] = replace(base, name=name, map='L0', index=900+len(out),
                                obstacles=(), object_polygon=local,
                                object_pose=Pose(2.5, 2.0, 0.0),
                                robot_poses=(Pose(1.75, 1.55), Pose(1.75, 2.45)),
                                goal_centre=(3.65, 2.0), goal_radius=.20)
    return out


SCENES = {**_load_scenes(), **review_scenes()}


def _scene_dict(inst):
    return dict(id=inst.name, name=inst.name, arena=list(inst.arena),
                goal=dict(x=inst.goal_centre[0], y=inst.goal_centre[1], radius=inst.goal_radius),
                object_size=[R.OBJECT_WIDTH, R.OBJECT_HEIGHT], robot_radius=R.ROBOT_RADIUS,
                object_polygon=(None if getattr(inst, 'object_polygon', None) is None else [list(v) for v in inst.object_polygon]),
                initial=dict(object=dict(x=inst.object_pose.x, y=inst.object_pose.y, angle=inst.object_pose.angle),
                             robots=[dict(x=p.x, y=p.y, angle=p.angle) for p in inst.robot_poses]))


def _auto_description(seq, created=None):
    labels = []
    for seg in (seq or [])[:3]:
        a0, a1 = int(seg['a0']), int(seg['a1'])
        if seg.get('budget_m') is not None or seg.get('time_budget_s') is not None:
            parts=[]
            if seg.get('budget_m') is not None: parts.append(f"{float(seg['budget_m']):.2f}m {seg.get('progress','object_normal')}")
            if seg.get('time_budget_s') is not None: parts.append(f"{float(seg['time_budget_s']):.1f}s")
            labels.append(f"{S.SPEC_BY_INDEX[a0].label}/{S.SPEC_BY_INDEX[a1].label} {'/'.join(parts)}")
        else:
            labels.append(f"{S.SPEC_BY_INDEX[a0].label}/{S.SPEC_BY_INDEX[a1].label} "
                          f"{int(seg.get('decisions', 1)) * 0.2:.1f}s")
    summary = ' ; '.join(labels) if labels else 'empty sequence'
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created if created else time.time()))
    return f"{summary} · {ts}"


def _seed_presets():
    PRESETS.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for name, inst in SCENES.items():
        path = PRESETS / f'seed_{name}.json'
        if path.exists():
            continue
        meta = _scene_dict(inst)
        data = dict(id=f'seed_{name}', name=f'seed_{name}', description=_auto_description([], now),
                    created=now, scene=name, base_scene=name,
                    object=meta['initial']['object'], robots=meta['initial']['robots'],
                    sequence=[], overrides={}, scene_meta=meta)
        path.write_text(json.dumps(data, indent=1))
    for pid, title, scene, sequence in review_presets():
        path = PRESETS / f'{pid}.json'
        if path.exists():
            continue
        meta = _scene_dict(SCENES[scene])
        path.write_text(json.dumps(dict(id=pid, name=title, description=_auto_description(sequence, now),
            created=now, scene=scene, base_scene=scene, object=meta['initial']['object'],
            robots=meta['initial']['robots'], sequence=sequence, overrides={}, scene_meta=meta), indent=1))


def _read_csv(path):
    import csv
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def _option_list():
    return [dict(key=s.key, label=s.label, family=s.family, action_index=s.action_index)
            for s in S.ACTIVE_SPECS]


def _param_list():
    out = []
    for r in _read_csv(PARAM_TABLE):
        out.append(dict(family=r['family'], name=r['name'], default=float(r['default']),
                        low=float(r['low']), high=float(r['high']), unit=r['unit'],
                        description=r['description']))
    for p in S.YIELD_PARAMS:
        out.append(dict(family='YIELD', name=p.name, default=p.default, low=p.low,
                        high=p.high, unit=p.unit, description=p.description))
    return out


OPTIONS = _option_list()
PARAMS = _param_list()


def _json_error(msg, code=400):
    return code, dict(error=msg)


def _scene_instance(scene_id, object_pose, robots, goal=None):
    if scene_id not in SCENES:
        raise ValueError('unknown scene')
    if len(robots) != 2:
        raise ValueError('two robots required')
    base = SCENES[scene_id]
    obj = Pose(float(object_pose['x']), float(object_pose['y']), float(object_pose.get('angle', 0.)))
    rs = tuple(Pose(float(r['x']), float(r['y']), float(r.get('angle', 0.))) for r in robots)
    kwargs = dict(object_pose=obj, robot_poses=rs)
    if goal:
        kwargs['goal_centre'] = (float(goal['x']), float(goal['y']))
        if goal.get('radius') is not None:
            kwargs['goal_radius'] = float(goal['radius'])
    return replace(base, **kwargs)


def _validate_sequence(seq):
    if not seq:
        raise ValueError('empty sequence')
    if len(seq) > 200:
        raise ValueError('sequence too long')
    for seg in seq:
        a0, a1 = int(seg['a0']), int(seg['a1'])
        if not (0 <= a0 < len(S.ACTIONS) and 0 <= a1 < len(S.ACTIONS)):
            raise ValueError('option index out of range')
        if not (1 <= int(seg.get('decisions', 1)) <= 200):
            raise ValueError('decisions out of range')
        if seg.get('budget_m') is not None:
            bm = float(seg['budget_m'])
            if not math.isfinite(bm) or bm <= 0:
                raise ValueError('budget_m must be positive')
            if seg.get('progress', 'object_normal') not in ('object_normal', 'robot_path'):
                raise ValueError('invalid progress kind')
        if seg.get('time_budget_s') is not None:
            tb = float(seg['time_budget_s'])
            if not math.isfinite(tb) or tb <= 0:
                raise ValueError('time_budget_s must be positive')


def _preset_save(body):
    scene = body.get('scene')
    controller_version = 'option-v2'
    if scene not in SCENES:
        raise ValueError('unknown scene')
    obj = body.get('object') or {}
    robots = body.get('robots') or []
    inst = _scene_instance(scene, obj, robots, body.get('goal'))
    seq = S.canonical_sequence(body.get('sequence') or [])
    _validate_sequence(seq)
    now = time.time()
    name = str(body.get('name') or '').strip()
    if not name:
        name = f"{scene}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}"
    pid = time.strftime('%Y%m%d_%H%M%S', time.localtime(now)) + '_' + uuid.uuid4().hex[:6]
    normalized = [dict(a0=int(s['a0']), a1=int(s['a1']), decisions=int(s.get('decisions', 1)), **({'budget_m': float(s['budget_m']), 'progress': s.get('progress','object_normal')} if s.get('budget_m') is not None else {}), **({'time_budget_s': float(s['time_budget_s'])} if s.get('time_budget_s') is not None else {})) for s in seq]
    data = dict(id=pid, name=name, description=_auto_description(normalized, now),
                created=now, scene=scene, base_scene=scene,
                object=dict(x=inst.object_pose.x, y=inst.object_pose.y, angle=inst.object_pose.angle),
                robots=[dict(x=p.x, y=p.y, angle=p.angle) for p in inst.robot_poses],
                sequence=normalized, overrides=body.get('overrides') or {}, controller_version='m3b',
                action_version=S.VERSION, action_codebook_hash=S.CODEBOOK_HASH,
                scene_meta=_scene_dict(inst))
    PRESETS.mkdir(parents=True, exist_ok=True)
    (PRESETS / f'{pid}.json').write_text(json.dumps(data, indent=1))
    return data


def _preset_list():
    PRESETS.mkdir(parents=True, exist_ok=True)
    out = []
    for p in PRESETS.glob('*.json'):
        try:
            r = json.loads(p.read_text())
            seq = r.get('sequence', [])
            out.append(dict(id=r.get('id', p.stem), name=r.get('name', p.stem),
                            description=_auto_description(seq, r.get('created')),
                            scene=r.get('scene'), base_scene=r.get('base_scene', r.get('scene')),
                            created=r.get('created'), sequence_len=len(seq), controller_version='m3b'))
        except Exception:
            continue
    return out


def _preset_get(pid):
    p = PRESETS / f'{pid}.json'
    if not p.exists():
        raise FileNotFoundError('preset not found')
    r = json.loads(p.read_text())
    r['sequence'] = S.canonical_sequence(r.get('sequence', []))
    r['action_version'] = S.VERSION
    if not r.get('scene_meta') and r.get('scene') in SCENES:
        r['scene_meta'] = _scene_dict(SCENES[r['scene']])
    r['description'] = _auto_description(r.get('sequence', []), r.get('created'))
    return r


def _preset_delete(pid):
    p = PRESETS / f'{pid}.json'
    if not p.exists():
        raise FileNotFoundError('preset not found')
    p.unlink()
    return dict(id=pid, deleted=True)


def _run(body):
    if _server_status()['restart_required']:
        raise ValueError('Controller source changed; restart testbed before running the new version.')
    seq = S.canonical_sequence(body.get('sequence') or [])
    _validate_sequence(seq)
    inst = _scene_instance(body.get('scene'), body.get('object') or {}, body.get('robots') or [], body.get('goal'))
    controller_version = 'm3b'
    session = TestbedSession(inst, overrides=body.get('overrides') or {}, controller_version=controller_version)
    for seg in seq:
        session.add(int(seg['a0']), int(seg['a1']), int(seg.get('decisions', 1)), seg.get('budget_m'), seg.get('time_budget_s'), seg.get('progress', 'decisions'))
    res = session.run()
    rid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    res.update(id=rid, name=body.get('name') or rid, scene=inst.name,
               runtime_source_sha256=RUNTIME_SOURCE_SHA256,
               object=dict(x=inst.object_pose.x, y=inst.object_pose.y, angle=inst.object_pose.angle),
               robots=[dict(x=p.x, y=p.y, angle=p.angle) for p in inst.robot_poses],
               overrides=body.get('overrides') or {}, controller_version=controller_version,
               scene_meta=_scene_dict(inst),
               created=time.time())
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f'{rid}.json'
    temp = RUNS / f'.{rid}.tmp'
    try:
        temp.write_text(json.dumps(res, indent=1))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    _index_run(path, res)
    return res


def _run_summary(run, stem):
    if not isinstance(run, dict):
        raise TypeError('run must be a JSON object')
    return dict(id=run.get('id', stem), name=run.get('name', stem),
                scene=run.get('scene'), success=run.get('success'),
                periods=run.get('periods'), created=run.get('created'),
                sequence_len=len(run.get('sequence', [])))


def _load_run_index():
    global _RUN_INDEX_CACHE
    if _RUN_INDEX_CACHE is None:
        try:
            data = json.loads(RUN_INDEX.read_text())
            _RUN_INDEX_CACHE = (data['runs'] if isinstance(data, dict) and data.get('version') == 1
                                and isinstance(data.get('runs'), dict) else {})
        except (OSError, ValueError, KeyError, TypeError):
            _RUN_INDEX_CACHE = {}
    return _RUN_INDEX_CACHE


def _save_run_index():
    RUN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    temp = RUN_INDEX.with_name(f'.run_index.{uuid.uuid4().hex}.tmp')
    try:
        temp.write_text(json.dumps(dict(version=1, runs=_RUN_INDEX_CACHE)))
        temp.replace(RUN_INDEX)
    finally:
        temp.unlink(missing_ok=True)


def _index_run(path, run):
    with _RUN_INDEX_LOCK:
        cache = _load_run_index()
        st = path.stat()
        cache[path.name] = dict(mtime_ns=st.st_mtime_ns, size=st.st_size,
                                summary=_run_summary(run, path.stem))
        _save_run_index()


def _run_list():
    """Read metadata only; unchanged history never needs trajectory decoding."""
    RUNS.mkdir(parents=True, exist_ok=True)
    with _RUN_INDEX_LOCK:
        cache = _load_run_index()
        files = [(p, p.stat()) for p in RUNS.glob('*.json')]
        out, seen, changed = [], set(), False
        for p, st in sorted(files, key=lambda x: x[1].st_mtime_ns, reverse=True):
            seen.add(p.name)
            entry = cache.get(p.name)
            if (not isinstance(entry, dict) or not isinstance(entry.get('summary'), dict)
                    or entry.get('mtime_ns') != st.st_mtime_ns or entry.get('size') != st.st_size):
                try:
                    run = json.loads(p.read_text())
                    entry = dict(mtime_ns=st.st_mtime_ns, size=st.st_size,
                                 summary=_run_summary(run, p.stem))
                    cache[p.name] = entry
                    changed = True
                except (OSError, ValueError, TypeError):
                    continue
            out.append(entry['summary'])
        for name in set(cache)-seen:
            del cache[name]
            changed = True
        if changed:
            _save_run_index()
        return out


def _get_override(name):
    name = unquote(name)
    if not name or '/' in name or '\\' in name or '..' in name:
        raise ValueError('invalid override name')
    p = OVERRIDES / f'{name}.json'
    if not p.exists():
        raise FileNotFoundError('override not found')
    return dict(name=name, overrides=json.loads(p.read_text()), created=p.stat().st_mtime)


def _save_override(body):
    name = str(body.get('name') or '').strip()
    if not name or '/' in name or '\\' in name or '..' in name:
        raise ValueError('invalid override name')
    ov = body.get('overrides') or {}
    OVERRIDES.mkdir(parents=True, exist_ok=True)
    path = OVERRIDES / f'{name}.json'
    path.write_text(json.dumps(ov, indent=1))
    return dict(path=str(path.relative_to(ROOT)), overrides=ov)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get('Content-Length', '0'))
        return json.loads(self.rfile.read(n).decode()) if n else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/status':
            return self._send(200, _server_status())
        if path == '/api/options':
            return self._send(200, OPTIONS)
        if path == '/api/params':
            return self._send(200, PARAMS)
        if path == '/api/runs':
            return self._send(200, _run_list())
        if path == '/api/presets':
            return self._send(200, _preset_list())
        if path.startswith('/api/presets/'):
            pid = path.split('/', 3)[3]
            try:
                return self._send(200, _preset_get(pid))
            except FileNotFoundError:
                return self._send(404, dict(error='preset not found'))
        if path == '/api/overrides':
            OVERRIDES.mkdir(parents=True, exist_ok=True)
            return self._send(200, sorted(p.stem for p in OVERRIDES.glob('*.json')))
        if path.startswith('/api/overrides/'):
            name = path.split('/', 3)[3]
            try:
                return self._send(200, _get_override(name))
            except FileNotFoundError:
                return self._send(404, dict(error='override not found'))
        if path.startswith('/api/runs/'):
            rid = path.split('/', 3)[3]
            p = RUNS / f'{rid}.json'
            if not p.exists():
                return self._send(404, dict(error='run not found'))
            run = json.loads(p.read_text())
            if not run.get('scene_meta') and run.get('scene') in SCENES:
                run['scene_meta'] = _scene_dict(SCENES[run['scene']])
            return self._send(200, run)
        if path == '/favicon.ico':
            self.send_response(204); self.end_headers(); return
        return super().do_GET()

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith('/api/presets/'):
            pid = path.split('/', 3)[3]
            try:
                return self._send(200, _preset_delete(pid))
            except FileNotFoundError:
                return self._send(404, dict(error='preset not found'))
        return self._send(404, dict(error='not found'))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == '/api/run':
                return self._send(200, _run(body))
            if path == '/api/overrides':
                return self._send(200, _save_override(body))
            if path == '/api/presets':
                return self._send(200, _preset_save(body))
            return self._send(404, dict(error='not found'))
        except Exception as exc:
            return self._send(400, dict(error=str(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    OVERRIDES.mkdir(parents=True, exist_ok=True)
    _seed_presets()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'CoImagine testbed http://{args.host}:{args.port}/ · runtime {RUNTIME_SOURCE_SHA256[:12]}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
