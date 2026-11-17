"""Small, standard-library-only runtime embedded in exported heuristic.py.

The interpreter enforces tree data flow; authored Python still needs the existing
process sandbox. This is not a sandbox against malicious semantic functions.
"""
from copy import deepcopy
import hashlib
import json
import math
import pickle
import sys


OBS_FIELDS = ('robot_id', 'tick', 'time', 'own', 'teammate', 'scan', 'patches', 'patch',
              'contacting_object', 'contact_point_robot', 'contacting_static',
              'contacting_teammate', 'goal_xy', 'goal_radius', 'arena', 'force_limit',
              'remaining_seconds', 'elapsed_seconds', 'static_nearest', 'teammate_nearest')
KINEMATICS_FIELDS = ('x', 'y', 'angle', 'vx', 'vy', 'omega')
FEEDBACK_FIELDS = ('stage', 'done', 'exit_reason', 'pause_reason', 'direction_valid',
                   'direction_unavailable', 'reference_velocity', 'desired_velocity', 'force')
SCAN_FIELDS = ('robot_id', 'tick', 'sample_time', 'origin', 'heading', 'ray_angles',
               'distances', 'valid', 'category')
PATCH_FIELDS = ('points_robot', 'points_world', 'distance', 'centroid_distance',
                'nearest_distance', 'normal_world', 'tangent_world', 'confidence',
                'arc_rays', 'span_m', 'rms_m', 'valid')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public_copy(value):
    if type(value) in (dict, list, tuple):
        return pickle.loads(pickle.dumps(value, protocol=4))
    return deepcopy(value)


def pick(value, fields):
    return {k: public_copy(value[k]) for k in fields if k in value}


def visible(observation):
    """Repeat the public sensor whitelist and return an independent public copy."""
    return _visible(observation, public_copy)


def _visible(observation, copy_value):
    """Shared whitelist; caller decides how already-owned values are transferred."""
    def select(value, fields):
        return {k: copy_value(value[k]) for k in fields if k in value}
    result = select(observation, ('protocol', 'decision', 'legal_pairs'))
    result['robots'] = []
    for robot in observation['robots']:
        r = select(robot, tuple(k for k in OBS_FIELDS if k not in ('own','teammate','scan','patches','patch')))
        for key in ('own', 'teammate'):
            r[key] = select(robot.get(key, {}), KINEMATICS_FIELDS)
        r['execution'] = select(robot.get('execution', {}), FEEDBACK_FIELDS)
        r['scan'] = select(robot.get('scan', {}), SCAN_FIELDS)
        r['patches'] = [select(p, PATCH_FIELDS) for p in robot.get('patches', [])]
        r['patch'] = select(robot['patch'], PATCH_FIELDS) if robot.get('patch') else None
        result['robots'].append(r)
    return result


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric cost required')
    return value


def expression(expr, features, action):
    if type(expr) in (float, int):
        return number(expr)
    key, value = next(iter(expr.items()))
    if key == 'feature':
        return number(features.get(value))
    if key == 'action':
        return number(action[value])
    args = [expression(x, features, action) for x in value]
    return number({'add': lambda: sum(args), 'mul': lambda: math.prod(args),
                   'min': lambda: min(args), 'max': lambda: max(args)}[key]())


def validate_trees(trees):
    if not isinstance(trees, dict) or set(trees) != {'value', 'rank'}:
        raise ValueError('value and rank trees required')
    count = 0

    def expr(x, allow_action, depth=0):
        if depth > 12:
            raise ValueError('expression depth limit')
        if type(x) in (float, int):
            number(x)
            return
        if not isinstance(x, dict) or len(x) != 1:
            raise ValueError('invalid cost expression')
        key, value = next(iter(x.items()))
        if key == 'feature' and isinstance(value, str):
            return
        if key == 'action' and allow_action and value in ('a0', 'a1', 'ticks'):
            return
        if key not in ('add', 'mul', 'min', 'max') or not isinstance(value, list) or not 1 <= len(value) <= 16:
            raise ValueError('invalid expression operation')
        for child in value:
            expr(child, allow_action, depth + 1)

    def walk(node, rank, depth=0):
        nonlocal count
        count += 1
        if count > 128 or depth > 12 or not isinstance(node, dict):
            raise ValueError('tree size/depth/type limit')
        if 'cost' in node:
            if set(node) - {'cost', 'proposals'} or (not rank and 'proposals' in node):
                raise ValueError('invalid leaf')
            expr(node['cost'], rank)
            proposals = node.get('proposals', [])
            if not isinstance(proposals, list) or len(proposals) > 8:
                raise ValueError('proposal limit')
            for proposal in proposals:
                if set(proposal) != {'a0', 'a1', 'ticks', 'cost'}:
                    raise ValueError('invalid proposal')
                for x in proposal.values():
                    expr(x, False)
            return
        if set(node) != {'feature', 'op', 'value', 'yes', 'no', 'unknown'}:
            raise ValueError('branch requires explicit yes/no/unknown')
        if not isinstance(node['feature'], str) or node['op'] not in ('eq', 'lt', 'le', 'gt', 'ge'):
            raise ValueError('invalid branch')
        if type(node['value']) not in (bool, int, float):
            raise ValueError('branch threshold must be scalar')
        if type(node['value']) is not bool:
            number(node['value'])
        for branch in ('yes', 'no', 'unknown'):
            walk(node[branch], rank, depth + 1)

    # Also bound total serialized expression size, including wide nested expressions.
    if len(canonical(trees).encode()) > 32768:
        raise ValueError('tree byte limit')
    walk(trees['value'], False)
    walk(trees['rank'], True)


def traverse(tree, features):
    path = []
    while 'cost' not in tree:
        value = features.get(tree['feature'])
        if value is None:
            branch = 'unknown'
        else:
            threshold = tree['value']
            op = tree['op']
            yes = {'eq': lambda: value == threshold, 'lt': lambda: value < threshold,
                   'le': lambda: value <= threshold, 'gt': lambda: value > threshold,
                   'ge': lambda: value >= threshold}[op]()
            branch = 'yes' if yes else 'no'
        path.append(dict(feature=tree['feature'], value=value, branch=branch))
        tree = tree[branch]
    return tree, path


class SemanticRuntime:
    def __init__(self, initial, update, features, trees, knowledge_id, emit=None):
        validate_trees(trees)
        self.initial, self.update, self.features = initial, update, features
        self.trees, self.knowledge_id = deepcopy(trees), knowledge_id
        self.emit = emit or (lambda row: print(canonical(row), file=sys.stderr, flush=True))
        self.real_memory = self.checked_memory(initial())
        self.root_decision = None
        self.root_frame = None
        self.branches = {}

    @staticmethod
    def checked_memory(memory):
        if not isinstance(memory, dict) or len(canonical(memory).encode()) > 8192:
            raise ValueError('memory must be a JSON object of at most 8192 bytes')
        return deepcopy(memory)

    def frame(self, node):
        observation = visible(node['observation'])
        state = node.get('state')
        estimate = None if state is None else {
            'object': pick(state['object'], KINEMATICS_FIELDS),
            'robots': [pick(r, KINEMATICS_FIELDS) for r in state['robots']]}
        return dict(source=node['source'], observation=observation, estimate=estimate,
                    goal=deepcopy(node['goal']), elapsed_ticks=node['elapsed_ticks'])

    @staticmethod
    def path_key(path):
        return tuple((a['a0'], a['a1'], a['ticks']) for a in path)

    def register(self, node, context):
        frame = self.frame(node)
        key = self.path_key(node['path_context'])
        if not key:
            decision = frame['observation']['decision']
            if decision != self.root_decision:
                if self.root_decision is not None and decision <= self.root_decision:
                    raise ValueError('new episode requires a fresh runtime')
                event = dict(frame=frame, action=deepcopy(context.get('previous_action')),
                             executed_ticks=None)
                if self.root_frame is not None:
                    now = frame['observation']['robots'][0]['elapsed_seconds']
                    before = self.root_frame['observation']['robots'][0]['elapsed_seconds']
                    event['executed_ticks'] = round((now - before) / .05)
                    if event['executed_ticks'] < 0:
                        raise ValueError('real clock moved backwards')
                self.real_memory = self.checked_memory(self.update(deepcopy(self.real_memory), deepcopy(event)))
                self.root_decision, self.root_frame = decision, deepcopy(frame)
                self.branches = {(): (deepcopy(self.real_memory), frame)}
            elif frame != self.root_frame:
                raise ValueError('same real boundary changed its visible input')
        elif key not in self.branches:
            parent_memory, parent_frame = self.branches[key[:-1]]
            delta = frame['elapsed_ticks'] - parent_frame['elapsed_ticks']
            if not 0 <= delta <= node['incoming_action']['ticks']:
                raise ValueError('invalid predicted elapsed duration')
            event = dict(frame=frame, action=deepcopy(node['incoming_action']), executed_ticks=delta)
            branch_memory = self.checked_memory(self.update(deepcopy(parent_memory), deepcopy(event)))
            self.branches[key] = (branch_memory, frame)
        return self.branches[key]

    def feature_values(self, memory, frame, action):
        values = self.features(deepcopy(memory), deepcopy(frame), deepcopy(action))
        if not isinstance(values, dict) or len(values) > 64 or len(canonical(values).encode()) > 8192:
            raise ValueError('at most 64 bounded scalar semantic features')
        for key, value in values.items():
            if not isinstance(key, str) or (value is not None and type(value) not in (bool, int, float)):
                raise ValueError('features are named scalar values or null/unknown')
            if type(value) in (int, float):
                number(value)
        return values

    def evaluate(self, node, context):
        memory, frame = self.register(node, context)
        features = self.feature_values(memory, frame, None)
        leaf, path = traverse(self.trees['value'], features)
        cost = expression(leaf['cost'], features, None)
        self.log('value', node, features=features, branch=path, cost=cost, memory_sha256=digest(memory))
        return cost

    def rank(self, node, candidates, context):
        memory, frame = self.register(node, context)
        rows = []
        for action in candidates:
            features = self.feature_values(memory, frame, action)
            leaf, path = traverse(self.trees['rank'], features)
            rows.append(dict(action=action, features=features, branch=path,
                             cost=expression(leaf['cost'], features, action)))
        features = self.feature_values(memory, frame, None)
        leaf, path = traverse(self.trees['rank'], features)
        proposals = []
        for spec in leaf.get('proposals', []):
            action = {k: expression(spec[k], features, None) for k in ('a0', 'a1', 'ticks')}
            if any(type(v) is not int for v in action.values()) or not 1 <= action['ticks'] <= 40:
                raise ValueError('proposal requires integer command')
            if [action['a0'], action['a1']] not in frame['observation']['legal_pairs']:
                raise ValueError('proposal is outside observed legal menu')
            proposals.append(dict(action=action, cost=expression(spec['cost'], features, None)))
        self.log('rank', node, candidates=rows, proposals=proposals, proposal_branch=path)
        return dict(costs=[r['cost'] for r in rows], proposals=proposals)

    def log(self, stage, node, **fields):
        self.emit(dict(kind='history_semantic_v0', knowledge_id=self.knowledge_id,
                       decision=self.root_decision, node_id=node['id'], source=node['source'],
                       stage=stage, **fields))
