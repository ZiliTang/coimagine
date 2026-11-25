"""Auditable first-supported-push example. The review concerns visible push-face width."""
import cython
from math import cos, pi, sin

SEMANTIC_API = 'typed-stream-v1'
NAMES = ['constant']
STATE_NAMES = ['observed_tick']
HISTORY_LENGTH = 1
POLICY_CHECK_RULES = {
    'version': 1,
    'widths': {'observed_lower_bound': ['sufficient', 'unknown']},
    'peer_models': {},
}


def initial_memory():
    return {}


def update_memory(memory, event):
    return memory


def review_push(memory, observation, pair):
    required = .10
    widths = []
    for index, action in enumerate(pair):
        if action >= 32:
            continue
        angle = (action // 4) * pi / 4
        spans = [patch['span_m'] for patch in observation['robots'][index]['patches']
                 if patch['valid'] and observation['active_target'] == observation['goal']['target_id']
                 and cos(angle)*patch['normal_world'][0] + sin(angle)*patch['normal_world'][1] > .9]
        widths.append(max(spans, default=0.))
    lower = min(widths)
    sufficient = lower >= required
    fact = dict(source='observed_lower_bound', lower_m=lower, upper_m=None,
                required_m=required, conclusion='sufficient' if sufficient else 'unknown')
    return dict(verdict='pass' if sufficient else 'unknown', peer_robots=[], widths=[fact])


def choose(memory, observation, menu):
    checked = []
    for pair in menu:
        checked.append(pair)
        if review_push(memory, observation, pair)['verdict'] == 'pass':
            return checked, pair
    return checked, None


def prepare_context(memory, observation):
    target = observation['goal']['target_id']
    seen = any(e['id'] == target and e['points'] and e['last_seen_tick'] == observation['tick']
               for e in observation['entities'])
    menu = [p for p in observation['entity_legal_pairs'].get(target, []) if min(p) < 32] if seen else []
    _, selected = choose(memory, observation, menu)
    first = dict(target_id=target, a0=selected[0], a1=selected[1], ticks=10) if selected else (
        dict(target_id=None, a0=44, a1=44, ticks=5))
    return dict(values=[0.], state=[observation['tick']], proposals=[first])


def audit_policy(memory, observation, menu):
    proposed, selected = choose(memory, observation, menu)
    return dict(version=1, proposed=proposed, selected=selected,
                claim='selected' if selected else 'no_passing_push', widths=[], peer_predictions=[])


def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        out[i, 0] = 0.
