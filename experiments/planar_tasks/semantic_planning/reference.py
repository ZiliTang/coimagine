"""Hand-written interface seed, NOT a learned policy or a task-solving macro."""
import math


def initial_memory():
    return {'real_samples': 0, 'synthetic_samples': 0, 'last_both_contact_time': None}


def update_memory(memory, event):
    frame = event['frame']
    synthetic = frame['source'] == 'predicted'
    memory['synthetic_samples' if synthetic else 'real_samples'] += 1
    if all(r.get('contacting_object') is True for r in frame['observation']['robots']):
        memory['last_both_contact_time'] = frame['observation']['robots'][0]['elapsed_seconds']
    return memory


def semantic_features(memory, frame, action):
    estimate = frame['estimate']
    distance = None
    if estimate is not None:
        obj = estimate['object']
        distance = math.hypot(obj['x'] - frame['goal'][0], obj['y'] - frame['goal'][1])
    return {'estimate_available': estimate is not None, 'distance': distance,
            'has_contact_evidence': memory['last_both_contact_time'] is not None,
            'real_samples': memory['real_samples'], 'synthetic_samples': memory['synthetic_samples']}


# Neutral distance estimate; action costs tie. Contact history is logged but does
# not yet select a hand-coded recovery. Evidence/time labels are not confidence.
TREES = {
    'value': {'feature': 'estimate_available', 'op': 'eq', 'value': True,
              'yes': {'cost': {'mul': [3.3333333333333335, {'feature': 'distance'}]}},
              'no': {'cost': 120.0}, 'unknown': {'cost': 120.0}},
    'rank': {'feature': 'estimate_available', 'op': 'eq', 'value': True,
             'yes': {'cost': {'mul': [3.3333333333333335, {'feature': 'distance'}]}},
             'no': {'cost': 120.0}, 'unknown': {'cost': 120.0}}
}
