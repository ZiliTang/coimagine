"""Neutral incremental format example; no task-specific successful policy."""
import cython
SEMANTIC_API = 'typed-stream-v1'
NAMES = ['duration_match']
STATE_NAMES = ['contact_fraction', 'blocked_fraction', 'observed_entities']
HISTORY_LENGTH = 8


def initial_memory() -> dict:
    return {'ticks': 0, 'contact': 0, 'blocked': 0}


def update_memory(memory: dict, event: dict) -> dict:
    dt = event['executed_ticks'] or 0
    robots = event['observation']['robots']
    memory['ticks'] += dt
    memory['contact'] += dt * sum(bool(r['contacting_object']) for r in robots)
    memory['blocked'] += dt * sum(r['execution'].get('stage') in ('blocked', 'move_blocked') for r in robots)
    return memory


def prepare_context(memory: dict, observation: dict) -> dict:
    denom = max(1, 2 * memory['ticks'])
    state = [memory['contact'] / denom, memory['blocked'] / denom, float(len(observation['entities']))]
    return dict(values=[20.0], state=state,
                proposals=[dict(target_id=None, a0=44, a1=44, ticks=20)])


def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        if candidates[i, 3] < 0:
            out[i, 0] = float('nan')
        else:
            out[i, 0] = -abs(candidates[i, 3] - context[0]) / 40.0
