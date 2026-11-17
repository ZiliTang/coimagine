"""API example only: BRAKE recommendation, not a task-solving baseline."""
import cython

SEMANTIC_API = 'typed-batch-v1'
NAMES = ['contact_fraction', 'duration_error', 'recommendation_match']


def initial_memory() -> dict:
    return {}


def update_memory(memory: dict, event: dict) -> dict:
    return memory


def prepare_context(memory: dict, observation: dict) -> dict:
    contact = sum(bool(r['contacting_object']) for r in observation['robots']) / 2.0
    return dict(values=[20., contact], proposals=[dict(target_id=None, a0=44, a1=44, ticks=20)])


def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        out[i, 0] = context[1]
        if candidates[i, 3] >= 0:
            out[i, 1] = -abs(candidates[i, 3] - context[0]) / 40.
        out[i, 2] = float((candidates[i, 0] == -1 or candidates[i, 0] == -2)
                         and (candidates[i, 1] == -1 or candidates[i, 1] == 44)
                         and (candidates[i, 2] == -1 or candidates[i, 2] == 44)
                         and (candidates[i, 3] == -1 or candidates[i, 3] == context[0]))
