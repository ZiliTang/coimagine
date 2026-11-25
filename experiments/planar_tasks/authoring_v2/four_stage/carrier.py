"""Checked-policy carrier for the versioned host-prepared four-stage interface.

The host preparation service is bound in the campaign source manifest. Compact
public-derived values fit the existing typed-stream memory budget; the submitted
Python includes the exact third-stage function used by the external checker.
"""
import ast
import hashlib
import json

from .compiler import Extractor
from .frontend import schema, OPTIONS


WRAPPER = '''
def Q(value, units):
    return value

def initial_memory():
    return {}

def update_memory(memory, event):
    return memory

def prepare_observations(memory, observation):
    # The versioned public preparation service fills these causal summaries.
    if memory.get("four_stage_tick") != observation["tick"]:
        raise ValueError("missing or stale four-stage preparation")
    values = memory.get("four_stage_values", [])
    if len(values) != len(SEMANTIC_REGISTRY):
        raise ValueError("semantic registry/value mismatch")
    return values

def register_symbols(prepared):
    return {name: prepared[i] for i, name in enumerate(SEMANTIC_REGISTRY)}

def check_candidates(memory, observation, z, candidates):
    permissions = memory.get("four_stage_permissions", {})
    ordered = sorted(enumerate(candidates), key=lambda item: (-item[1][1], item[0]))
    for index, item in ordered:
        name, score = item
        if z["legal:" + name] and permissions.get(name, False):
            spec = ACTION_COMMANDS[name]
            return name, dict(target_id=observation["goal"]["target_id"] if spec[3] else None,
                              a0=spec[0], a1=spec[1], ticks=spec[2])
    return None, None

def choose(memory, observation):
    prepared = prepare_observations(memory, observation)
    z = register_symbols(prepared)
    return check_candidates(memory, observation, z, decide(z))

def prepare_context(memory, observation):
    name, command = choose(memory, observation)
    if name and name.startswith("push_"):
        review = review_push(memory, observation, [command["a0"], command["a1"]])
        if review["verdict"] != "pass":
            raise ValueError("selected push contradicts checked permissions")
    return dict(values=[0.], state=[observation["tick"]], proposals=[command] if command else [])

def review_push(memory, observation, pair):
    z = register_symbols(prepare_observations(memory, observation))
    permissions = memory.get("four_stage_permissions", {})
    matches = [name for name, spec in ACTION_COMMANDS.items()
               if spec[3] and min(spec[0], spec[1]) < 32 and [spec[0], spec[1]] == pair]
    allowed = any(z["legal:" + name] and permissions.get(name, False) for name in matches)
    return dict(verdict="pass" if allowed else "unknown", peer_robots=[])

def audit_policy(memory, observation, menu):
    z = register_symbols(prepare_observations(memory, observation))
    candidates = decide(z)
    proposed = []
    for name, score in candidates:
        spec = ACTION_COMMANDS[name]
        pair = [spec[0], spec[1]]
        if spec[3] and min(pair) < 32 and pair in menu and pair not in proposed:
            proposed.append(pair)
    name, command = choose(memory, observation)
    selected = [command["a0"], command["a1"]] if name and name.startswith("push_") else None
    return dict(version=1, proposed=proposed, selected=selected,
                claim="selected" if selected else "abstain", widths=[], peer_predictions=[])

def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        out[i, 0] = 0.
'''


def build(decision_source):
    host = schema()
    p = Extractor(decision_source, host)
    function = ast.unparse(p.function)
    commands = {name: [*opt["pair"], opt["ticks"], opt["family"] in ("push", "orbit")]
                for name, opt in OPTIONS.items()}
    source = ('import cython\n'
              'SEMANTIC_API = "typed-stream-v1"\n'
              'NAMES = ["constant"]\nSTATE_NAMES = ["tick"]\nHISTORY_LENGTH = 1\n'
              'POLICY_CHECK_RULES = {"version": 1, "widths": {}, "peer_models": {}}\n'
              'FOUR_STAGE_PROFILE = "host-public-preparation-v1"\n'
              'SEMANTIC_REGISTRY = ' + repr(list(host["metrics"])) + '\n'
              'ACTION_COMMANDS = ' + repr(commands) + '\n' + function + '\n' + WRAPPER)
    if len(source.encode()) > 49152:
        raise ValueError("carrier exceeds the checked typed-stream source limit")
    return source


def verify(source):
    parsed = ast.parse(source)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "decide"]
    if len(functions) != 1:
        raise ValueError("missing or duplicate third stage")
    regenerated = build(ast.unparse(functions[0]))
    if ast.dump(parsed) != ast.dump(ast.parse(regenerated)):
        raise ValueError("carrier differs from its declared public preparation/checking profile")
    return Extractor(source, schema())


def memory_packet(prepared, permissions):
    z = prepared["symbols"]
    packet = dict(four_stage_tick=prepared["observation"]["tick"],
                  four_stage_values=[z[k] for k in schema()["metrics"]],
                  four_stage_permissions={k: bool(v) for k, v in permissions.items()})
    if len(json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()) > 8192:
        raise ValueError("prepared public memory exceeds typed-stream limit")
    return packet


def checked_boundary(source, prepared, permissions):
    return dict(observation=prepared["observation"], memory=memory_packet(prepared, permissions),
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
