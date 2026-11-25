"""Isolated text-only CAP author calls and authoring_v2 seals for composed policies."""

import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from experiments.planar_tasks.authoring_v2 import random_room_author as host

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-6.2-sol"
EFFORT = "high"
CALL_SECONDS = 360
SOURCE_BYTES = 10_000
REVISIONS = (1, 2)
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "source": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["source", "rationale"],
}


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def make_prompt(public_packet, interface_text, parent_sources, revision, feedback):
    guides = {p: (ROOT / p).read_text() for p in host.GUIDES}
    return (
        "You are the CAP author for one round of a two-robot simulated transport study. "
        "The supplied context is complete. Do not use tools, read files, simulate, or query the web. "
        "Deliver your own causal module as JSON source and rationale. "
        "The development rooms use fixed start and goal poses, one stationary and one movable obstacle. "
        "The history and observations visible at execution are the only deployment inputs. "
        "Case IDs, future outcomes, offline graphs, simulator state and hidden geometry cannot be deployment inputs. "
        "Look for common failures and natural recoveries. Add one local capability with a narrow public entry, "
        "progress check, and give-back condition. Prioritize whole-episode rescues while retaining parent successes. "
        "A recoverable observation maneuver may be used when its additional physical time is justified. "
        "Return only the two top-level functions module_update(local,event,memory) and "
        "module_act(local,memory,observation,base_action), max 10000 UTF-8 bytes. "
        "They must be pure, deterministic functions. module_update returns an updated dict from an initial {}. "
        "module_act returns a valid complete joint action or None to defer. No import, globals, "
        "extra dependencies, top-level helpers, test code or Markdown. "
        "The specific interface below takes precedence over examples and historical tool menus in the common guides. "
        "There are two author calls per round; the second receives the first candidate's development feedback.\n"
        + json.dumps(dict(interface=interface_text, public_guides=guides, immutable_sources=parent_sources,
                          development_packet=public_packet, revision=revision, prior_feedback=feedback),
                     ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def check_response(response, *, source_limit=SOURCE_BYTES):
    if set(response) != {"source", "rationale"} or not all(isinstance(s, str) for s in response.values()):
        raise ValueError("author reply must contain source and rationale strings")
    source = response["source"]
    if len(source.encode()) > source_limit:
        raise ValueError(f"module exceeds {source_limit} bytes")
    functions = ast.parse(source).body
    if len(functions) != 2 or any(type(f) is not ast.FunctionDef for f in functions):
        raise ValueError("expected exactly two module function definitions")
    contract = {"module_update": ["local", "event", "memory"],
                "module_act": ["local", "memory", "observation", "base_action"]}
    if {f.name for f in functions} != set(contract):
        raise ValueError("module_update and module_act required")
    for f in functions:
        a = f.args
        if (f.decorator_list or f.returns or a.defaults or a.kwonlyargs or a.posonlyargs or
                a.vararg or a.kwarg or any(v.annotation for v in a.args) or
                [v.arg for v in a.args] != contract[f.name]):
            raise ValueError(f"unexpected signature of {f.name}")
    return source


def propose(round_dir, public_packet, interface_text, parent_sources, revision, feedback=None):
    """One metered call; events and input digest persist even on failure."""
    round_dir = Path(round_dir)
    if revision not in REVISIONS:
        raise ValueError("two candidate slots per round")
    round_dir.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(public_packet, interface_text, parent_sources, revision, feedback)
    slot = round_dir / f"call_{revision}"
    if slot.exists():
        record = host.read(slot / "request.json")
        if record["prompt_sha"] != sha(prompt):
            raise ValueError("existing author call used a different prompt")
        return check_response(host.read(slot / "response.json"))
    slot.mkdir()
    host.save(slot / "request.json", dict(model=MODEL, effort=EFFORT, prompt_sha=sha(prompt),
                                           prompt_bytes=len(prompt.encode()), seconds=CALL_SECONDS,
                                           source_bytes=SOURCE_BYTES, started=time.time()))
    (slot / "prompt.txt").write_text(prompt)
    started = time.monotonic()
    # The paired author sessions run through the orchestrator's model-specific backend.
    # A transport failure before model generation is preserved separately.
    import time as clock
    reply = slot / "response.json"
    while clock.monotonic() - started < CALL_SECONDS:
        if reply.exists():
            response = host.read(reply)
            host.save(slot / "accounting.json", dict(elapsed=clock.monotonic()-started,
                      provider="collaboration", model=MODEL, usage=None, finished=clock.time()))
            return check_response(response)
        clock.sleep(.5)
    raise TimeoutError(f"author did not reply in {CALL_SECONDS}s: {slot}")



def seal(round_dir, source, fixtures):
    """Freeze each composed candidate and the identical public knowledge via the v2 campaign host."""
    root = Path(round_dir).resolve() / "campaign"
    path = root / "composed.py"
    if root.exists() and (root / "manifest.json").is_file():
        campaign = host.RoomCampaign(root)
        result = campaign.verify_frozen()
        if path.read_text() != source or host.read(root / "declared_fixtures.json") != {
                k: sha(json.dumps(v, sort_keys=True)) for k, v in fixtures.items()}:
            raise ValueError("frozen candidate has different source or development fixtures")
        return result
    if root.exists():
        raise ValueError(f"incomplete candidate seal: {root}")
    campaign = host.RoomCampaign.create(root, fixtures)
    with campaign.locked():
        campaign.state["model"].update(name=MODEL, reasoning_effort=EFFORT,
                                       provider="collaboration", filesystem_isolation="author instructed to access only per-slot supplied prompt and reply")
        campaign.state["author_protocol"].update(version="graph-prefix-module-v1", max_revisions=len(REVISIONS),
                                                 author_call_seconds=CALL_SECONDS, source_bytes=SOURCE_BYTES,
                                                 no_prior_semantics=False)
        campaign._save()
    path.write_text(source)
    host.save(root / "declared_fixtures.json", {k: sha(json.dumps(v, sort_keys=True)) for k,v in fixtures.items()})
    host.save(root / "composed_source.json", dict(sha256=sha(source), path="composed.py"))
    campaign.freeze()
    return campaign.verify_frozen()
