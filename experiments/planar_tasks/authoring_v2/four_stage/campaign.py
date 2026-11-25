"""Versioned authoring_v2 experiment entry for the four-stage host adapter."""
from pathlib import Path
import hashlib

from ..relational_transfer_author import GUIDES
from experiments.planar_tasks.relational_search.pipeline import Campaign, ROOT, read, save
from experiments.planar_tasks.relational_search.contract import digest
from .carrier import verify


EXTRA_GUIDES = [
    "experiments/planar_tasks/authoring_v2/checked_policy_guide.md",
    "experiments/planar_tasks/authoring_v2/checked_policy_example.py",
    "experiments/planar_tasks/authoring_v2/four_stage/README.md",
    "docs/research/future_directions/2026-11-25_cap_author_packet.md",
    "experiments/planar_tasks/authoring_v2/four_stage/geometry_kernel.cpp",
    "tools/cap_four_stage_campaign.py",
]


def create(path, fixture, source, protocol):
    verify(source)
    c = Campaign.create(path, fixture=fixture,
        sequence=[dict(target_id=None, a0=44, a1=44, ticks=4)],
        model=dict(provider="native-current-thread", name="Codex",
                   clean_context=False, filesystem_isolation="none for author; public-only deployment adapter"))
    sources = read(c.path / "sources.json")
    for i, relative in enumerate([*GUIDES, *EXTRA_GUIDES]):
        p = ROOT / relative
        data = p.read_bytes()
        dest = c.path / "sources" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        sources[relative] = hashlib.sha256(data).hexdigest()
        (c.path / "knowledge" / f"four_stage_{i}_{p.name}").write_bytes(data)
    save(c.path / "sources.json", sources)
    state = read(c.path / "campaign.json")
    state.update(profile="authoring-v2-four-stage-v1", source_hash=digest(sources),
                 author_protocol=protocol,
                 accounting="Physical work reserved and settled in adjacent runs; this immutable package binds author and host sources.",
                 llm_usage=dict(requests=None, input_tokens=None, output_tokens=None,
                                provenance="Current conversation authoring; no separate provider call or clean-context author claimed."))
    save(c.path / "campaign.json", state)
    (c.path / "policy.py").write_text(source)
    save(c.path / "protocol.json", protocol)
    return c


def freeze(path, checked):
    from ..checked_submission import verify_checked
    receipt = verify_checked(Path(checked))
    c = Campaign(path)
    source = (c.path / "policy.py").read_bytes()
    manifest = read(Path(checked) / "manifest.json")
    if manifest["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("checked submission differs from campaign policy")
    save(c.path / "checked_receipt.json", dict(path=str(Path(checked).resolve()), receipt=receipt,
                                               manifest=manifest))
    return c.freeze()


def verify_frozen(path, *, live=True):
    c = Campaign(path)
    result = c.verify_frozen()
    if live:
        c.verify_sources()
    verify((c.path / "policy.py").read_text())
    return result
