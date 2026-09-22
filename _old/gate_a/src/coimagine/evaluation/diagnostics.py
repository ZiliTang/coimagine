from __future__ import annotations

from coimagine.core.types import OptionKind, PlannerMode
from coimagine.envs.grid.generation.procedural import ScenarioInstance
from coimagine.evaluation.events import TopologyDelta
from coimagine.planning.exact_search import ExactTopologySearch, SearchResult


def _trace_has_kind(result: SearchResult, depth: int, kind: OptionKind, entity_id: str | None = None) -> bool:
    if len(result.trace) <= depth:
        return False
    for option in result.trace[depth].joint_option.options:
        if option.kind == kind and (entity_id is None or option.entity_id == entity_id):
            return True
    return False


def validate_diagnostic_mechanism(
    instance: ScenarioInstance,
    evolving: SearchResult,
    fixed: SearchResult,
    delta: TopologyDelta,
    planner: ExactTopologySearch,
) -> tuple[bool, str]:
    """Return an explicit mechanism acceptance result for controlled diagnostics.

    Natural instances are never filtered by this function; diagnostic validity is
    reported so pilot runs can detect a generator that no longer realizes its named
    mechanism after code changes.
    """

    task = instance.task_name
    if instance.suite != "diagnostic":
        return True, "Natural instance; no post-result mechanism filter applied."

    root_candidates = evolving.root_admission.candidate_options
    exact_edges = {joint.template_key for joint in evolving.root_admission.joint_options}

    if task == "edge_creation":
        valid = (
            delta.created_edge_count > 0
            and evolving.value > fixed.value + 1e-9
            and evolving.joint_option.stable_key != fixed.joint_option.stable_key
            and _trace_has_kind(evolving, 1, OptionKind.PUSH)
        )
        return valid, (
            f"created={delta.created_edge_count}, destroyed={delta.destroyed_edge_count}; "
            f"gain={evolving.value-fixed.value:.3f}; continuation_push={_trace_has_kind(evolving, 1, OptionKind.PUSH)}"
        )

    if task == "edge_destruction":
        valid = delta.dominant_event == "pure_edge_destruction" and delta.destroyed_edge_count > 0
        return valid, (
            f"dominant={delta.dominant_event}; created={delta.created_edge_count}; "
            f"destroyed={delta.destroyed_edge_count}"
        )

    if task == "edge_replacement":
        valid = (
            delta.created_edge_count > 0
            and delta.destroyed_edge_count > 0
            and _trace_has_kind(evolving, 0, OptionKind.PUSH, "blocker-A")
            and _trace_has_kind(evolving, 1, OptionKind.PUSH, "blocker-B")
        )
        return valid, (
            f"created={delta.created_edge_count}, destroyed={delta.destroyed_edge_count}; "
            f"A_then_B={_trace_has_kind(evolving, 0, OptionKind.PUSH, 'blocker-A') and _trace_has_kind(evolving, 1, OptionKind.PUSH, 'blocker-B')}"
        )

    if task == "reveal_act":
        valid = (
            "information_reveal" in delta.labels
            and evolving.joint_option.stable_key != fixed.joint_option.stable_key
            and _trace_has_kind(evolving, 0, OptionKind.OBSERVE, "hidden-blocker")
            and _trace_has_kind(evolving, 1, OptionKind.PUSH, "hidden-blocker")
            and evolving.value > fixed.value + 1e-9
        )
        return valid, (
            f"revealed={sorted(delta.revealed_entities)}; reversal={evolving.joint_option.stable_key != fixed.joint_option.stable_key}; "
            f"observe_then_hidden_push={_trace_has_kind(evolving, 0, OptionKind.OBSERVE, 'hidden-blocker') and _trace_has_kind(evolving, 1, OptionKind.PUSH, 'hidden-blocker')}"
        )

    if task == "hold_pass":
        second = evolving.trace[1].joint_option.options if len(evolving.trace) > 1 else ()
        has_hold = any(option.kind == OptionKind.HOLD for option in second)
        has_pass = any(option.role_token == "pass_goal" for option in second)
        valid = len(evolving.trace) > 1 and has_hold and has_pass and evolving.value > fixed.value + 1e-9
        return valid, f"two_step={len(evolving.trace)>1}; second_has_hold={has_hold}; second_has_pass={has_pass}"

    if task in {"positive_higher_order_synergy", "higher_order_conflict"}:
        pairwise = planner.admission.admit(
            instance.state,
            root_candidates,
            PlannerMode.PAIRWISE_ONLY,
        )
        pairwise_edges = {joint.template_key for joint in pairwise.joint_options}
        if task == "positive_higher_order_synergy":
            missed = exact_edges - pairwise_edges
            valid = any(
                len(edge) == 3 and all(option_key[1] == OptionKind.PUSH.value for option_key in edge)
                for edge in missed
            )
            return valid, f"exact_edges={len(exact_edges)}; pairwise_edges={len(pairwise_edges)}; missed={len(missed)}"
        false_positive = pairwise_edges - exact_edges
        valid = any(
            len(edge) == 3 and all(option_key[1] == OptionKind.PUSH.value for option_key in edge)
            for edge in false_positive
        )
        return valid, (
            f"exact_edges={len(exact_edges)}; pairwise_edges={len(pairwise_edges)}; "
            f"false_positive={len(false_positive)}"
        )

    return True, "No dedicated diagnostic acceptance rule."
