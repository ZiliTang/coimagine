from __future__ import annotations

from coimagine.core.types import OptionKind, PlannerMode

BASELINE_DISPLAY_NAMES = {
    PlannerMode.COST.value: "CST",
    PlannerMode.COST_FROZEN_K.value: "CST-Frozen-K",
    PlannerMode.IET.value: "CoImagine-IET",
    PlannerMode.PRIVILEGED_IET_ORACLE.value: "Privileged IET Oracle",
    PlannerMode.EVOLVING.value: "Oracle Evolving Topology",
    PlannerMode.ROOT_STRUCTURE_REUSE.value: "Root-Structure Reuse",
    PlannerMode.STRICT_ROOT_REUSE.value: "Strict Root-Candidate Reuse",
    PlannerMode.LOCAL_AVAILABILITY.value: "Local Availability Only",
    PlannerMode.PAIRWISE_ONLY.value: "Pairwise-Clique Verification",
    PlannerMode.RECURSIVE_FACTORS.value: "Recursive Exact Factors",
    PlannerMode.FULL_TUPLE_ORACLE.value: "Exact Full-Tuple Oracle",
}

TASK_DISPLAY_NAMES = {
    "local_imagination_chain": "Local-observation imagination chain",
    "multi_blocker_clearing": "Multi-blocker clearing",
    "complex_channel_clearing": "Mixed-mass channel clearing",
    "congested_transport": "Congested transport",
    "topology_stable_control": "Topology-stable control",
    "edge_creation": "Coordination-edge creation",
    "edge_destruction": "Coordination-edge destruction",
    "edge_replacement": "Coordination-edge replacement",
    "positive_higher_order_synergy": "Positive three-robot synergy",
    "higher_order_conflict": "Three-robot resource conflict",
    "hold_pass": "Temporary hold-pass coordination",
    "reveal_act": "Information reveal-then-act",
}

EVENT_DISPLAY_NAMES = {
    "pure_edge_creation": "Pure edge creation",
    "pure_edge_destruction": "Pure edge destruction",
    "mixed_edge_turnover": "Mixed edge turnover",
    "information_reveal": "Information reveal",
    "no_change": "No topology change",
}

OPTION_DISPLAY_NAMES = {
    OptionKind.REPOSITION: "Reposition",
    OptionKind.APPROACH: "Approach",
    OptionKind.PUSH: "Push",
    OptionKind.HOLD: "Hold",
    OptionKind.OBSERVE: "Observe",
    OptionKind.IDLE: "Idle",
}

# Stable semantic colors used only in publication/review figures.
OPTION_COLORS = {
    OptionKind.REPOSITION: "#6C8EBF",
    OptionKind.APPROACH: "#4C78A8",
    OptionKind.PUSH: "#E45756",
    OptionKind.HOLD: "#72B7B2",
    OptionKind.OBSERVE: "#F2CF5B",
    OptionKind.IDLE: "#9D9D9D",
}

BASELINE_DEFINITIONS = {
    "CST": (
        "Current-Observation-Supported Topology re-instantiates the complete template "
        "bank grounded by the current real observations at every predicted depth, "
        "then reapplies the common top-K candidate budget."
    ),
    "CST-Frozen-K": (
        "Diagnostic baseline that freezes only the templates selected into the root "
        "top-K candidate bank; it quantifies artificial loss caused by root truncation."
    ),
    "CoImagine-IET": (
        "Imagined Evolving Topology uses optimistic rule-based local-belief "
        "imagination to regenerate future observation-supported options, factors, "
        "and coordination hyperedges."
    ),
    "Privileged IET Oracle": (
        "Uses the same root information and action set as IET, but renders the true "
        "successor local observations from the simulator before regenerating the "
        "future topology."
    ),
    "Oracle Evolving Topology": (
        "Regenerates local candidates, factors, and exact executable joint-option edges "
        "on every imagined branch using exact GridWorld transitions."
    ),
    "Root-Structure Reuse": (
        "Updates successor state and state-dependent option parameters, but projects "
        "successor candidates onto root option-template slots and reuses root joint-template edges."
    ),
    "Strict Root-Candidate Reuse": (
        "Reuses the concrete root candidates, including their original metric targets and contacts, "
        "at all imagined depths."
    ),
    "Local Availability Only": (
        "Regenerates each robot's local candidates independently and admits their Cartesian product "
        "without factor-level joint-executability checks."
    ),
    "Pairwise-Clique Verification": (
        "For higher-order tuples, requires every singleton and pair to be factor-executable; it can "
        "miss positive higher-order synergy and admit irreducible higher-order conflicts."
    ),
    "Recursive Exact Factors": (
        "Recursively retains partial factor tuples when they are exactly extendable and emits complete "
        "tuples when exactly executable, up to arity P."
    ),
    "Exact Full-Tuple Oracle": (
        "Enumerates and exactly executes the complete Cartesian product; it is the admission-set "
        "reference for recall, precision, and verifier-call comparisons."
    ),
}

TOPOLOGY_DEFINITIONS = {
    "Root topology": (
        "Exact candidate-induced coordination topology at the current option boundary before the "
        "selected first joint option is executed."
    ),
    "Regenerated successor topology": (
        "Exact topology rebuilt from successor candidates after the selected first joint option and "
        "the resulting observation/belief update."
    ),
    "Local-option node": (
        "One robot-indexed local option; nodes are not robots or physical objects."
    ),
    "Executable joint-option edge": (
        "A complete one-option-per-robot combination that reaches a valid next option boundary under "
        "the exact GridWorld contracts. For N>2 it is a hyperedge, drawn as a polyline."
    ),
    "Created edge": "A successor joint-option template absent from the root topology.",
    "Destroyed edge": "A root joint-option template absent from the successor topology.",
    "Persistent edge": "A joint-option template present at both boundaries.",
    "Selected edge": "The first or continuation joint option selected by the evolving planner.",
}

METRIC_DEFINITIONS = {
    "Oracle gain": (
        "Delta_oracle = J*_evolve - J*_reuse, with exact state, transition, task consequence, and "
        "executability."
    ),
    "First-action regret": (
        "The evolving-oracle value loss incurred by executing the root action selected by "
        "Root-Structure Reuse."
    ),
    "First-action reversal": "Whether the two planners select different concrete root joint options.",
    "Rank agreement": (
        "Fraction of pairwise action-order comparisons that agree on the root actions shared by both planners."
    ),
    "Oracle recall": "Fraction of exact executable full tuples retained by an approximate admission method.",
    "Oracle precision": "Fraction of a method's admitted full tuples that are exactly executable.",
}

TASK_DEFINITIONS = {
    "Mixed-mass channel clearing": (
        "Three robots clear a transit corridor containing a two-robot heavy blocker and one-robot light blockers. "
        "One rear contact becomes available only after the heavy blocker moves; all robots retain the same universal option library."
    ),
    "Multi-blocker clearing": (
        "A procedural corridor-clearing task with one or more movable blockers. Robots may act on "
        "different entities in parallel; blocker count controls entity-level combinatorics."
    ),
    "Congested transport": (
        "A procedural payload-transport task with narrow shared resources and multi-robot coupling. "
        "It is used to probe robot count, resource capacity, and higher-order conflicts."
    ),
    "Topology-stable control": (
        "A negative-control task whose executable joint-option template set is intended to remain "
        "stable over the evaluated horizon; topology regeneration should provide no structural gain."
    ),
    "Coordination-edge creation": (
        "Robots first reposition; the successor candidate bank contains a valuable coordinated push "
        "edge absent at the root. The transition can also destroy obsolete approach edges."
    ),
    "Coordination-edge destruction": (
        "A valid root coordination relation is consumed or becomes unavailable after execution; the "
        "diagnostic acceptance rule requires destroyed edges and no created edges."
    ),
    "Coordination-edge replacement": (
        "Execution removes a root coordination relation on one entity and creates a different relation "
        "on another entity; both created and destroyed edge sets must be nonempty."
    ),
    "Positive three-robot synergy": (
        "Singleton and pair push tuples are not executable, while a three-robot tuple is executable; "
        "low-order tuples remain strictly extendable."
    ),
    "Three-robot resource conflict": (
        "Every singleton and pair can use a shared resource, but the three-robot tuple exceeds its "
        "capacity and is infeasible."
    ),
    "Temporary hold-pass coordination": (
        "A support robot must reach and execute Hold while the payload robot traverses the guarded passage."
    ),
    "Information reveal-then-act": (
        "Observe reveals a hidden blocker and creates a high-value push candidate; an immediately "
        "available lower-value push provides the competing short-sighted root action."
    ),
}


def baseline_display_name(mode: str) -> str:
    return BASELINE_DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())


def task_display_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())


def event_display_name(event: str) -> str:
    return EVENT_DISPLAY_NAMES.get(event, event.replace("_", " ").title())


def format_local_option(option) -> str:
    label = OPTION_DISPLAY_NAMES.get(option.kind, option.kind.value.title())
    details: list[str] = []
    if option.entity_id:
        details.append(option.entity_id)
    if option.direction:
        details.append(option.direction.value)
    if option.role_token and option.role_token not in {"push", "idle", "approach_contact"}:
        details.append(option.role_token)
    suffix = ", ".join(details)
    return f"R{option.robot_id}:{label}" + (f"[{suffix}]" if suffix else "")


def format_joint_option(joint) -> str:
    return " + ".join(format_local_option(option) for option in sorted(joint.options, key=lambda item: item.robot_id))


def _compact_entity_code(entity_id: str | None) -> str:
    if not entity_id:
        return ""
    return {
        "heavy-gate": "H",
        "light-front": "F",
        "light-rear": "R",
    }.get(entity_id, entity_id.replace("light-extra-", "E").replace("blocker-", "B"))


def format_joint_option_compact(joint) -> str:
    """Compact semantic action string for paper tables and episode records."""
    active = [option for option in joint.options if option.kind != OptionKind.IDLE]
    options = active or list(joint.options)
    groups: dict[tuple, list[int]] = {}
    exemplars = {}
    for option in options:
        key = (option.kind, option.entity_id, option.direction, option.role_token)
        groups.setdefault(key, []).append(option.robot_id)
        exemplars[key] = option
    pieces: list[str] = []
    for key in sorted(groups, key=lambda item: (item[0].value, str(item[1]), str(item[2]))):
        option = exemplars[key]
        robots = ",".join(f"R{rid}" for rid in sorted(groups[key]))
        entity = _compact_entity_code(option.entity_id)
        if option.kind == OptionKind.REPOSITION:
            pieces.append(f"{robots} Reposition")
        elif option.kind == OptionKind.APPROACH:
            pieces.append(f"{robots} Approach({entity or 'target'})")
        elif option.kind == OptionKind.PUSH:
            arrow = {
                "north": "↑", "east": "→", "south": "↓", "west": "←"
            }.get(option.direction.value if option.direction else "", "")
            pieces.append(f"{robots} Push({entity}){arrow}")
        elif option.kind == OptionKind.HOLD:
            pieces.append(f"{robots} Hold")
        elif option.kind == OptionKind.OBSERVE:
            pieces.append(f"{robots} Observe({entity})")
        else:
            pieces.append(f"{robots} Idle")
    return "; ".join(pieces)
