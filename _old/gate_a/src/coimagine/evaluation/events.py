from __future__ import annotations

from dataclasses import dataclass

from coimagine.belief.symbolic import RuleBasedBeliefUpdater
from coimagine.core.types import PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.state import GridState
from coimagine.options.generator import GridOptionGenerator
from coimagine.topology.admission import TopologyAdmission


@dataclass(frozen=True)
class ExactTopologySnapshot:
    node_templates: frozenset[tuple]
    edge_templates: frozenset[tuple]


@dataclass(frozen=True)
class TopologyDelta:
    before_nodes: frozenset[tuple]
    after_nodes: frozenset[tuple]
    before_edges: frozenset[tuple]
    after_edges: frozenset[tuple]
    created_nodes: frozenset[tuple]
    destroyed_nodes: frozenset[tuple]
    created_edges: frozenset[tuple]
    destroyed_edges: frozenset[tuple]
    persistent_edges: frozenset[tuple]
    dominant_event: str
    labels: tuple[str, ...]
    revealed_entities: frozenset[str]
    higher_order_context: bool

    @property
    def created_edge_count(self) -> int:
        return len(self.created_edges)

    @property
    def destroyed_edge_count(self) -> int:
        return len(self.destroyed_edges)

    @property
    def persistent_edge_count(self) -> int:
        return len(self.persistent_edges)


def exact_topology_snapshot(
    state: GridState,
    env: GridEnvironment,
    generator: GridOptionGenerator,
) -> ExactTopologySnapshot:
    # A terminal task has no future decision domain by definition.
    if env.is_terminal(state):
        return ExactTopologySnapshot(frozenset(), frozenset())
    updater = RuleBasedBeliefUpdater()
    beliefs = {
        rid: updater.initialize(env.observe(state, rid), state.width, state.height)
        for rid in state.robots
    }
    candidates = {rid: generator.generate(belief) for rid, belief in beliefs.items()}
    admission = TopologyAdmission(env, max_arity=max(3, len(state.robots)))
    result = admission.admit(state, candidates, PlannerMode.FULL_TUPLE_ORACLE)
    nodes = frozenset(
        option.template_key for options in candidates.values() for option in options
    )
    edges = frozenset(joint.template_key for joint in result.joint_options)
    return ExactTopologySnapshot(nodes, edges)


def exact_template_edges(
    state: GridState,
    env: GridEnvironment,
    generator: GridOptionGenerator,
) -> frozenset[tuple]:
    return exact_topology_snapshot(state, env, generator).edge_templates


def classify_transition(
    before: GridState,
    after: GridState,
    env: GridEnvironment,
    generator: GridOptionGenerator,
) -> TopologyDelta:
    root = exact_topology_snapshot(before, env, generator)
    successor = exact_topology_snapshot(after, env, generator)
    created_edges = successor.edge_templates - root.edge_templates
    destroyed_edges = root.edge_templates - successor.edge_templates
    persistent_edges = root.edge_templates & successor.edge_templates
    created_nodes = successor.node_templates - root.node_templates
    destroyed_nodes = root.node_templates - successor.node_templates

    if created_edges and destroyed_edges:
        dominant = "mixed_edge_turnover"
    elif created_edges:
        dominant = "pure_edge_creation"
    elif destroyed_edges:
        dominant = "pure_edge_destruction"
    else:
        dominant = "no_change"

    revealed = frozenset(after.revealed_entities - before.revealed_entities)
    labels: list[str] = [dominant]
    if revealed:
        labels.append("information_reveal")
    higher_order = any(
        blocker.required_pushers >= 3 for blocker in before.blockers.values()
    ) or any(capacity >= 2 for capacity in before.task.resource_capacities.values())
    if higher_order:
        labels.append("higher_order_context")

    return TopologyDelta(
        before_nodes=root.node_templates,
        after_nodes=successor.node_templates,
        before_edges=root.edge_templates,
        after_edges=successor.edge_templates,
        created_nodes=created_nodes,
        destroyed_nodes=destroyed_nodes,
        created_edges=created_edges,
        destroyed_edges=destroyed_edges,
        persistent_edges=persistent_edges,
        dominant_event=dominant,
        labels=tuple(labels),
        revealed_entities=revealed,
        higher_order_context=higher_order,
    )
