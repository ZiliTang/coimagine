from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Iterable

from coimagine.core.types import Direction, TaskKind
from coimagine.envs.grid.state import BlockerState, GridState, RobotState, TaskSpec
from coimagine.envs.grid.scenarios import two_robot_approach_then_push


@dataclass(frozen=True)
class ScenarioInstance:
    instance_id: str
    suite: str
    task_name: str
    seed: int
    state: GridState
    tags: tuple[str, ...] = ()


def _border_walls(width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        {(x, 0) for x in range(width)}
        | {(x, height - 1) for x in range(width)}
        | {(0, y) for y in range(height)}
        | {(width - 1, y) for y in range(height)}
    )


def generate_multi_blocker(seed: int, blockers: int = 3, robots: int = 2) -> ScenarioInstance:
    rng = Random(seed)
    width = max(12, 5 + 3 * blockers)
    height = 11
    corridor_y = 5
    corridor = frozenset((x, corridor_y) for x in range(1, width - 1))
    walls = set(_border_walls(width, height))

    blocker_states = {}
    slots = [2 + 3 * index for index in range(blockers)]
    for index, x in enumerate(slots):
        length = 1 if rng.random() < 0.55 else 2
        cells = frozenset((x + offset, corridor_y) for offset in range(length))
        required = 2 if robots >= 2 and rng.random() < 0.45 else 1
        blocker_states[f"blocker-{index}"] = BlockerState(
            entity_id=f"blocker-{index}", cells=cells, required_pushers=required
        )

    robot_states = {}
    free_x = [x for x in range(2, width - 2) if (x, 2) not in walls]
    rng.shuffle(free_x)
    for rid in range(robots):
        position = (free_x[rid], 2 + rid % 2) if rid < len(free_x) else (2 + rid, 2)
        robot_states[rid] = RobotState(rid, position)

    if blocker_states and rng.random() < 0.35:
        first = blocker_states["blocker-0"]
        contact = min(first.cells)
        robot_states[0] = RobotState(0, (contact[0], contact[1] - 1))

    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots=robot_states,
        blockers=blocker_states,
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=sum(
                len(blocker.cells & corridor) for blocker in blocker_states.values()
            ),
            metadata={"candidate_budget": 16} if blockers >= 4 else {},
        ),
    )
    return ScenarioInstance(
        instance_id=f"natural-multiblocker-{blockers}-{robots}-{seed}",
        suite="natural",
        task_name="multi_blocker_clearing",
        seed=seed,
        state=state,
        tags=(f"M={blockers}", f"N={robots}"),
    )


def generate_congested_transport(seed: int, robots: int = 3, required: int | None = None) -> ScenarioInstance:
    rng = Random(seed)
    width, height = 13, 13
    walls = set(_border_walls(width, height))
    for y in range(4, 10):
        walls.add((4, y))
        walls.add((8, y))
    object_cells = frozenset({(5, 5), (6, 5), (7, 5)})
    goal = frozenset({(5, 10), (6, 10), (7, 10)})
    required_pushers = required or min(robots, 2 + int(rng.random() < 0.35))
    robot_states = {
        rid: RobotState(rid, (2 + 2 * rid, 2 + (rid % 2))) for rid in range(robots)
    }
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots=robot_states,
        blockers={
            "payload": BlockerState(
                entity_id="payload", cells=object_cells, required_pushers=required_pushers
            )
        },
        task=TaskSpec(
            kind=TaskKind.TRANSPORT,
            goal_region=goal,
            transport_entity_id="payload",
            resource_capacities={"passage:central": 2},
            resource_regions={"passage:central": frozenset((x, y) for x in range(5, 8) for y in range(4, 10))},
            metadata={
                "initial_distance": 5.0,
                **({"candidate_budget": 8, "horizon": 1} if robots >= 3 else {}),
            },
        ),
    )
    return ScenarioInstance(
        instance_id=f"natural-transport-{robots}-{required_pushers}-{seed}",
        suite="natural",
        task_name="congested_transport",
        seed=seed,
        state=state,
        tags=(f"N={robots}", f"P={required_pushers}"),
    )


def generate_stable_control(seed: int) -> ScenarioInstance:
    rng = Random(seed)
    width, height = 10, 9
    corridor = frozenset((x, 5) for x in range(2, 8)) | frozenset(
        (x, 6) for x in range(2, 8)
    )
    blocker_cells = frozenset({(4, 5)})
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={
            0: RobotState(0, (4, 4)),
            1: RobotState(1, (2 + rng.randint(0, 1), 2)),
        },
        blockers={"blocker-0": BlockerState("blocker-0", blocker_cells, required_pushers=1)},
        task=TaskSpec(
            kind=TaskKind.STABLE_CONTROL,
            corridor=corridor,
            initial_blocked_cells=2,
        ),
    )
    return ScenarioInstance(
        instance_id=f"natural-stable-{seed}",
        suite="natural",
        task_name="topology_stable_control",
        seed=seed,
        state=state,
        tags=("negative_control",),
    )


def generate_hold_pass(seed: int) -> ScenarioInstance:
    width, height = 11, 9
    passage = frozenset({(5, 3), (5, 4), (5, 5)})
    # The hold station is intentionally offset from the payload's straight-line
    # route so the support robot does not occlude the task-goal observation.
    hold_cells = frozenset({(3, 5)})
    goal = frozenset({(7, 4)})
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={0: RobotState(0, (2, 4)), 1: RobotState(1, (3, 2))},
        blockers={},
        task=TaskSpec(
            kind=TaskKind.HOLD_PASS,
            payload_robot_id=0,
            goal_region=goal,
            hold_cells=hold_cells,
            guarded_passage=passage,
            metadata={"initial_distance": 5.0, "diagnostic_target": "temporary_hold_pass"},
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-hold-pass-{seed}",
        suite="diagnostic",
        task_name="hold_pass",
        seed=seed,
        state=state,
        tags=("temporary_coordination", "requires_two_step_continuation"),
    )


def generate_reveal_act(seed: int) -> ScenarioInstance:
    """Reveal a high-value hidden blocker versus pushing a visible low-value decoy."""

    width, height = 12, 10
    corridor = frozenset((x, 6) for x in range(2, 10))
    visible = BlockerState("visible-decoy", frozenset({(3, 6)}), required_pushers=1)
    hidden = BlockerState("hidden-blocker", frozenset({(6, 6), (7, 6)}), required_pushers=1)
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={
            0: RobotState(0, (3, 5)),  # can either observe or push the visible decoy
            1: RobotState(1, (6, 5)),  # already supports the hidden high-value push
        },
        blockers={"visible-decoy": visible, "hidden-blocker": hidden},
        task=TaskSpec(
            kind=TaskKind.REVEAL_ACT,
            corridor=corridor,
            initial_blocked_cells=3,
            observation_cells=frozenset({(3, 5)}),
            hidden_entities=frozenset({"hidden-blocker"}),
            metadata={
                "observe_duration": 2.0,
                "diagnostic_target": "reveal_then_push_with_competing_decoy",
            },
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-reveal-{seed}",
        suite="diagnostic",
        task_name="reveal_act",
        seed=seed,
        state=state,
        tags=("information_reveal", "first_action_competition", "two_step_reveal_act"),
    )


def generate_edge_destruction(seed: int) -> ScenarioInstance:
    width, height = 9, 8
    corridor = frozenset({(4, 4)})
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={0: RobotState(0, (4, 3)), 1: RobotState(1, (2, 2))},
        blockers={"consumed": BlockerState("consumed", frozenset({(4, 4)}), 1)},
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=1,
            metadata={"horizon": 1, "diagnostic_target": "pure_edge_destruction"},
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-destruction-{seed}",
        suite="diagnostic",
        task_name="edge_destruction",
        seed=seed,
        state=state,
        tags=("pure_edge_destruction", "terminal_consumption"),
    )


def generate_edge_replacement(seed: int) -> ScenarioInstance:
    """Push blocker A to create the support pose for blocker B while disabling A."""

    width, height = 10, 9
    walls = set(_border_walls(width, height))
    walls.add((4, 4))  # prevents a second eastward push of blocker A
    corridor = frozenset({(2, 4), (2, 5)})
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots={0: RobotState(0, (1, 4)), 1: RobotState(1, (7, 2))},
        blockers={
            "blocker-A": BlockerState("blocker-A", frozenset({(2, 4)}), 1),
            "blocker-B": BlockerState("blocker-B", frozenset({(2, 5)}), 1),
        },
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=2,
            metadata={"diagnostic_target": "entity_edge_replacement"},
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-replacement-{seed}",
        suite="diagnostic",
        task_name="edge_replacement",
        seed=seed,
        state=state,
        tags=("mixed_edge_turnover", "entity_replacement", "two_step_push_sequence"),
    )


def generate_higher_order_conflict(seed: int) -> ScenarioInstance:
    """Every singleton/pair push fits a capacity-two lane; the triple does not."""

    width, height = 12, 10
    contacts = frozenset({(4, 5), (5, 5), (6, 5)})
    support_lane = frozenset({(4, 4), (5, 4), (6, 4)})
    corridor = frozenset((x, 5) for x in range(2, 10))
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={
            0: RobotState(0, (4, 4)),
            1: RobotState(1, (5, 4)),
            2: RobotState(2, (6, 4)),
        },
        blockers={"payload": BlockerState("payload", contacts, required_pushers=1)},
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=3,
            resource_capacities={"shared:push-lane": 2},
            resource_regions={"shared:push-lane": support_lane},
            metadata={
                "candidate_budget": 8,
                "horizon": 1,
                "diagnostic_target": "pairwise_feasible_triple_infeasible",
            },
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-conflict-{seed}",
        suite="diagnostic",
        task_name="higher_order_conflict",
        seed=seed,
        state=state,
        tags=("higher_order", "pairwise_false_positive", "resource_capacity_2"),
    )


def generate_positive_higher_order_synergy(seed: int) -> ScenarioInstance:
    width, height = 12, 10
    cells = frozenset({(4, 5), (5, 5), (6, 5)})
    corridor = frozenset((x, 5) for x in range(2, 10))
    state = GridState(
        width=width,
        height=height,
        walls=_border_walls(width, height),
        robots={
            0: RobotState(0, (4, 4)),
            1: RobotState(1, (5, 4)),
            2: RobotState(2, (6, 4)),
        },
        blockers={"heavy": BlockerState("heavy", cells, required_pushers=3)},
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=3,
            metadata={
                "candidate_budget": 8,
                "horizon": 1,
                "diagnostic_target": "positive_three_robot_synergy",
            },
        ),
    )
    return ScenarioInstance(
        instance_id=f"diagnostic-synergy-{seed}",
        suite="diagnostic",
        task_name="positive_higher_order_synergy",
        seed=seed,
        state=state,
        tags=("higher_order", "positive_synergy", "three_distinct_contacts"),
    )



def generate_channel_clearing_showcase(seed: int = 0) -> ScenarioInstance:
    """Canonical three-robot, three-obstacle Gate-A story.

    All robots use the universal option library. The state only enables concrete
    instances. A two-cell heavy blocker needs two robots, an immediately reachable
    light blocker can be cleared by one robot, and a rear light blocker becomes
    contact-accessible only after the heavy blocker leaves the corridor. The setup
    is intentionally small enough for exact H=2 search but rich enough to show
    parallel assignment, coalition creation, and delayed option activation.
    """

    width, height = 18, 11
    corridor_y = 6
    corridor = frozenset((x, corridor_y) for x in range(2, 16))
    walls = _border_walls(width, height)

    heavy = BlockerState(
        entity_id="heavy-gate",
        cells=frozenset({(6, corridor_y), (7, corridor_y)}),
        required_pushers=2,
        mass_class="heavy",
        task_weight=2.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Heavy gate",
    )
    light = BlockerState(
        entity_id="light-front",
        cells=frozenset({(11, corridor_y)}),
        required_pushers=1,
        mass_class="light",
        task_weight=1.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Light front",
    )
    rear = BlockerState(
        entity_id="light-rear",
        cells=frozenset({(14, corridor_y)}),
        required_pushers=1,
        mass_class="light",
        task_weight=1.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Rear blocker",
    )
    blockers = {
        heavy.entity_id: heavy,
        light.entity_id: light,
        rear.entity_id: rear,
    }
    initial_weight = sum(
        blocker.task_weight * len(blocker.cells & corridor)
        for blocker in blockers.values()
    )
    state = GridState(
        width=width,
        height=height,
        walls=walls,
        robots={
            0: RobotState(0, (6, 2)),
            1: RobotState(1, (7, 2)),
            2: RobotState(2, (11, 5)),
        },
        blockers=blockers,
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=sum(len(blocker.cells & corridor) for blocker in blockers.values()),
            initial_blocked_weight=initial_weight,
            activation_dependencies={"light-rear": "heavy-gate"},
            metadata={
                "candidate_budget": 8,
                "horizon": 2,
                "episode_max_steps": 6,
                "showcase": True,
                "task_label": "Clear the transit corridor",
                "story": "Create the two-robot heavy-push edge while clearing the front light blocker in parallel.",
                "seed": seed,
            },
        ),
    )
    return ScenarioInstance(
        instance_id=f"showcase-channel-clearing-{seed}",
        suite="showcase",
        task_name="complex_channel_clearing",
        seed=seed,
        state=state,
        tags=("main_paper", "three_robots", "three_blockers", "mixed_mass"),
    )


def generate_channel_clearing_variant(seed: int, blockers: int = 3) -> ScenarioInstance:
    """Unscreened natural variants of the integrated channel-clearing task.

    Every variant preserves the same scientific decision competition: one light
    blocker is immediately pushable, while two robots must prepare a heavy-gate
    coalition whose completion activates a rear contact family. Geometry, robot
    starts, light-object weights, and extra blocker locations vary by seed. No
    instance is retained or rejected according to the realized method gain.
    """

    if blockers < 3:
        raise ValueError("channel clearing requires at least three blockers")
    if blockers > 7:
        raise ValueError("the compact exact benchmark supports at most seven blockers")

    rng = Random(seed)
    base = generate_channel_clearing_showcase(seed)
    corridor = base.state.task.corridor
    corridor_y = 6

    front_x = rng.choice((10, 11, 12))
    rear_candidates = tuple(x for x in (13, 14, 15) if x > front_x)
    rear_x = rng.choice(rear_candidates)

    heavy = BlockerState(
        entity_id="heavy-gate",
        cells=frozenset({(6, corridor_y), (7, corridor_y)}),
        required_pushers=2,
        mass_class="heavy",
        task_weight=1.8 + 0.4 * rng.random(),
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Heavy gate",
    )
    front = BlockerState(
        entity_id="light-front",
        cells=frozenset({(front_x, corridor_y)}),
        required_pushers=1,
        mass_class="light",
        task_weight=0.8 + 0.4 * rng.random(),
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Front light",
    )
    rear = BlockerState(
        entity_id="light-rear",
        cells=frozenset({(rear_x, corridor_y)}),
        required_pushers=1,
        mass_class="light",
        task_weight=0.8 + 0.4 * rng.random(),
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="Rear light",
    )
    blocker_map: dict[str, BlockerState] = {
        heavy.entity_id: heavy,
        front.entity_id: front,
        rear.entity_id: rear,
    }

    available_extra_x = [
        x for x in range(9, 16)
        if x not in {front_x, rear_x}
    ]
    rng.shuffle(available_extra_x)
    for index in range(blockers - 3):
        x = available_extra_x[index]
        entity_id = f"light-extra-{index + 1}"
        blocker_map[entity_id] = BlockerState(
            entity_id=entity_id,
            cells=frozenset({(x, corridor_y)}),
            required_pushers=1,
            mass_class="light",
            task_weight=0.7 + 0.5 * rng.random(),
            allowed_push_directions=frozenset({Direction.NORTH}),
            display_name=f"Extra light {index + 1}",
        )

    # Seed-controlled start perturbations preserve the heavy contact lanes and
    # keep the front light blocker immediately executable by R2.
    r0 = (5 + rng.randrange(2), 2 + rng.randrange(2))
    r1 = (7 + rng.randrange(2), 2 + rng.randrange(2))
    if r1 == r0:
        r1 = (8, 2)
    robots = {
        0: RobotState(0, r0),
        1: RobotState(1, r1),
        2: RobotState(2, (front_x, corridor_y - 1)),
    }

    initial_weight = sum(
        blocker.task_weight * len(blocker.cells & corridor)
        for blocker in blocker_map.values()
    )
    task = TaskSpec(
        kind=TaskKind.CLEARING,
        corridor=corridor,
        initial_blocked_cells=sum(len(blocker.cells & corridor) for blocker in blocker_map.values()),
        initial_blocked_weight=initial_weight,
        activation_dependencies={"light-rear": "heavy-gate"},
        metadata={
            "candidate_budget": 8,
            "horizon": 2,
            "episode_max_steps": max(8, 4 + 2 * (blockers - 3)),
            "benchmark_family": "channel_clearing",
            "unscreened": True,
        },
    )
    variant = GridState(
        width=base.state.width,
        height=base.state.height,
        walls=base.state.walls,
        robots=robots,
        blockers=blocker_map,
        task=task,
    )
    return ScenarioInstance(
        instance_id=f"natural-channel-{blockers}-{seed}",
        suite="natural",
        task_name="complex_channel_clearing",
        seed=seed,
        state=variant,
        tags=(f"M={blockers}", "N=3", "mixed_mass", "unscreened"),
    )


def generate_local_imagination_showcase(
    seed: int = 0,
    *,
    surprise_obstacle: bool = False,
) -> ScenarioInstance:
    """Canonical COST-versus-IET local-observation benchmark.

    Both robots initially observe only the left contact of a two-cell heavy
    blocker. R0 can approach that contact, while R1 can execute a currently
    grounded Reposition option. From the new viewpoint, IET predicts a second
    contact and therefore values the three-option chain

        Reposition/Approach -> Approach -> coordinated Push.

    COST propagates the same known states but forbids future option templates
    outside the current observation-supported topology. The optional surprise
    wall is outside both root observations and invalidates the optimistic second
    contact after the real first option boundary; it is used to quantify optimism
    cost rather than to define the main mechanism.
    """

    del seed  # Geometry is canonical; the seed remains in the instance identity.
    width, height = 13, 11
    walls = set(_border_walls(width, height))
    if surprise_obstacle:
        # This support cell is outside both root sensing disks but becomes visible
        # from the reposition anchor. It makes the optimistic prior falsifiable.
        walls.add((8, 6))

    heavy_cells = frozenset({(7, 7), (8, 7)})
    heavy = BlockerState(
        entity_id="heavy-view",
        cells=heavy_cells,
        required_pushers=2,
        mass_class="heavy",
        task_weight=1.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="View-dependent heavy blocker",
    )
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots={
            0: RobotState(0, (2, 7)),
            1: RobotState(1, (2, 4)),
        },
        blockers={heavy.entity_id: heavy},
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=heavy_cells,
            initial_blocked_cells=len(heavy_cells),
            initial_blocked_weight=float(len(heavy_cells)),
            metadata={
                "candidate_budget": 8,
                "episode_max_steps": 5,
                "enable_reposition": True,
                "reposition_anchors": ((8, 4),),
                "max_reposition_candidates": 1,
                "observation_radius": 6,
                "showcase": True,
                "surprise_obstacle": surprise_obstacle,
                "task_label": "Clear the heavy blocker from local observations",
            },
        ),
    )
    suffix = "surprise" if surprise_obstacle else "open"
    return ScenarioInstance(
        instance_id=f"local-imagination-{suffix}-0",
        suite="showcase" if not surprise_obstacle else "robustness",
        task_name="local_imagination_chain",
        seed=0,
        state=state,
        tags=(
            "cost_vs_iet",
            "reposition_approach_push",
            "H3_required",
            suffix,
        ),
    )


def generate_local_imagination_channel_showcase(seed: int = 0) -> ScenarioInstance:
    """Multi-blocker channel-clearing storyboard used by the Gate-A main figure.

    The first light blocker is directly pushable by both COST and IET.  After this
    shared progress, the heavy two-cell gate exposes only one observation-grounded
    contact.  A currently visible reposition anchor creates the second contact in
    the successor local observation, enabling the IET-only three-boundary chain

        Reposition -> joint Approach -> coordinated Push.

    Two additional light blockers keep the scene non-trivial and give COST a
    sensible current-support task instead of forcing an immediate idle decision.
    """

    del seed
    width, height = 19, 13
    channel_y = 7
    walls = set(_border_walls(width, height))
    # The horizontal channel is open at both lateral boundaries.  The plotting
    # code also marks its upper/lower limits with dashed lines and clearing arrows.
    for y in (channel_y - 1, channel_y, channel_y + 1):
        walls.discard((0, y))
        walls.discard((width - 1, y))

    blockers = {
        "B0": BlockerState(
            "B0",
            frozenset({(4, channel_y)}),
            required_pushers=1,
            mass_class="light",
            task_weight=1.2,
            allowed_push_directions=frozenset({Direction.NORTH}),
            display_name="Shared entry blocker",
        ),
        "B1": BlockerState(
            "B1",
            frozenset({(7, channel_y)}),
            required_pushers=1,
            mass_class="light",
            task_weight=0.8,
            allowed_push_directions=frozenset({Direction.NORTH}),
            display_name="Current-support blocker",
        ),
        "H": BlockerState(
            "H",
            frozenset({(11, channel_y), (12, channel_y)}),
            required_pushers=2,
            mass_class="heavy",
            task_weight=1.5,
            allowed_push_directions=frozenset({Direction.NORTH}),
            display_name="View-dependent heavy gate",
        ),
        "B3": BlockerState(
            "B3",
            frozenset({(16, channel_y)}),
            required_pushers=1,
            mass_class="light",
            task_weight=0.6,
            allowed_push_directions=frozenset({Direction.NORTH}),
            display_name="Far blocker",
        ),
    }
    corridor = frozenset((x, channel_y) for x in range(1, width - 1))
    initial_weight = sum(
        blocker.task_weight * len(blocker.cells & corridor)
        for blocker in blockers.values()
    )
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots={
            0: RobotState(0, (4, channel_y - 1)),
            1: RobotState(1, (3, channel_y - 1)),
        },
        blockers=blockers,
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=sum(
                len(blocker.cells & corridor) for blocker in blockers.values()
            ),
            initial_blocked_weight=initial_weight,
            metadata={
                "candidate_budget": 10,
                "episode_max_steps": 8,
                "enable_reposition": True,
                "reposition_anchors": ((9, 4), (12, 4), (15, 5)),
                "max_reposition_candidates": 3,
                "observation_radius": 10,
                "showcase": True,
                "channel_y": channel_y,
                "channel_boundary_rows": (channel_y, channel_y + 1),
                "task_label": "Clear the horizontal passage",
            },
        ),
    )
    return ScenarioInstance(
        instance_id="local-imagination-channel-showcase-0",
        suite="showcase",
        task_name="multi_blocker_channel_imagination",
        seed=0,
        state=state,
        tags=(
            "cost_vs_iet",
            "channel_clearing",
            "multi_blocker",
            "factor_topology",
            "H3_required",
        ),
    )


def generate_local_imagination_variant(seed: int) -> ScenarioInstance:
    """Unscreened geometric variants for horizon and optimism studies.

    Geometry is sampled before planning and is never filtered by method outcome.
    Three out of every four seeds use an open continuation; every fourth seed
    contains an unseen wall at the imagined second contact. Translation,
    reflection, map size, and setup distances vary while preserving the same
    causal requirement: Reposition/Approach -> Approach -> coordinated Push.
    """

    rng = Random(seed)
    surprise = seed % 4 == 3
    width = rng.choice((13, 14, 15, 16))
    height = rng.choice((10, 11, 12))
    mirror = bool(rng.randrange(2))
    # These gaps are fixed by the benchmark information pattern: both robots
    # observe one grounded contact, the second contact is occluded, and the
    # reposition anchor lies on the current sensing boundary. Translation and
    # reflection vary without outcome-based screening.
    horizontal_gap = 5
    vertical_gap = 3
    hy = rng.randint(max(5, vertical_gap + 2), height - 3)

    if not mirror:
        # Robots start to the left. R0 observes the left contact; R1 can only
        # reposition to a currently visible anchor near the right contact.
        hx_min = horizontal_gap + 1
        hx_max = width - 3
        hx = rng.randint(hx_min, hx_max)
        heavy_cells = frozenset({(hx, hy), (hx + 1, hy)})
        robots = {
            0: RobotState(0, (hx - horizontal_gap, hy)),
            1: RobotState(1, (hx - horizontal_gap, hy - vertical_gap)),
        }
        anchor = (hx + 1, hy - vertical_gap)
        surprise_support = (hx + 1, hy - 1)
    else:
        # Horizontal reflection: R0 observes the right contact and R1
        # repositions toward the left contact.
        hx_min = 2
        hx_max = width - horizontal_gap - 3
        hx = rng.randint(hx_min, hx_max)
        heavy_cells = frozenset({(hx, hy), (hx + 1, hy)})
        robots = {
            0: RobotState(0, (hx + 1 + horizontal_gap, hy)),
            1: RobotState(1, (hx + 1 + horizontal_gap, hy - vertical_gap)),
        }
        anchor = (hx, hy - vertical_gap)
        surprise_support = (hx, hy - 1)

    walls = set(_border_walls(width, height))
    if surprise:
        walls.add(surprise_support)
    blocker = BlockerState(
        entity_id="heavy-view",
        cells=heavy_cells,
        required_pushers=2,
        mass_class="heavy",
        task_weight=1.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="View-dependent heavy blocker",
    )
    task = TaskSpec(
        kind=TaskKind.CLEARING,
        corridor=heavy_cells,
        initial_blocked_cells=len(heavy_cells),
        initial_blocked_weight=float(len(heavy_cells)),
        metadata={
            "candidate_budget": 8,
            "episode_max_steps": 5,
            "enable_reposition": True,
            "reposition_anchors": (anchor,),
            "max_reposition_candidates": 1,
            "observation_radius": 6,
            "showcase": False,
            "surprise_obstacle": surprise,
            "unscreened": True,
            "variant_seed": seed,
            "mirror": mirror,
            "horizontal_gap": horizontal_gap,
            "vertical_gap": vertical_gap,
        },
    )
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots=robots,
        blockers={blocker.entity_id: blocker},
        task=task,
    )
    suffix = "surprise" if surprise else "open"
    return ScenarioInstance(
        instance_id=f"local-imagination-{suffix}-{seed}",
        suite="natural",
        task_name="local_imagination_chain",
        seed=seed,
        state=state,
        tags=("cost_vs_iet", "H3_required", suffix, "unscreened"),
    )



LOCAL_IMAGINATION_FAMILIES = (
    "direct_push",
    "approach_push",
    "mixed_approach_push",
    "reposition_approach_push",
)


def generate_local_imagination_benchmark_case(
    seed: int,
    *,
    family: str,
    scene_type: str = "open",
) -> ScenarioInstance:
    """Generate one unscreened, family-stratified local-imagination case.

    ``scene_type='open'`` is valid for every causal family.  ``'surprise'`` is
    reserved for the Reposition--Approach--Push family and inserts an unseen
    support-wall that falsifies the optimistic successor observation.  Geometry
    is sampled from the seed before planning and is never filtered by method
    outcome.
    """

    if family not in LOCAL_IMAGINATION_FAMILIES:
        raise ValueError(f"Unknown local-imagination family: {family}")
    if scene_type not in {"open", "surprise"}:
        raise ValueError(f"Unsupported scene_type: {scene_type}")
    if scene_type == "surprise" and family != "reposition_approach_push":
        raise ValueError("Surprise cases are defined only for the three-step family")

    # Mix the family into the RNG stream so the same integer seed does not create
    # correlated geometry across all four strata.
    family_index = LOCAL_IMAGINATION_FAMILIES.index(family)
    rng = Random(10_000_019 * (seed + 1) + 97_409 * family_index)
    width = rng.choice((14, 15, 16, 17, 18, 19))
    height = rng.choice((11, 12, 13, 14))
    mirror = bool(rng.randrange(2))
    hy = rng.randint(6, height - 3)
    anchors: tuple[tuple[int, int], ...] = ()
    surprise = scene_type == "surprise"
    surprise_support: tuple[int, int] | None = None

    if family == "reposition_approach_push":
        # Keep the information pattern invariant while varying translation,
        # reflection, map size, and the two robot start rows.  A five-cell
        # horizontal offset keeps one contact grounded and the second outside the
        # root support; the vertical offset varies without changing the causal
        # depth.
        horizontal_gap = 5
        vertical_gap = rng.choice((2, 3))
        if not mirror:
            hx = rng.randint(horizontal_gap + 1, width - 3)
            heavy_cells = frozenset({(hx, hy), (hx + 1, hy)})
            r0 = (hx - horizontal_gap, hy)
            r1 = (hx - horizontal_gap, hy - vertical_gap)
            positions = (r0, r1)
            anchors = ((hx + 1, hy - vertical_gap),)
            surprise_support = (hx + 1, hy - 1)
        else:
            hx = rng.randint(2, width - horizontal_gap - 3)
            heavy_cells = frozenset({(hx, hy), (hx + 1, hy)})
            r0 = (hx + 1 + horizontal_gap, hy)
            r1 = (hx + 1 + horizontal_gap, hy - vertical_gap)
            positions = (r0, r1)
            anchors = ((hx, hy - vertical_gap),)
            surprise_support = (hx, hy - 1)
        dependency_depth = 3
        difficulty_variant = "future_support_required"
    else:
        hx = rng.randint(4, width - 7)
        heavy_cells = frozenset({(hx, hy), (hx + 1, hy)})
        left_support = (hx, hy - 1)
        right_support = (hx + 1, hy - 1)
        if family == "direct_push":
            positions = (left_support, right_support)
            dependency_depth = 1
            difficulty_variant = "both_contacts_ready"
        elif family == "approach_push":
            left_dx = rng.choice((1, 2, 3, 4))
            right_dx = rng.choice((1, 2, 3, 4))
            y_offset = rng.choice((1, 2, 3))
            positions = (
                (max(1, hx - left_dx), max(1, hy - y_offset)),
                (min(width - 2, hx + 1 + right_dx), max(1, hy - y_offset)),
            )
            dependency_depth = 2
            difficulty_variant = "both_contacts_grounded"
        else:
            far_dx = rng.choice((2, 3, 4, 5))
            y_offset = rng.choice((1, 2, 3))
            positions = (
                left_support,
                (min(width - 2, hx + 1 + far_dx), max(1, hy - y_offset)),
            )
            dependency_depth = 2
            difficulty_variant = "one_contact_ready"

        if mirror:
            def reflect(cell: tuple[int, int]) -> tuple[int, int]:
                return (width - 1 - cell[0], cell[1])
            heavy_cells = frozenset(reflect(cell) for cell in heavy_cells)
            positions = tuple(reflect(cell) for cell in positions)
        if rng.random() < 0.5:
            positions = (positions[1], positions[0])

    walls = set(_border_walls(width, height))
    if surprise and surprise_support is not None:
        walls.add(surprise_support)

    blocker = BlockerState(
        entity_id="heavy-view",
        cells=heavy_cells,
        required_pushers=2,
        mass_class="heavy",
        task_weight=1.0,
        allowed_push_directions=frozenset({Direction.NORTH}),
        display_name="View-dependent heavy blocker",
    )
    observation_radius = 6
    task = TaskSpec(
        kind=TaskKind.CLEARING,
        corridor=heavy_cells,
        initial_blocked_cells=len(heavy_cells),
        initial_blocked_weight=float(len(heavy_cells)),
        metadata={
            "candidate_budget": 10,
            "episode_max_steps": 6,
            "enable_reposition": bool(anchors),
            "reposition_anchors": anchors,
            "max_reposition_candidates": len(anchors),
            "observation_radius": observation_radius,
            "showcase": False,
            "surprise_obstacle": surprise,
            "scene_type": scene_type,
            "unscreened": True,
            "variant_seed": seed,
            "mirror": mirror,
            "scene_family": family,
            "dependency_depth": dependency_depth,
            "difficulty_variant": difficulty_variant,
            "randomized_initial_positions": True,
            "benchmark_version": "v0.8",
        },
    )
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots={0: RobotState(0, positions[0]), 1: RobotState(1, positions[1])},
        blockers={blocker.entity_id: blocker},
        task=task,
    )
    return ScenarioInstance(
        instance_id=f"local-imagination-v08-{family}-{scene_type}-{seed}",
        suite="natural" if scene_type == "open" else "robustness",
        task_name=f"local_imagination_{family}",
        seed=seed,
        state=state,
        tags=(
            "cst_vs_iet",
            family,
            f"H{dependency_depth}_dependency",
            scene_type,
            difficulty_variant,
            "randomized_initial_positions",
            "unscreened",
        ),
    )


def generate_local_imagination_benchmark_variant(seed: int) -> ScenarioInstance:
    """Backward-compatible v0.7 mixed-cycle wrapper.

    New quantitative studies should call
    :func:`generate_local_imagination_benchmark_case` explicitly so every open
    causal family receives an independently controlled sample count.
    """

    family = LOCAL_IMAGINATION_FAMILIES[seed % len(LOCAL_IMAGINATION_FAMILIES)]
    if family == "reposition_approach_push":
        hard_index = seed // len(LOCAL_IMAGINATION_FAMILIES)
        scene_type = "surprise" if hard_index % 2 == 1 else "open"
    else:
        scene_type = "open"
    return generate_local_imagination_benchmark_case(
        seed,
        family=family,
        scene_type=scene_type,
    )


RANDOMIZED_CHANNEL_DIFFICULTIES = ("easy", "moderate", "hard")


def _farthest_anchor_subset(
    candidates: Iterable[tuple[int, int]],
    *,
    target_count: int,
    forced: Iterable[tuple[int, int]] = (),
    rng: Random,
) -> list[tuple[int, int]]:
    """Select a spatially dispersed anchor subset with deterministic randomness.

    Forced anchors are retained first. Remaining anchors are selected by a
    farthest-point rule, with a small seeded jitter used only to break ties. This
    avoids the strongly clustered layouts produced by ``shuffle + truncate``.
    """

    pool = list(dict.fromkeys(candidates))
    selected = [cell for cell in dict.fromkeys(forced) if cell in pool]
    available = [cell for cell in pool if cell not in selected]
    if not selected and available:
        # Start near the spatial centre so the subsequent farthest-point choices
        # spread toward both lateral ends and both sides of the channel.
        centre_x = sum(cell[0] for cell in available) / len(available)
        centre_y = sum(cell[1] for cell in available) / len(available)
        first = min(
            available,
            key=lambda cell: (
                abs(cell[0] - centre_x) + abs(cell[1] - centre_y),
                rng.random(),
            ),
        )
        selected.append(first)
        available.remove(first)

    while available and len(selected) < target_count:
        choice = max(
            available,
            key=lambda cell: (
                min(
                    abs(cell[0] - other[0]) + abs(cell[1] - other[1])
                    for other in selected
                ),
                rng.random(),
            ),
        )
        selected.append(choice)
        available.remove(choice)
    return selected[:target_count]


def _stratified_channel_anchors(
    *,
    width: int,
    height: int,
    channel_y: int,
    occupied: set[tuple[int, int]],
    level: int,
    rng: Random,
    forced: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Create a two-sided, approximately uniform reposition-anchor bank."""

    lower_rows = [channel_y - 2, channel_y - 4]
    upper_rows = [channel_y + 3, channel_y + 5]
    rows = [y for y in lower_rows + upper_rows if 0 < y < height - 1]
    stride = (4, 3, 3)[level]
    phase = rng.randrange(stride)
    candidates = []
    for row_index, y in enumerate(rows):
        # Stagger alternate rows to avoid vertically aligned anchor columns.
        row_phase = (phase + (row_index % 2) * max(1, stride // 2)) % stride
        for x in range(2 + row_phase, width - 2, stride):
            cell = (x, y)
            if cell not in occupied:
                candidates.append(cell)

    forced_valid = [
        cell
        for cell in forced
        if 0 < cell[0] < width - 1
        and 0 < cell[1] < height - 1
        and cell not in occupied
    ]
    candidates.extend(forced_valid)
    target_count = (8, 10, 12)[level]
    return _farthest_anchor_subset(
        candidates,
        target_count=target_count,
        forced=forced_valid,
        rng=rng,
    )


def _critical_slot_options(options, critical_entity: str):
    """Return contact-preparation options for one critical entity by slot."""

    from coimagine.core.types import OptionKind

    return {
        option.slot_token: option
        for option in options
        if option.entity_id == critical_entity
        and option.slot_token is not None
        and option.kind in {OptionKind.APPROACH, OptionKind.PUSH}
    }


def _match_distinct_approaches(options_by_robot, required: int):
    """Find distinct-contact Approach options for distinct robots."""

    from coimagine.core.types import OptionKind

    robot_ids = tuple(sorted(options_by_robot))

    def search(index: int, selected: list, slots: set[str], targets: set[tuple[int, int]]):
        if len(selected) >= required:
            return tuple(selected)
        if len(robot_ids) - index < required - len(selected):
            return None
        if index >= len(robot_ids):
            return None
        robot_id = robot_ids[index]
        skipped = search(index + 1, selected, slots, targets)
        if skipped is not None:
            return skipped
        for option in options_by_robot[robot_id]:
            if option.kind != OptionKind.APPROACH or option.slot_token is None:
                continue
            if option.slot_token in slots or option.target is None or option.target in targets:
                continue
            selected.append(option)
            slots.add(option.slot_token)
            targets.add(option.target)
            matched = search(index + 1, selected, slots, targets)
            if matched is not None:
                return matched
            targets.remove(option.target)
            slots.remove(option.slot_token)
            selected.pop()
        return None

    return search(0, [], set(), set())


def _validate_randomized_channel_structure(
    instance: ScenarioInstance,
    *,
    validation_candidate_budget: int = 8,
) -> dict[str, object]:
    """Validate the intended causal geometry without planner-outcome screening.

    A strict future-support scene must realize the three-boundary chain
    Reposition -> two distinct Approaches -> coordinated Push.  The check uses
    only exact sensing, option grounding, and execution semantics; it never runs
    CST or IET and therefore cannot select scenes according to method success.
    """

    from coimagine.belief.symbolic import RuleBasedBeliefUpdater
    from coimagine.core.types import JointOption, OptionKind
    from coimagine.envs.grid.environment import GridEnvironment
    from coimagine.options.generator import GridOptionGenerator

    state = instance.state
    metadata = state.task.metadata
    future_required = bool(metadata.get("future_support_required", False))
    critical_entity = str(metadata.get("critical_entity", ""))
    observation_radius = int(metadata.get("observation_radius", 8))
    env = GridEnvironment(observation_radius=observation_radius)
    updater = RuleBasedBeliefUpdater()
    generator = GridOptionGenerator(candidate_budget=validation_candidate_budget)

    beliefs = {
        robot_id: updater.initialize(
            env.observe(state, robot_id), state.width, state.height
        )
        for robot_id in state.robots
    }
    root_support = {
        robot_id: generator.generate_support_pool(belief)
        for robot_id, belief in beliefs.items()
    }
    root_candidates = {
        robot_id: generator.select_candidates(beliefs[robot_id], root_support[robot_id])
        for robot_id in beliefs
    }
    nonidle_root = sum(
        option.kind != OptionKind.IDLE
        for options in root_candidates.values()
        for option in options
    )
    result: dict[str, object] = {
        "structural_validation_candidate_budget": validation_candidate_budget,
        "root_nonidle_candidate_count": nonidle_root,
        "future_support_structure_valid": False,
        "future_support_validation_reason": "not_required",
        "critical_root_slot_count": 0,
        "critical_successor_slot_count": 0,
        "critical_new_slot_count": 0,
        "critical_reposition_anchor": (),
    }
    if not future_required:
        result["root_supported_structure_valid"] = nonidle_root > 0
        return result
    result["root_supported_structure_valid"] = False
    critical = state.blockers.get(critical_entity)
    if critical is None:
        result["future_support_validation_reason"] = "missing_critical_entity"
        return result
    required = int(critical.required_pushers)
    if required < 2 or len(critical.cells) < required:
        result["future_support_validation_reason"] = "insufficient_critical_contact_slots"
        return result

    root_slots_by_robot = {
        robot_id: set(_critical_slot_options(options, critical_entity))
        for robot_id, options in root_support.items()
    }
    root_slot_union = set().union(*root_slots_by_robot.values())
    result["critical_root_slot_count"] = len(root_slot_union)
    if len(root_slot_union) >= required:
        result["future_support_validation_reason"] = "critical_coalition_already_root_supported"
        return result

    idle_by_robot = {
        robot_id: next(
            (option for option in options if option.kind == OptionKind.IDLE),
            None,
        )
        for robot_id, options in root_candidates.items()
    }
    for mover, options in root_candidates.items():
        for reposition in options:
            if reposition.kind != OptionKind.REPOSITION or reposition.target is None:
                continue
            joint_items = []
            for robot_id in sorted(state.robots):
                option = reposition if robot_id == mover else idle_by_robot.get(robot_id)
                if option is None:
                    joint_items = []
                    break
                joint_items.append(option)
            if len(joint_items) != len(state.robots):
                continue
            reposition_transition = env.execute(state, JointOption(tuple(joint_items)))
            if not reposition_transition.result.success:
                continue
            successor_state = reposition_transition.state
            successor_beliefs = {
                robot_id: updater.update(
                    beliefs[robot_id],
                    env.observe(successor_state, robot_id),
                    source="structural_successor_observation",
                )
                for robot_id in beliefs
            }
            successor_support = {
                robot_id: generator.generate_support_pool(belief)
                for robot_id, belief in successor_beliefs.items()
            }
            successor_candidates = {
                robot_id: generator.select_candidates(
                    successor_beliefs[robot_id], successor_support[robot_id]
                )
                for robot_id in successor_beliefs
            }
            successor_slots_by_robot = {
                robot_id: set(_critical_slot_options(options, critical_entity))
                for robot_id, options in successor_candidates.items()
            }
            successor_slot_union = set().union(*successor_slots_by_robot.values())
            new_slots = successor_slot_union - root_slot_union
            if len(successor_slot_union) < required or not new_slots:
                continue

            approach_pools = {
                robot_id: tuple(
                    option
                    for option in options
                    if option.entity_id == critical_entity
                    and option.kind == OptionKind.APPROACH
                )
                for robot_id, options in successor_candidates.items()
            }
            approaches = _match_distinct_approaches(approach_pools, required)
            if approaches is None:
                continue
            approach_transition = env.execute(
                successor_state,
                JointOption(tuple(sorted(approaches, key=lambda option: option.robot_id))),
            )
            if not approach_transition.result.success:
                continue

            contact_state = approach_transition.state
            contact_beliefs = {
                robot_id: updater.update(
                    successor_beliefs[robot_id],
                    env.observe(contact_state, robot_id),
                    source="structural_contact_observation",
                )
                for robot_id in successor_beliefs
            }
            contact_candidates = {
                robot_id: generator.generate(belief)
                for robot_id, belief in contact_beliefs.items()
            }
            push_items = []
            for approach in approaches:
                push = next(
                    (
                        option
                        for option in contact_candidates[approach.robot_id]
                        if option.kind == OptionKind.PUSH
                        and option.entity_id == critical_entity
                        and option.slot_token == approach.slot_token
                    ),
                    None,
                )
                if push is None:
                    push_items = []
                    break
                push_items.append(push)
            if len(push_items) != required:
                continue
            push_transition = env.execute(
                contact_state,
                JointOption(tuple(sorted(push_items, key=lambda option: option.robot_id))),
            )
            if not push_transition.result.success:
                continue

            result.update(
                {
                    "future_support_structure_valid": True,
                    "future_support_validation_reason": "strict_h3_chain",
                    "critical_successor_slot_count": len(successor_slot_union),
                    "critical_new_slot_count": len(new_slots),
                    "critical_reposition_anchor": reposition.target,
                    "critical_reposition_robot": mover,
                    "critical_approach_slots": tuple(
                        option.slot_token for option in approaches
                    ),
                }
            )
            return result

    result["future_support_validation_reason"] = "no_strict_h3_chain"
    return result


def _generate_randomized_channel_benchmark_case_once(
    seed: int,
    *,
    difficulty: str,
    planned_future_support: bool,
    generation_attempt: int = 0,
) -> ScenarioInstance:
    """Generate one outcome-unscreened, causally calibrated channel task.

    Difficulty controls task load and nuisance geometry, while the independent
    future-support flag controls whether a strict viewpoint-dependent contact
    chain is embedded.  No planner outcome is queried.
    """

    rng = Random(
        91_827_341 * (seed + 1)
        + 104_729 * (generation_attempt + 1)
        + 7_919
    )
    if difficulty not in RANDOMIZED_CHANNEL_DIFFICULTIES:
        raise ValueError(f"Unsupported randomized-channel difficulty: {difficulty}")
    level = RANDOMIZED_CHANNEL_DIFFICULTIES.index(difficulty)

    # Keep the hard tier inside a discriminative regime rather than conflating
    # larger maps, many blockers, coalition objects, occlusion, and activation.
    width = rng.randint((18, 20, 21)[level], (21, 23, 25)[level])
    height = rng.randint((12, 13, 14)[level], (15, 16, 17)[level])
    channel_y = rng.randint(5, height - 6)
    channel_rows = (channel_y, channel_y + 1)
    blocker_count = rng.choices(
        ((2, 3), (3, 4), (3, 4))[level],
        weights=((7, 3), (8, 2), (4, 6))[level],
        k=1,
    )[0]
    # Keep sensing range matched across difficulty strata.  Difficulty should
    # primarily come from task structure and geometry, not from silently giving
    # moderate/hard cases a narrower observation model.
    observation_radius = rng.randint(7, 9)
    initial_position_mode = rng.choices(
        ("same_side", "split_side", "cross_side_required"),
        weights=((5, 3, 2), (4, 4, 2), (3, 4, 3))[level],
        k=1,
    )[0]
    # Strict future-support scenes use a same-side lateral-viewpoint change.  This
    # isolates the intended option-support mechanism; split/cross-side layouts
    # remain well represented in root-supported scenes.
    if planned_future_support:
        initial_position_mode = "same_side"
    initial_side = rng.choice(("lower", "upper"))
    primary_direction = (
        Direction.NORTH if initial_side == "lower" else Direction.SOUTH
    )

    walls = set(_border_walls(width, height))
    for y in channel_rows:
        walls.discard((0, y))
        walls.discard((width - 1, y))

    # Sample separated longitudinal blocker slots.  A greedy randomized pass is
    # used instead of outcome screening, preserving a broad but valid geometry.
    available_x = list(range(3, width - 4))
    rng.shuffle(available_x)
    slots: list[int] = []
    for x in available_x:
        if all(abs(x - other) >= 3 for other in slots):
            slots.append(x)
        if len(slots) >= blocker_count:
            break
    slots.sort()
    if len(slots) < blocker_count:
        blocker_count = len(slots)

    critical_index = blocker_count - 1 if planned_future_support else None
    blockers: dict[str, BlockerState] = {}
    corridor: set[tuple[int, int]] = set()
    heavy_ids: list[str] = []
    for index, x in enumerate(slots):
        is_critical = index == critical_index
        length = (
            2
            if is_critical
            else (
                2
                if rng.random() < (0.25 + 0.08 * level) and x + 1 < width - 2
                else 1
            )
        )
        if length == 2 and x + 1 >= width - 2:
            length = 1
        cells = frozenset((x + dx, channel_y) for dx in range(length))
        heavy_probability = (0.10, 0.22, 0.35)[level]
        required = 2 if is_critical or rng.random() < heavy_probability else 1
        if index == 0 and blocker_count > 1 and not is_critical:
            required = 1
        both_direction_probability = (0.18, 0.25, 0.30)[level]
        if is_critical:
            allowed_directions = frozenset({primary_direction})
        elif rng.random() < both_direction_probability:
            allowed_directions = frozenset({Direction.NORTH, Direction.SOUTH})
        elif rng.random() < 0.72:
            allowed_directions = frozenset({primary_direction})
        else:
            allowed_directions = frozenset(
                {
                    Direction.SOUTH
                    if primary_direction == Direction.NORTH
                    else Direction.NORTH
                }
            )
        entity_id = f"C{index}"
        if required == 2:
            heavy_ids.append(entity_id)
        blocker = BlockerState(
            entity_id=entity_id,
            cells=cells,
            required_pushers=required,
            mass_class="heavy" if required == 2 else "light",
            task_weight=round(rng.uniform(0.75, 1.45), 3),
            allowed_push_directions=allowed_directions,
            display_name=f"Channel blocker {index}",
        )
        blockers[entity_id] = blocker
        corridor.update(cells)

    # Root-supported moderate/hard scenes still contain one coalition object, but
    # the number of heavy entities is capped so hard does not collapse into an
    # all-coalition stress test.
    if level >= 1 and blockers and not heavy_ids:
        target_id = sorted(blockers)[-1]
        old = blockers[target_id]
        blockers[target_id] = BlockerState(
            entity_id=old.entity_id,
            cells=old.cells,
            required_pushers=2,
            mass_class="heavy",
            task_weight=old.task_weight,
            allowed_push_directions=old.allowed_push_directions,
            display_name=old.display_name,
        )
        heavy_ids.append(target_id)

    max_heavy = (1, 1, 2)[level]
    critical_name = f"C{critical_index}" if critical_index is not None else None
    ordered_heavy = sorted(
        heavy_ids,
        key=lambda entity_id: (entity_id != critical_name, entity_id),
    )
    for entity_id in ordered_heavy[max_heavy:]:
        old = blockers[entity_id]
        blockers[entity_id] = BlockerState(
            entity_id=old.entity_id,
            cells=old.cells,
            required_pushers=1,
            mass_class="light",
            task_weight=old.task_weight,
            allowed_push_directions=old.allowed_push_directions,
            display_name=old.display_name,
        )
    heavy_ids = [
        entity_id
        for entity_id, blocker in blockers.items()
        if blocker.required_pushers == 2
    ]

    # Choose one critical heavy blocker for a possible viewpoint-dependent
    # contact.  Its geometry is embedded in the multi-object scene rather than
    # defining the entire task.
    critical_id = f"C{critical_index}" if critical_index is not None else None
    critical = blockers.get(critical_id) if critical_id is not None else None
    future_support_required = planned_future_support and critical is not None

    occupied = set(walls) | set().union(*(set(b.cells) for b in blockers.values()))

    def side_candidates(side: str) -> list[tuple[int, int]]:
        if side == "lower":
            y_min = max(1, channel_y - observation_radius + 1)
            y_max = channel_y - 1
        else:
            y_min = channel_y + 2
            y_max = min(height - 2, channel_y + observation_radius)
        return [
            (x, y)
            for x in range(1, width - 1)
            for y in range(y_min, y_max + 1)
            if (x, y) not in occupied
        ]

    lower_candidates = side_candidates("lower")
    upper_candidates = side_candidates("upper")
    if not lower_candidates or not upper_candidates:
        raise RuntimeError("Both channel sides must contain collision-free robot poses")

    def support_side(direction: Direction) -> str:
        return "lower" if direction == Direction.NORTH else "upper"

    forced_anchors: list[tuple[int, int]] = []
    if future_support_required and critical is not None:
        critical_cells = sorted(critical.cells)
        left_x = min(x for x, _ in critical_cells)
        right_x = max(x for x, _ in critical_cells)
        direction = sorted(critical.allowed_push_directions, key=lambda d: d.value)[0]
        required_side = support_side(direction)
        anchor_y = channel_y - 2 if required_side == "lower" else channel_y + 3
        # The root view is placed to the left of the two-cell critical object;
        # this forced anchor supplies the complementary right-side viewpoint.
        anchor_x = right_x + 2
        anchor_x = min(width - 2, max(1, anchor_x))
        forced_anchors.append((anchor_x, anchor_y))

    stride = (4, 3, 3)[level]
    anchors = _stratified_channel_anchors(
        width=width,
        height=height,
        channel_y=channel_y,
        occupied=occupied,
        level=level,
        rng=rng,
        forced=forced_anchors,
    )

    first_blocker = blockers[sorted(blockers)[0]]
    first_direction = (
        primary_direction
        if primary_direction in first_blocker.allowed_push_directions
        else sorted(first_blocker.allowed_push_directions, key=lambda d: d.value)[0]
    )
    first_cell = min(first_blocker.cells)
    dx, dy = first_direction.delta
    first_support = (first_cell[0] - dx, first_cell[1] - dy)

    def choose_spread(
        candidates: list[tuple[int, int]],
        *,
        near: tuple[int, int] | None = None,
        exclude: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        excluded = exclude or set()
        pool = [cell for cell in candidates if cell not in excluded]
        if not pool:
            raise RuntimeError("No collision-free randomized robot pose is available")
        if near is not None:
            near_pool = [
                cell
                for cell in pool
                if abs(cell[0] - near[0]) + abs(cell[1] - near[1]) <= 4
            ]
            if near_pool:
                pool = near_pool
        return rng.choice(pool)

    def choose_nearest(
        candidates: list[tuple[int, int]],
        target: tuple[int, int],
        *,
        exclude: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        excluded = exclude or set()
        pool = [cell for cell in candidates if cell not in excluded]
        if not pool:
            raise RuntimeError("No collision-free randomized robot pose is available")
        return min(
            pool,
            key=lambda cell: (
                abs(cell[0] - target[0]) + abs(cell[1] - target[1]),
                cell,
            ),
        )

    if future_support_required and critical is not None:
        from coimagine.envs.grid.environment import GridEnvironment

        direction = next(iter(critical.allowed_push_directions))
        required_side = support_side(direction)
        initial_side = required_side
        side_pool = lower_candidates if required_side == "lower" else upper_candidates
        left_x = min(x for x, _ in critical.cells)
        view_y = channel_y - 2 if required_side == "lower" else channel_y + 3
        anchor_x = forced_anchors[0][0] if forced_anchors else left_x + 3
        root_view_x = anchor_x - observation_radius
        forced_anchor = forced_anchors[0] if forced_anchors else (anchor_x, view_y)
        probe_state = GridState(
            width=width,
            height=height,
            walls=frozenset(walls),
            robots={},
            blockers=blockers,
            task=TaskSpec(kind=TaskKind.CLEARING),
        )
        probe_env = GridEnvironment(observation_radius=observation_radius)
        by_visible_contact: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for candidate in side_pool:
            visible = probe_env.visible_cells_from_pose(
                probe_state,
                candidate,
                exclude_robot=None,
            )
            visible_critical = set(critical.cells) & set(visible)
            if len(visible_critical) != 1 or forced_anchor not in visible:
                continue
            contact = next(iter(visible_critical))
            by_visible_contact.setdefault(contact, []).append(candidate)

        causal_groups = [
            group for group in by_visible_contact.values() if len(group) >= 2
        ]
        if causal_groups:
            group = max(
                causal_groups,
                key=lambda cells: max(
                    abs(a[0] - b[0]) + abs(a[1] - b[1])
                    for a in cells
                    for b in cells
                ),
            )
            r0 = min(
                group,
                key=lambda cell: (
                    abs(cell[0] - root_view_x) + abs(cell[1] - view_y),
                    cell,
                ),
            )
            r1 = max(
                (cell for cell in group if cell != r0),
                key=lambda cell: (
                    abs(cell[0] - r0[0]) + abs(cell[1] - r0[1]),
                    -abs(cell[0] - root_view_x) - abs(cell[1] - view_y),
                ),
            )
        else:
            # The structural validator rejects this attempt; the deterministic
            # fallback only keeps construction total for the retry loop.
            vertical_offset = -1 if required_side == "lower" else 1
            r0 = choose_nearest(side_pool, (root_view_x, view_y))
            r1 = choose_nearest(
                side_pool,
                (root_view_x + 1, view_y + vertical_offset),
                exclude={r0},
            )
        positions = [r0, r1]
    elif initial_position_mode == "same_side":
        side_pool = lower_candidates if initial_side == "lower" else upper_candidates
        near = first_support if support_side(first_direction) == initial_side else None
        r0 = choose_spread(side_pool, near=near)
        far_pool = [
            cell
            for cell in side_pool
            if cell != r0 and abs(cell[0] - r0[0]) + abs(cell[1] - r0[1]) >= 3
        ]
        r1 = choose_spread(far_pool or side_pool, exclude={r0})
        positions = [r0, r1]
    elif initial_position_mode == "split_side":
        lower_near = first_support if support_side(first_direction) == "lower" else None
        upper_near = first_support if support_side(first_direction) == "upper" else None
        r0 = choose_spread(lower_candidates, near=lower_near)
        r1 = choose_spread(upper_candidates, near=upper_near)
        positions = [r0, r1]
    else:
        # Cross-side stress without a causal label: both robots start opposite to
        # the first grounded blocker and may need ordinary closed-loop reposition.
        direction = first_direction
        required_side = support_side(direction)
        wrong_side = "upper" if required_side == "lower" else "lower"
        side_pool = upper_candidates if wrong_side == "upper" else lower_candidates
        blocker_centre = (
            round(sum(x for x, _ in first_blocker.cells) / len(first_blocker.cells)),
            channel_y,
        )
        r0 = choose_spread(side_pool, near=blocker_centre)
        far_pool = [
            cell
            for cell in side_pool
            if cell != r0 and abs(cell[0] - r0[0]) + abs(cell[1] - r0[1]) >= 3
        ]
        r1 = choose_spread(far_pool or side_pool, exclude={r0})
        positions = [r0, r1]

    # Geometry-only collision repair. Unlike v0.9, fallback poses may be drawn
    # from either side, preserving the intended same/split/cross-side mixture.
    all_pose_candidates = lower_candidates + upper_candidates
    resolved_positions: list[tuple[int, int]] = []
    for desired in positions:
        if (
            desired not in occupied
            and desired not in resolved_positions
            and 0 < desired[0] < width - 1
            and 0 < desired[1] < height - 1
        ):
            resolved_positions.append(desired)
            continue
        alternatives = [
            cell
            for cell in all_pose_candidates
            if cell not in resolved_positions
        ]
        if not alternatives:
            raise RuntimeError("No collision-free randomized robot pose is available")
        resolved_positions.append(
            min(
                alternatives,
                key=lambda cell: (
                    abs(cell[0] - desired[0]) + abs(cell[1] - desired[1]),
                    cell,
                ),
            )
        )
    positions = resolved_positions

    # Add sparse static occluders outside the channel band.  They alter sensing
    # and local reachability without sealing the lateral passage openings.
    support_cells = {
        (cell[0] - direction.delta[0], cell[1] - direction.delta[1])
        for blocker in blockers.values()
        for cell in blocker.cells
        for direction in blocker.allowed_push_directions
        if 0 < cell[0] - direction.delta[0] < width - 1
        and 0 < cell[1] - direction.delta[1] < height - 1
    }
    forbidden = (
        set(occupied)
        | set(positions)
        | set(anchors)
        | set(corridor)
        | support_cells
    )
    occluder_max = (
        (0, 1, 1)[level]
        if future_support_required
        else (1, 2, 2)[level]
    )
    occluder_budget = rng.randint(0, occluder_max)
    occluder_candidates = [
        (x, y)
        for x in range(2, width - 2)
        for y in range(max(1, channel_y - 5), min(height - 1, channel_y + 5))
        if y not in channel_rows and (x, y) not in forbidden
    ]
    rng.shuffle(occluder_candidates)
    for cell in occluder_candidates[:occluder_budget]:
        walls.add(cell)

    # Optional activation chains create delayed but still physically local task
    # dependencies.  They are sampled independently of planner performance.
    activation_dependencies: dict[str, str] = {}
    if (
        not future_support_required
        and level >= 1
        and len(blockers) >= 3
        and rng.random() < (0.12, 0.22)[level - 1]
    ):
        ordered_ids = sorted(blockers, key=lambda key: min(x for x, _ in blockers[key].cells))
        child_index = rng.randint(1, len(ordered_ids) - 1)
        activation_dependencies[ordered_ids[child_index]] = ordered_ids[child_index - 1]

    # Every object may require Reposition -> Approach -> Push. The feasibility
    # budget therefore has a hard 3M floor; difficulty comes from geometry,
    # sensing, candidate competition, coalition requirements, and dependencies,
    # rather than an episode budget that is insufficient by construction.
    candidate_budget = rng.randint((6, 6, 7)[level], (9, 9, 10)[level])
    minimum_episode_steps = 3 * blocker_count
    # Generated slack is used only when the CLI does not explicitly override it.
    # A small recovery allowance prevents the nominal 3M lower bound from becoming
    # an accidental hard cap in scenes with cross-side preparation or replanning.
    extra_steps = rng.randint((2, 3, 3)[level], (4, 5, 6)[level])
    episode_max_steps = minimum_episode_steps + extra_steps

    initial_weight = sum(
        blocker.task_weight * len(blocker.cells & frozenset(corridor))
        for blocker in blockers.values()
    )
    task = TaskSpec(
        kind=TaskKind.CLEARING,
        corridor=frozenset(corridor),
        initial_blocked_cells=sum(len(blocker.cells & frozenset(corridor)) for blocker in blockers.values()),
        initial_blocked_weight=initial_weight,
        activation_dependencies=activation_dependencies,
        metadata={
            "candidate_budget": candidate_budget,
            "episode_max_steps": episode_max_steps,
            "minimum_episode_steps": minimum_episode_steps,
            "enable_reposition": True,
            "reposition_anchors": tuple(dict.fromkeys(anchors)),
            "max_reposition_candidates": min(len(anchors), (3, 4, 4)[level]),
            "observation_radius": observation_radius,
            "anchor_stride": stride,
            "anchor_sampling": "two_sided_stratified_farthest_point",
            "scene_type": "open",
            "scene_family": "randomized_channel",
            "scene_subprofile": (
                "future_support_core"
                if future_support_required
                else "root_supported_core"
            ),
            "dependency_depth": 3 if future_support_required else (2 if any(b.required_pushers == 2 for b in blockers.values()) else 1),
            "difficulty_variant": difficulty,
            "difficulty_level": level,
            "future_support_required": future_support_required,
            "critical_entity": critical_id or "",
            "blocker_count": blocker_count,
            "heavy_blocker_count": sum(b.required_pushers == 2 for b in blockers.values()),
            "occluder_count": occluder_budget,
            "activation_dependency_count": len(activation_dependencies),
            "channel_y": channel_y,
            "channel_boundary_rows": channel_rows,
            "initial_position_mode": initial_position_mode,
            "initial_side": initial_side,
            "north_push_entities": sum(
                Direction.NORTH in blocker.allowed_push_directions
                for blocker in blockers.values()
            ),
            "south_push_entities": sum(
                Direction.SOUTH in blocker.allowed_push_directions
                for blocker in blockers.values()
            ),
            "task_label": "Clear the randomized horizontal passage",
            "randomized_initial_positions": True,
            "outcome_screened": False,
            "structural_screening": "causal_geometry_only",
            "generation_attempt": generation_attempt,
            "benchmark_version": "v0.14",
        },
    )
    state = GridState(
        width=width,
        height=height,
        walls=frozenset(walls),
        robots={0: RobotState(0, positions[0]), 1: RobotState(1, positions[1])},
        blockers=blockers,
        task=task,
    )
    return ScenarioInstance(
        instance_id=f"randomized-channel-v014-{difficulty}-{seed}",
        suite="randomized_channel",
        task_name="randomized_multi_blocker_channel",
        seed=seed,
        state=state,
        tags=(
            "cst_vs_iet",
            "randomized_channel",
            difficulty,
            "multi_blocker",
            "randomized_initial_positions",
            "outcome_unscreened",
        ),
    )


def generate_randomized_channel_benchmark_case(
    seed: int,
    *,
    difficulty: str | None = None,
) -> ScenarioInstance:
    """Generate a causally calibrated, outcome-unscreened channel task.

    The future-support profile is fixed by ``seed``. Geometry is then retried
    deterministically until the corresponding structural label is valid. This is
    not outcome screening: CST, IET, scores, and completion are never evaluated.
    """

    base_seed = 91_827_341 * (seed + 1) + 7_919
    difficulty_rng = Random(base_seed)
    if difficulty is None:
        difficulty = RANDOMIZED_CHANNEL_DIFFICULTIES[difficulty_rng.randrange(3)]
    if difficulty not in RANDOMIZED_CHANNEL_DIFFICULTIES:
        raise ValueError(f"Unsupported randomized-channel difficulty: {difficulty}")
    level = RANDOMIZED_CHANNEL_DIFFICULTIES.index(difficulty)
    profile_rng = Random(base_seed + 2_000_033 * (level + 1))
    planned_future_support = profile_rng.random() < (0.35, 0.70, 0.80)[level]

    last_diagnostics: dict[str, object] = {}
    for generation_attempt in range(64):
        instance = _generate_randomized_channel_benchmark_case_once(
            seed,
            difficulty=difficulty,
            planned_future_support=planned_future_support,
            generation_attempt=generation_attempt,
        )
        diagnostics = _validate_randomized_channel_structure(
            instance,
            validation_candidate_budget=8,
        )
        last_diagnostics = diagnostics
        future_required = bool(
            instance.state.task.metadata.get("future_support_required", False)
        )
        valid = (
            bool(diagnostics.get("future_support_structure_valid", False))
            if future_required
            else bool(diagnostics.get("root_supported_structure_valid", False))
        )
        if not valid:
            continue
        metadata = dict(instance.state.task.metadata)
        metadata.update(diagnostics)
        metadata["generation_attempt"] = generation_attempt
        metadata["structurally_validated"] = True
        task = replace(instance.state.task, metadata=metadata)
        state = replace(instance.state, task=task)
        return replace(instance, state=state)

    raise RuntimeError(
        "Unable to generate a structurally valid randomized-channel scene "
        f"after 64 attempts: seed={seed}, difficulty={difficulty}, "
        f"future_support={planned_future_support}, diagnostics={last_diagnostics}"
    )


def generate_diagnostic(task_name: str, seed: int) -> ScenarioInstance:
    if task_name == "edge_creation":
        state = two_robot_approach_then_push()
        return ScenarioInstance(
            instance_id=f"diagnostic-creation-{seed}",
            suite="diagnostic",
            task_name=task_name,
            seed=seed,
            state=state,
            tags=("created_edge_required", "first_action_reversal_expected"),
        )
    if task_name == "edge_destruction":
        return generate_edge_destruction(seed)
    if task_name == "edge_replacement":
        return generate_edge_replacement(seed)
    if task_name == "positive_higher_order_synergy":
        return generate_positive_higher_order_synergy(seed)
    if task_name == "higher_order_conflict":
        return generate_higher_order_conflict(seed)
    if task_name == "hold_pass":
        return generate_hold_pass(seed)
    if task_name == "reveal_act":
        return generate_reveal_act(seed)
    raise ValueError(f"Unknown diagnostic task: {task_name}")


def natural_instances(
    seeds: Iterable[int],
    *,
    blocker_counts: tuple[int, ...] = (1, 2, 3),
    robot_counts: tuple[int, ...] = (2, 3),
) -> list[ScenarioInstance]:
    instances: list[ScenarioInstance] = []
    for seed in seeds:
        for blockers in blocker_counts:
            instances.append(generate_multi_blocker(seed * 100 + blockers, blockers, 2))
        for robots in robot_counts:
            instances.append(generate_congested_transport(seed * 1000 + robots, robots))
        instances.append(generate_stable_control(seed))
    return instances
