"""Stage 2 scene library, authored as readable ASCII maps.

Legend (rows are listed top-to-bottom, i.e. decreasing ``y``)::

    #  wall (prior map, public)
    .  free cell
    T  goal region cell
    O  object cell (the 2x2 object must occupy a 2x2 block of them)
    A  robot A pose
    B  robot B pose

Authoring scenes as ASCII keeps the geometry reviewable and makes the rendered
figures easy to check by eye.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .state import (
    Cell,
    Direction,
    HiddenPhysics,
    ObjectState,
    RobotState,
    Scene,
    StaticMap,
)

LEGEND = {"wall": "#", "free": ".", "target": "T", "object": "O", "robot_a": "A", "robot_b": "B"}


def scene_from_ascii(
    rows: Sequence[str],
    *,
    name: str = "scene",
    hidden: Optional[HiddenPhysics] = None,
    headings: Optional[Mapping[int, Direction]] = None,
) -> Scene:
    """Build a :class:`Scene` from an ASCII map."""

    if not rows:
        raise ValueError("An ASCII scene needs at least one row.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All ASCII rows must have the same width.")
    height = len(rows)

    walls: set = set()
    target: set = set()
    object_cells: set = set()
    robots: List[RobotState] = []

    for row_index, row in enumerate(rows):
        y = height - 1 - row_index
        for x, char in enumerate(row):
            cell = (x, y)
            if char == "#":
                walls.add(cell)
            elif char == ".":
                pass
            elif char == "T":
                target.add(cell)
            elif char == "O":
                object_cells.add(cell)
            elif char == "A":
                robots.append(RobotState(0, cell, (headings or {}).get(0)))
            elif char == "B":
                robots.append(RobotState(1, cell, (headings or {}).get(1)))
            else:
                raise ValueError(f"Unknown ASCII scene character {char!r} at {cell}.")

    if not object_cells:
        raise ValueError("The scene must contain an object ('O' cells).")
    origin = (min(c[0] for c in object_cells), min(c[1] for c in object_cells))
    obj = ObjectState(origin)
    if obj.cells != frozenset(object_cells):
        raise ValueError(
            "Object cells must form a contiguous 2x2 block; got "
            f"{sorted(object_cells)}."
        )
    if len(robots) != 2:
        raise ValueError("The scene must contain exactly one 'A' and one 'B'.")

    scene = Scene(
        static=StaticMap(width, height, frozenset(walls), frozenset(target), name=name),
        obj=obj,
        robots=tuple(sorted(robots, key=lambda r: r.robot_id)),
        physics=hidden if hidden is not None else HiddenPhysics(),
    )
    return scene


def with_hidden(scene: Scene, physics: HiddenPhysics) -> Scene:
    """Attach hidden physics to a scene."""

    return replace(scene, physics=physics)


def with_target(scene: Scene, target_cells) -> Scene:
    """Replace the goal region programmatically.

    A 2x2 object moved by one cell always overlaps its previous footprint, so a
    one-push goal region cannot be written with ``T`` in the ASCII map (a cell cannot
    be both ``O`` and ``T``). Setting it explicitly keeps the scenes reviewable while
    allowing single-push task variants.
    """

    from .state import StaticMap

    static = StaticMap(
        width=scene.static.width,
        height=scene.static.height,
        walls=scene.static.walls,
        target=frozenset(target_cells),
        name=scene.static.name,
    )
    return replace(scene, static=static)


# --------------------------------------------------------------------------
# A0-A3 mechanism scenes (execution prompt stage 2)
# --------------------------------------------------------------------------

# Both robots are on the west face; one cooperative push east reaches the goal.
_A_BASE = (
    "......",
    "AOO...",
    "BOO...",
    "......",
    "......",
    "......",
)
#: Goal region one push east of the object's start (object starts at origin (1, 3)).
_A_TARGET = frozenset({(2, 3), (3, 3), (2, 4), (3, 4)})


def _a_scene(name: str, physics: HiddenPhysics) -> Scene:
    scene = scene_from_ascii(_A_BASE, name=name, hidden=physics)
    return with_target(scene, _A_TARGET)


def a0_public_sufficient() -> Scene:
    """A0: a cooperative push works for every contact type; messaging adds nothing."""

    return _a_scene(
        "A0_public_sufficient",
        HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")),
    )


def a1_stall_cause() -> Scene:
    """A1: the same push stalls for different reasons; recovery differs by cause."""

    return _a_scene(
        "A1_stall_cause",
        HiddenPhysics(resistance=2, constraint="free", traction=("grip", "grip")),
    )


def a2_preparation() -> Scene:
    """A2: a latched object needs an explicit release before the push."""

    return _a_scene(
        "A2_preparation",
        HiddenPhysics(resistance=1, constraint="constrained", traction=("grip", "grip")),
    )


def a3_identify_prepare_execute() -> Scene:
    """A3: identify the cause, prepare the contact, then execute."""

    return _a_scene(
        "A3_identify_prepare_execute",
        HiddenPhysics(resistance=2, constraint="constrained", traction=("grip", "grip")),
    )


def mechanism_scenes() -> Tuple[Scene, ...]:
    return (a0_public_sufficient(), a1_stall_cause(), a2_preparation(), a3_identify_prepare_execute())


# --------------------------------------------------------------------------
# Scene library
# --------------------------------------------------------------------------


_OPEN_TRANSPORT = (
    "......",
    "...TT.",
    "...TT.",
    ".OO...",
    ".OO...",
    "A..B..",
)

_PILLAR_OCCLUSION = (
    "....TT",
    ".#..TT",
    ".#.OO.",
    ".#.OO.",
    ".#....",
    "A....B",
)

_NARROW_PASSAGE = (
    "......",
    "...TT.",
    "##.##.",
    ".OO...",
    ".OO...",
    "A..B..",
)

_CORNER_GOAL = (
    "....TT",
    "....TT",
    "......",
    "..OO..",
    "..OO..",
    "AB....",
)


def open_transport() -> Scene:
    """Both robots can see the object; no occlusion."""

    return scene_from_ascii(
        _OPEN_TRANSPORT,
        name="open_transport",
        hidden=HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")),
        headings={0: Direction.EAST, 1: Direction.WEST},
    )


def pillar_occlusion() -> Scene:
    """A pillar hides the object from robot B but not from robot A."""

    return scene_from_ascii(
        _PILLAR_OCCLUSION,
        name="pillar_occlusion",
        hidden=HiddenPhysics(resistance=2, constraint="free", traction=("grip", "slip")),
        headings={0: Direction.EAST, 1: Direction.WEST},
    )


def narrow_passage() -> Scene:
    """The object must be pushed through a two-cell gap."""

    return scene_from_ascii(
        _NARROW_PASSAGE,
        name="narrow_passage",
        hidden=HiddenPhysics(resistance=2, constraint="constrained", traction=("grip", "grip")),
        headings={0: Direction.EAST, 1: Direction.WEST},
    )


def corner_goal() -> Scene:
    """The goal region sits in a corner, so the push direction is constrained."""

    return scene_from_ascii(
        _CORNER_GOAL,
        name="corner_goal",
        hidden=HiddenPhysics(resistance=1, constraint="constrained", traction=("slip", "grip")),
        headings={0: Direction.EAST, 1: Direction.EAST},
    )


def default_scenes() -> Tuple[Scene, ...]:
    return (open_transport(), pillar_occlusion(), narrow_passage(), corner_goal())


ASCII_BY_NAME: Mapping[str, Sequence[str]] = {
    "open_transport": _OPEN_TRANSPORT,
    "pillar_occlusion": _PILLAR_OCCLUSION,
    "narrow_passage": _NARROW_PASSAGE,
    "corner_goal": _CORNER_GOAL,
}
