"""Collision-free point-to-point routing for the feedback controllers.

A ``move_to`` that drives straight at its target will happily push the robot through the
object. That is not a planning result, it is a controller defect: a real robot has to go
around. This module provides a small 8-connected A* over the configuration space, with the
walls and the object inflated by the robot radius, plus a line-of-sight shortcut so a clear
straight run stays a straight run.

It plans **one robot's path to one target**. It is not a task solver: it never sees the
goal region, the candidate set or the task score, and it cannot choose what to do.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

from .scenes import Box

Vec = Tuple[float, float]


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _inside_box(x: float, y: float, box: Box) -> bool:
    cos, sin = math.cos(-box.angle), math.sin(-box.angle)
    dx, dy = x - box.cx, y - box.cy
    local_x = dx * cos - dy * sin
    local_y = dx * sin + dy * cos
    return abs(local_x) <= box.half_w and abs(local_y) <= box.half_h


def _distance_to_box(x: float, y: float, box: Box) -> float:
    if _inside_box(x, y, box):
        return 0.0
    cos, sin = math.cos(-box.angle), math.sin(-box.angle)
    dx, dy = x - box.cx, y - box.cy
    local_x = dx * cos - dy * sin
    local_y = dx * sin + dy * cos
    clamped_x = max(-box.half_w, min(box.half_w, local_x))
    clamped_y = max(-box.half_h, min(box.half_h, local_y))
    return math.hypot(local_x - clamped_x, local_y - clamped_y)


def _distance_to_polygon(x: float, y: float, corners: Sequence[Vec]) -> float:
    inside = True
    sign = 0
    for index, corner in enumerate(corners):
        nxt = corners[(index + 1) % len(corners)]
        cross = (nxt[0] - corner[0]) * (y - corner[1]) - (nxt[1] - corner[1]) * (x - corner[0])
        if abs(cross) < 1e-12:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            inside = False
            break
    if inside and sign != 0:
        return 0.0
    return min(
        _point_segment_distance(x, y, corners[i][0], corners[i][1], corners[(i + 1) % len(corners)][0], corners[(i + 1) % len(corners)][1])
        for i in range(len(corners))
    )


def _clearance(
    x: float,
    y: float,
    arena: Tuple[float, float],
    obstacles: Sequence[Box],
    object_corners: Sequence[Vec],
) -> float:
    width, height = arena
    wall = min(x, y, width - x, height - y)
    obstacle = min((_distance_to_box(x, y, box) for box in obstacles), default=float("inf"))
    object_distance = _distance_to_polygon(x, y, object_corners) if object_corners else float("inf")
    return min(wall, obstacle, object_distance)


def line_clear(
    start: Vec,
    end: Vec,
    *,
    arena: Tuple[float, float],
    obstacles: Sequence[Box],
    object_corners: Sequence[Vec],
    robot_radius: float,
    samples: int = 24,
) -> bool:
    for index in range(samples + 1):
        t = index / samples
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        if _clearance(x, y, arena, obstacles, object_corners) < robot_radius:
            return False
    return True


def plan_path(
    start: Vec,
    goal: Vec,
    *,
    arena: Tuple[float, float],
    obstacles: Sequence[Box],
    object_corners: Sequence[Vec],
    robot_radius: float,
    resolution: float = 0.08,
) -> List[Vec]:
    """Waypoints from ``start`` to ``goal``; ``[goal]`` when the straight line is clear."""

    if line_clear(
        start, goal, arena=arena, obstacles=obstacles, object_corners=object_corners,
        robot_radius=robot_radius,
    ):
        return [goal]

    width, height = arena
    nx = max(2, int(width / resolution) + 1)
    ny = max(2, int(height / resolution) + 1)

    def to_cell(point: Vec) -> Tuple[int, int]:
        return (
            max(0, min(nx - 1, int(round(point[0] / resolution)))),
            max(0, min(ny - 1, int(round(point[1] / resolution)))),
        )

    def to_world(cell: Tuple[int, int]) -> Vec:
        return (cell[0] * resolution, cell[1] * resolution)

    def blocked(cell: Tuple[int, int]) -> bool:
        x, y = to_world(cell)
        return _clearance(x, y, arena, obstacles, object_corners) < robot_radius

    start_cell = to_cell(start)
    goal_cell = to_cell(goal)
    if blocked(start_cell):
        # The robot starts in contact with the object; allow the first cell.
        pass
    if blocked(goal_cell):
        return [goal]

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_cell))
    came_from = {}
    cost = {start_cell: 0.0}
    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            found = True
            break
        for dx, dy in neighbours:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                continue
            if blocked(nxt) and nxt != goal_cell:
                continue
            step = math.hypot(dx, dy)
            candidate = cost[current] + step
            if candidate < cost.get(nxt, float("inf")):
                cost[nxt] = candidate
                heuristic = math.hypot(nxt[0] - goal_cell[0], nxt[1] - goal_cell[1])
                heapq.heappush(open_heap, (candidate + heuristic, nxt))
                came_from[nxt] = current
    if not found:
        return [goal]

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came_from[cells[-1]])
    cells.reverse()

    waypoints: List[Vec] = []
    index = 1
    while index < len(cells):
        waypoints.append(to_world(cells[index]))
        index += 1
    # simplify: drop any waypoint that the previous one can see directly
    simplified: List[Vec] = []
    anchor: Vec = start
    for waypoint in waypoints:
        if line_clear(
            anchor, waypoint, arena=arena, obstacles=obstacles,
            object_corners=object_corners, robot_radius=robot_radius,
        ):
            simplified.append(waypoint)
            anchor = waypoint
    if not simplified or simplified[-1] != goal:
        simplified.append(goal)
    return simplified
