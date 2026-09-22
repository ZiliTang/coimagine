from __future__ import annotations

from coimagine.core.types import Cell, Direction


def add_cell(a: Cell, b: Cell) -> Cell:
    return a[0] + b[0], a[1] + b[1]


def sub_cell(a: Cell, b: Cell) -> Cell:
    return a[0] - b[0], a[1] - b[1]


def translate(cells: set[Cell] | frozenset[Cell], direction: Direction) -> frozenset[Cell]:
    dx, dy = direction.delta
    return frozenset((x + dx, y + dy) for x, y in cells)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
