"""
Shared spatial hash — O(1) amortised neighbour lookup.

Used by chemistry, thermal, and accretion to avoid O(N²) pair loops.
Cell size should match the interaction cutoff radius so each cell has
only a handful of particles, keeping the 27-cell stencil search fast.
"""

from __future__ import annotations
import numpy as np


def build_grid(pos: np.ndarray, cell_size: float) -> dict:
    """Map each particle index to its integer grid cell."""
    grid: dict = {}
    for idx in range(len(pos)):
        cell = tuple((pos[idx] / cell_size).astype(int))
        grid.setdefault(cell, []).append(idx)
    return grid


def neighbors(idx: int, pos: np.ndarray, grid: dict, cell_size: float) -> list[int]:
    """Return all particle indices in the 27 cells surrounding particle idx."""
    cell = tuple((pos[idx] / cell_size).astype(int))
    result: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c2 = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if c2 in grid:
                    result.extend(grid[c2])
    return result
