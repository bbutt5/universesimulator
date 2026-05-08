"""Tests for sim/spatial.py — spatial hash utilities."""

from __future__ import annotations
import numpy as np
import pytest
from sim.spatial import build_grid, neighbors


class TestBuildGrid:
    def test_single_particle(self):
        pos  = np.array([[5.0, 5.0, 5.0]])
        grid = build_grid(pos, cell_size=10.0)
        assert len(grid) == 1
        assert 0 in next(iter(grid.values()))

    def test_two_particles_same_cell(self):
        pos  = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        grid = build_grid(pos, cell_size=10.0)
        assert len(grid) == 1
        cell = next(iter(grid.values()))
        assert 0 in cell and 1 in cell

    def test_two_particles_different_cells(self):
        pos  = np.array([[0.5, 0.5, 0.5], [15.0, 15.0, 15.0]])
        grid = build_grid(pos, cell_size=10.0)
        assert len(grid) == 2

    def test_all_particles_indexed(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 100, (20, 3))
        grid = build_grid(pos, cell_size=20.0)
        all_indices = [idx for indices in grid.values() for idx in indices]
        assert sorted(all_indices) == list(range(20))


class TestNeighbors:
    def test_finds_nearby_particle(self):
        pos  = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        grid = build_grid(pos, cell_size=10.0)
        result = neighbors(0, pos, grid, cell_size=10.0)
        assert 1 in result

    def test_does_not_find_distant_particle(self):
        pos  = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        grid = build_grid(pos, cell_size=10.0)
        result = neighbors(0, pos, grid, cell_size=10.0)
        assert 1 not in result

    def test_includes_self(self):
        pos  = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        grid = build_grid(pos, cell_size=10.0)
        result = neighbors(0, pos, grid, cell_size=10.0)
        assert 0 in result

    def test_stencil_covers_all_26_neighbours(self):
        # Place one particle at origin, 26 at cell-edge positions
        pos = np.zeros((27, 3))
        for i, dx in enumerate([-1, 0, 1]):
            for j, dy in enumerate([-1, 0, 1]):
                for k, dz in enumerate([-1, 0, 1]):
                    idx = i * 9 + j * 3 + k
                    pos[idx] = [dx * 10.0, dy * 10.0, dz * 10.0]
        grid = build_grid(pos, cell_size=10.0)
        result = neighbors(13, pos, grid, cell_size=10.0)  # 13 = origin cell
        assert set(range(27)) == set(result)
