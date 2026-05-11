"""Tests for sim/physics.py — gravity backend equivalence and properties.

The numba and NumPy backends must produce numerically identical results
for the same input — otherwise the numba path is unsafe to ship as a
"transparent" speedup."""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS
from sim import physics


HAS_NUMBA = physics.HAS_NUMBA
needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason='numba not installed')


def _place(world, sym, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


def _seed_pairs(world, n, seed=42):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        pos = rng.normal(0, 200, 3)
        vel = rng.normal(0, 5, 3)
        _place(world, 'Fe', pos, vel)


class TestNumPyBackendBasics:
    def test_isolated_particle_no_force(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0, 0, 0])
        forces = np.zeros((1, 3))
        physics.add_gravity_numpy(w, forces)
        np.testing.assert_array_equal(forces, 0)

    def test_pair_attractive(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0,   0.0, 0.0])
        _place(w, 'Fe', [100.0, 0.0, 0.0])
        forces = np.zeros((2, 3))
        physics.add_gravity_numpy(w, forces)
        # Particle 0 should be pulled toward +x; particle 1 toward −x
        assert forces[0, 0] > 0
        assert forces[1, 0] < 0

    def test_newton_third_law(self, cfg):
        w = World(cfg)
        _seed_pairs(w, 10)
        forces = np.zeros((w.n, 3))
        physics.add_gravity_numpy(w, forces)
        total = forces.sum(axis=0)
        # Σ F_i should be machine-precision zero for an isolated system
        np.testing.assert_allclose(total, 0.0, atol=1e-8)


@needs_numba
class TestBackendEquivalence:
    """The numba backend must match the NumPy backend on the same input."""

    @pytest.mark.parametrize('n', [2, 5, 20, 100])
    def test_force_field_matches(self, cfg, n):
        w = World(cfg)
        _seed_pairs(w, n)

        f_np = np.zeros((w.n, 3))
        f_nb = np.zeros((w.n, 3))
        physics.add_gravity_numpy(w, f_np)
        physics.add_gravity_numba(w, f_nb)

        np.testing.assert_allclose(f_np, f_nb, rtol=1e-10, atol=1e-10)

    def test_numba_newton_third_law(self, cfg):
        w = World(cfg)
        _seed_pairs(w, 30)
        f = np.zeros((w.n, 3))
        physics.add_gravity_numba(w, f)
        total = f.sum(axis=0)
        # Pairwise forces should cancel system-wide (Newton 3 emergent here)
        np.testing.assert_allclose(total, 0.0, atol=1e-8)


@needs_numba
class TestDispatch:
    def test_default_dispatch_picks_numba_when_available(self):
        assert physics.add_gravity is physics.add_gravity_numba

    def test_dispatch_is_module_attribute(self):
        """Other modules import 'add_gravity' by name — that name must
        always point to the live function regardless of backend."""
        assert callable(physics.add_gravity)
