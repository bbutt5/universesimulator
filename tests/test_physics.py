"""Tests for sim/physics.py — gravity and bond forces."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond
from sim import physics


@pytest.fixture
def two_particle_world(cfg):
    """World with two H atoms placed on the x-axis."""
    world = World(cfg)
    h = ELEMENTS['H']
    world.add_particle(h, np.array([-100.0, 0.0, 0.0]), np.zeros(3))
    world.add_particle(h, np.array([ 100.0, 0.0, 0.0]), np.zeros(3))
    return world


class TestGravity:
    def test_gravity_is_attractive(self, two_particle_world):
        w = two_particle_world
        forces = np.zeros((w.n, 3))
        physics.add_gravity(w, forces)
        # Particle 0 is at -100, should be pulled in +x direction
        assert forces[0, 0] > 0
        # Particle 1 is at +100, should be pulled in -x direction
        assert forces[1, 0] < 0

    def test_gravity_satisfies_newton_third_law(self, two_particle_world):
        w = two_particle_world
        forces = np.zeros((w.n, 3))
        physics.add_gravity(w, forces)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-12)

    def test_gravity_zero_with_single_particle(self, cfg):
        w = World(cfg)
        w.add_particle(ELEMENTS['H'], np.zeros(3), np.zeros(3))
        forces = np.zeros((w.n, 3))
        physics.add_gravity(w, forces)
        np.testing.assert_array_equal(forces, 0)

    def test_gravity_stronger_when_closer(self, cfg):
        h = ELEMENTS['H']

        w_close = World(cfg)
        w_close.add_particle(h, np.array([-10.0, 0.0, 0.0]), np.zeros(3))
        w_close.add_particle(h, np.array([ 10.0, 0.0, 0.0]), np.zeros(3))
        f_close = np.zeros((2, 3))
        physics.add_gravity(w_close, f_close)

        w_far = World(cfg)
        w_far.add_particle(h, np.array([-100.0, 0.0, 0.0]), np.zeros(3))
        w_far.add_particle(h, np.array([ 100.0, 0.0, 0.0]), np.zeros(3))
        f_far = np.zeros((2, 3))
        physics.add_gravity(w_far, f_far)

        assert abs(f_close[0, 0]) > abs(f_far[0, 0])

    def test_gravity_scales_with_mass(self, cfg):
        h = ELEMENTS['H']
        c = ELEMENTS['C']   # heavier

        w_light = World(cfg)
        w_light.add_particle(h, np.array([-50.0, 0.0, 0.0]), np.zeros(3))
        w_light.add_particle(h, np.array([ 50.0, 0.0, 0.0]), np.zeros(3))
        f_light = np.zeros((2, 3))
        physics.add_gravity(w_light, f_light)

        w_heavy = World(cfg)
        w_heavy.add_particle(c, np.array([-50.0, 0.0, 0.0]), np.zeros(3))
        w_heavy.add_particle(c, np.array([ 50.0, 0.0, 0.0]), np.zeros(3))
        f_heavy = np.zeros((2, 3))
        physics.add_gravity(w_heavy, f_heavy)

        assert abs(f_heavy[0, 0]) > abs(f_light[0, 0])


class TestBondForces:
    def _world_with_bond(self, cfg, r: float, r_eq: float = 62.0) -> World:
        """Two H atoms separated by r, bonded with equilibrium length r_eq."""
        w = World(cfg)
        h = ELEMENTS['H']
        w.add_particle(h, np.array([0.0, 0.0, 0.0]), np.zeros(3))
        w.add_particle(h, np.array([r,   0.0, 0.0]), np.zeros(3))
        bond = Bond(0, 1, r_eq, 50.0, 0.8)
        w.bonds.append(bond)
        return w

    def test_no_force_at_equilibrium(self, cfg):
        r_eq = 62.0
        w = self._world_with_bond(cfg, r=r_eq, r_eq=r_eq)
        forces = np.zeros((2, 3))
        physics.add_bond_forces(w, forces)
        np.testing.assert_allclose(forces, 0, atol=1e-10)

    def test_attractive_when_stretched(self, cfg):
        r_eq = 62.0
        w = self._world_with_bond(cfg, r=r_eq * 1.5, r_eq=r_eq)
        forces = np.zeros((2, 3))
        physics.add_bond_forces(w, forces)
        # particle 0 at origin should be pulled toward particle 1 (+x)
        assert forces[0, 0] > 0
        # particle 1 should be pulled back (-x)
        assert forces[1, 0] < 0

    def test_repulsive_when_compressed(self, cfg):
        r_eq = 62.0
        w = self._world_with_bond(cfg, r=r_eq * 0.5, r_eq=r_eq)
        forces = np.zeros((2, 3))
        physics.add_bond_forces(w, forces)
        # particle 0 should be pushed away from particle 1 (-x)
        assert forces[0, 0] < 0
        # particle 1 should be pushed away (+x)
        assert forces[1, 0] > 0

    def test_newton_third_law_for_bonds(self, cfg):
        r_eq = 62.0
        w = self._world_with_bond(cfg, r=r_eq * 1.3, r_eq=r_eq)
        forces = np.zeros((2, 3))
        physics.add_bond_forces(w, forces)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-12)

    def test_no_bonds_no_force(self, cfg):
        w = World(cfg)
        h = ELEMENTS['H']
        w.add_particle(h, np.zeros(3), np.zeros(3))
        w.add_particle(h, np.array([100.0, 0.0, 0.0]), np.zeros(3))
        forces = np.zeros((2, 3))
        physics.add_bond_forces(w, forces)
        np.testing.assert_array_equal(forces, 0)
