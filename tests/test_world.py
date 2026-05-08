"""Tests for sim/world.py — World state management and step loop."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond


@pytest.fixture
def world(cfg):
    return World(cfg)


class TestParticleManagement:
    def test_initial_state(self, world):
        assert world.n == 0
        assert world.time == 0.0
        assert world.bonds == []
        assert world.total_injected == 0

    def test_add_particle_increments_count(self, world):
        h = ELEMENTS['H']
        world.add_particle(h, np.zeros(3), np.zeros(3))
        assert world.n == 1
        assert world.total_injected == 1

    def test_add_particle_stores_position_velocity(self, world):
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, 0.2, 0.3])
        world.add_particle(ELEMENTS['H'], pos, vel)
        np.testing.assert_array_equal(world.positions[0], pos)
        np.testing.assert_array_equal(world.velocities[0], vel)

    def test_add_particle_stores_mass(self, world):
        world.add_particle(ELEMENTS['H'], np.zeros(3), np.zeros(3))
        assert world.masses[0] == pytest.approx(ELEMENTS['H'].mass)

    def test_remove_last_particle(self, world):
        h = ELEMENTS['H']
        world.add_particle(h, np.zeros(3), np.zeros(3))
        world.remove_particle(0)
        assert world.n == 0

    def test_remove_middle_swaps_with_last(self, world):
        h = ELEMENTS['H']
        c = ELEMENTS['C']
        world.add_particle(h, np.array([1.0, 0.0, 0.0]), np.zeros(3))
        world.add_particle(c, np.array([2.0, 0.0, 0.0]), np.zeros(3))
        world.add_particle(h, np.array([3.0, 0.0, 0.0]), np.zeros(3))

        world.remove_particle(0)   # removes index 0, last (index 2) swaps in

        assert world.n == 2
        # index 0 should now hold what was at index 2 (pos.x = 3.0)
        assert world.positions[0, 0] == pytest.approx(3.0)

    def test_remove_particle_cleans_bonds(self, world):
        h = ELEMENTS['H']
        world.add_particle(h, np.zeros(3), np.zeros(3))
        world.add_particle(h, np.array([62.0, 0.0, 0.0]), np.zeros(3))
        bond = Bond(0, 1, 62.0, 50.0, 0.8)
        world.bonds.append(bond)
        world.bond_counts[0] = 1
        world.bond_counts[1] = 1

        world.remove_particle(0)

        assert len(world.bonds) == 0
        assert world.bond_counts[0] == 0   # was particle 1, now at index 0

    def test_remove_patches_bond_indices(self, world):
        h = ELEMENTS['H']
        for _ in range(3):
            world.add_particle(h, np.zeros(3), np.zeros(3))
        # Bond between 1 and 2
        bond = Bond(1, 2, 62.0, 50.0, 0.8)
        world.bonds.append(bond)
        world.bond_counts[1] = 1
        world.bond_counts[2] = 1

        world.remove_particle(0)   # swap: index 2 → index 0

        surviving = world.bonds
        assert len(surviving) == 1
        b = surviving[0]
        # 1 stays 1, 2 becomes 0
        assert b.i == 1 or b.j == 1
        assert 0 in (b.i, b.j)


class TestCapacityGrowth:
    def test_grows_beyond_initial_allocation(self, world):
        from sim.world import _INIT_CAP
        h = ELEMENTS['H']
        for _ in range(_INIT_CAP + 1):
            world.add_particle(h, np.zeros(3), np.zeros(3))
        assert world.n == _INIT_CAP + 1
        assert world._cap >= _INIT_CAP + 1


class TestActiveSliceProperties:
    def test_pos_property_matches_positions_slice(self, world):
        world.add_particle(ELEMENTS['H'], np.array([1.0, 2.0, 3.0]), np.zeros(3))
        np.testing.assert_array_equal(world.pos, world.positions[:world.n])

    def test_vel_property(self, world):
        world.add_particle(ELEMENTS['H'], np.zeros(3), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(world.vel, world.velocities[:world.n])


class TestStepLoop:
    def test_step_advances_time(self, world):
        dt = 0.01
        world.step(dt)
        assert world.time == pytest.approx(dt)

    def test_step_injects_particles_over_time(self, world):
        dt = world.cfg.simulation.time_step
        for _ in range(200):
            world.step(dt)
        assert world.n > 0

    def test_step_empty_world_advances_time(self, world):
        world.cfg.simulation.injection_rate = 0
        world.cfg.simulation.max_particles  = 0
        world.step(0.01)
        assert world.time == pytest.approx(0.01)
