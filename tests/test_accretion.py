"""Tests for sim/accretion.py — gravitational accretion."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS, ELEMENTS_LIST
from sim import accretion


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestAccretionMerge:
    def test_bound_particles_merge(self, cfg):
        """Two inert particles with relative KE < gravitational PE should merge."""
        w = World(cfg)
        r = 50.0
        _place(w, 'Fe', [0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])
        _place(w, 'Fe', [r,   0.0, 0.0], vel=[0.0, 0.0, 0.0])  # at rest → deeply bound
        accretion.update(w)
        assert w.n == 1

    def test_merge_conserves_momentum(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0], vel=[1.0, 0.0, 0.0])
        _place(w, 'Fe', [50.0, 0.0, 0.0], vel=[3.0, 0.0, 0.0])
        m_fe = ELEMENTS['Fe'].mass
        p_before = 2 * m_fe * np.array([2.0, 0.0, 0.0])   # CoM momentum = m*(v1+v2)/2 *2
        accretion.update(w)
        assert w.n == 1
        p_after = w.masses[0] * w.velocities[0]
        np.testing.assert_allclose(p_after, m_fe * np.array([1.0, 0.0, 0.0]) +
                                            m_fe * np.array([3.0, 0.0, 0.0]), rtol=1e-10)

    def test_merge_combines_mass(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'Fe', [50.0, 0.0, 0.0])
        expected_mass = 2 * ELEMENTS['Fe'].mass
        accretion.update(w)
        assert w.n == 1
        assert w.masses[0] == pytest.approx(expected_mass)

    def test_merge_uses_heavier_element_identity(self, cfg):
        """After merging Fe + C, result should be Fe (heavier)."""
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])   # mass 55.8
        _place(w, 'C',  [50.0, 0.0, 0.0])   # mass 12.0
        accretion.update(w)
        assert w.n == 1
        assert ELEMENTS_LIST[w.elem_ids[0]].symbol == 'Fe'

    def test_successive_accretion_grows_mass(self, cfg):
        """Multiple merge events should accumulate mass correctly."""
        w   = World(cfg)
        m0  = ELEMENTS['Fe'].mass
        for x in [0.0, 50.0, 100.0]:
            _place(w, 'Fe', [x, 0.0, 0.0])
        # Two merge events over two steps
        accretion.update(w)
        accretion.update(w)
        assert w.n == 1
        assert w.masses[0] == pytest.approx(3 * m0)


class TestAccretionConditions:
    def test_unbound_particles_do_not_merge(self, cfg):
        """Fast-moving particles whose KE > PE should not accrete."""
        w = World(cfg)
        G = cfg.physics.gravity_constant
        r = 50.0
        m = ELEMENTS['Fe'].mass
        # Set relative speed well above escape velocity: v_esc = sqrt(2GM/r)
        v_esc = np.sqrt(2 * G * m / r)
        _place(w, 'Fe', [0.0, 0.0, 0.0], vel=[-v_esc * 3, 0.0, 0.0])
        _place(w, 'Fe', [r,   0.0, 0.0], vel=[ v_esc * 3, 0.0, 0.0])
        accretion.update(w)
        assert w.n == 2

    def test_fusable_particles_do_not_accrete(self, cfg):
        """H and He are fusable — they should never accrete, only fuse."""
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0])
        _place(w, 'H', [50.0, 0.0, 0.0])
        accretion.update(w)
        assert w.n == 2   # H has can_fuse=True → excluded from accretion

    def test_distant_particles_do_not_merge(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0,    0.0, 0.0])
        _place(w, 'Fe', [1000.0, 0.0, 0.0])  # beyond accretion_radius
        accretion.update(w)
        assert w.n == 2

    def test_skipped_when_accretion_cfg_absent(self, cfg):
        del cfg.accretion
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'Fe', [50.0, 0.0, 0.0])
        accretion.update(w)   # must not raise
        assert w.n == 2

    def test_total_accretions_counter(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'Fe', [50.0, 0.0, 0.0])
        accretion.update(w)
        assert w.total_accretions == 1

    def test_single_particle_no_crash(self, cfg):
        w = World(cfg)
        _place(w, 'Fe', [0.0, 0.0, 0.0])
        accretion.update(w)   # must not raise
        assert w.n == 1
