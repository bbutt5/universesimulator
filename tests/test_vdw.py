"""Tests for sim/vdw.py — van der Waals attractive force."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.vdw import add_vdw_forces


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestVdWAttraction:
    def test_cold_neutral_pair_attracts(self, cfg):
        w = World(cfg)
        # Two He atoms (max_bonds=0, never covalent) separated by 150 SU
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        # Force on 0 should point toward +x (toward 1); force on 1 toward −x
        assert forces[0, 0] > 0
        assert forces[1, 0] < 0

    def test_newton_third_law(self, cfg):
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-12)

    def test_force_stronger_when_closer(self, cfg):
        def _force_at(sep):
            w = World(cfg)
            _place(w, 'He', [0.0, 0.0, 0.0])
            _place(w, 'He', [sep, 0.0, 0.0])
            forces = np.zeros((w.n, 3))
            add_vdw_forces(w, forces)
            return abs(forces[0, 0])
        assert _force_at(100.0) > _force_at(250.0)

    def test_no_force_beyond_cutoff(self, cfg):
        w = World(cfg)
        cutoff = cfg.thermal.vdw_cutoff
        _place(w, 'He', [0.0,           0.0, 0.0])
        _place(w, 'He', [cutoff + 50.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)


class TestVdWInhibitors:
    def test_bonded_pair_skipped(self, cfg):
        from sim.particle import Bond
        w = World(cfg)
        _place(w, 'H', [0.0,   0.0, 0.0])
        _place(w, 'H', [150.0, 0.0, 0.0])
        w.bonds.append(Bond(0, 1, 62.0, 50.0, 0.8))
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_ionized_particle_skipped(self, cfg):
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0])
        w.ionized[0] = True
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_covalent_regime_distance_skipped(self, cfg):
        """Pairs inside r_min (covalent regime) should not feel VdW."""
        w = World(cfg)
        # He covalent_radius=28 → r_min=56; place at 30 (well inside)
        _place(w, 'He', [0.0,  0.0, 0.0])
        _place(w, 'He', [30.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_skipped_when_thermal_cfg_absent(self, cfg):
        del cfg.thermal
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_array_equal(forces, 0.0)

    def test_no_force_single_particle(self, cfg):
        w = World(cfg)
        _place(w, 'He', [0.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_vdw_forces(w, forces)
        np.testing.assert_array_equal(forces, 0.0)


class TestVdWvsThermalPressure:
    """The two forces should oppose each other: VdW wins when cold, thermal
    pressure wins when hot."""

    def test_cold_pair_net_attractive(self, cfg):
        from sim.thermal import add_thermal_pressure
        w = World(cfg)
        # Two cold He atoms (no relative motion → no thermal pressure)
        _place(w, 'He', [0.0,   0.0, 0.0], vel=[0.0, 0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        add_vdw_forces(w, forces)
        # Net attractive
        assert forces[0, 0] > 0
        assert forces[1, 0] < 0

    def test_hot_pair_net_repulsive(self, cfg):
        from sim.thermal import add_thermal_pressure
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0], vel=[ 80.0, 0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0], vel=[-80.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        add_vdw_forces(w, forces)
        # Net repulsive
        assert forces[0, 0] < 0
        assert forces[1, 0] > 0
