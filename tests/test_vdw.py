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


class TestLondonDispersion:
    """The new vdw_dispersion_scale path uses the real London formula:
        C₆ = (3/2) α₁ α₂ · I₁·I₂/(I₁+I₂)
        F  = 6 C₆ / r⁷
    All inputs are measured atomic data (polarisability + first IE)."""

    def test_london_force_falls_off_as_one_over_r_seven(self, cfg):
        """Doubling the distance should reduce the force by 2⁷ = 128×."""
        cfg.thermal.vdw_strength = 0.0       # disable legacy
        cfg.thermal.vdw_dispersion_scale = 1e10   # enable London
        def f_at(sep):
            w = World(cfg)
            _place(w, 'He', [0.0, 0.0, 0.0])
            _place(w, 'He', [sep, 0.0, 0.0])
            forces = np.zeros((w.n, 3))
            add_vdw_forces(w, forces)
            return abs(forces[0, 0])
        f_short = f_at(100.0)
        f_long  = f_at(200.0)
        assert f_short > 0
        assert f_long  > 0
        # 2⁷ = 128; allow numerical slack
        assert f_short / f_long == pytest.approx(128.0, rel=0.01)

    def test_london_stronger_for_more_polarisable_pair(self, cfg):
        """At the same distance, a more polarisable pair has stronger
        attraction. C (α=1.76 Å³) > F (α=0.557 Å³), so C-C > F-F."""
        cfg.thermal.vdw_strength = 0.0
        cfg.thermal.vdw_dispersion_scale = 1e10
        def f_pair(sym, sep=200.0):
            w = World(cfg)
            _place(w, sym, [0.0, 0.0, 0.0])
            _place(w, sym, [sep, 0.0, 0.0])
            forces = np.zeros((w.n, 3))
            add_vdw_forces(w, forces)
            return abs(forces[0, 0])
        # Both elements have r_min < 200 (C rc=77, F rc=57) so VdW is active
        f_c = f_pair('C')
        f_f = f_pair('F')
        assert f_c > f_f

    def test_disabled_when_scale_zero(self, cfg):
        cfg.thermal.vdw_strength = 0.0
        cfg.thermal.vdw_dispersion_scale = 0.0
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [150.0, 0.0, 0.0])
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
