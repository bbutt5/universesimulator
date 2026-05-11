"""Tests for sim/chemistry.py — bond formation/breaking and nuclear fusion."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond
from sim import chemistry


def _place(world, elem_sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[elem_sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestBondFormation:
    def test_two_close_slow_atoms_bond(self, cfg):
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2   # 62 SU
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0])    # within bond_formation_factor
        chemistry._form_bonds(w)
        assert len(w.bonds) == 1

    def test_bond_formation_respects_velocity_threshold(self, cfg):
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0], vel=[2000.0, 0.0, 0.0])  # too fast
        chemistry._form_bonds(w)
        assert len(w.bonds) == 0

    def test_bond_formation_respects_max_bonds(self, cfg):
        w = World(cfg)
        # H has max_bonds=1; add a second H that already has a bond
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.1, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0])
        # Manually saturate particle 1
        w.bond_counts[1] = 1
        chemistry._form_bonds(w)
        # Particle 1 already maxed — only one bond can form (0-2 or nothing)
        for b in w.bonds:
            assert b.i != 1 and b.j != 1

    def test_noble_gas_never_bonds(self, cfg):
        w = World(cfg)
        r_eq = 28.0  # He covalent_radius
        _place(w, 'He', [0.0, 0.0, 0.0])
        _place(w, 'He', [r_eq * 1.2, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert len(w.bonds) == 0

    def test_bond_updates_bond_counts(self, cfg):
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert w.bond_counts[0] == 1
        assert w.bond_counts[1] == 1

    def test_no_duplicate_bonds(self, cfg):
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0])
        chemistry._form_bonds(w)
        chemistry._form_bonds(w)   # second call should not add a duplicate
        assert len(w.bonds) == 1

    def test_distant_atoms_do_not_bond(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [1000.0, 0.0, 0.0])  # far away
        chemistry._form_bonds(w)
        assert len(w.bonds) == 0


class TestBondBreaking:
    def test_stretched_bond_breaks(self, cfg):
        w = World(cfg)
        r_eq = 62.0
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 3.0, 0.0, 0.0])   # beyond break_factor=2.8
        bond = Bond(0, 1, r_eq, 50.0, 0.8)
        w.bonds.append(bond)
        w.bond_counts[0] = 1
        w.bond_counts[1] = 1

        chemistry._break_bonds(w)

        assert len(w.bonds) == 0
        assert w.bond_counts[0] == 0
        assert w.bond_counts[1] == 0

    def test_intact_bond_survives(self, cfg):
        w = World(cfg)
        r_eq = 62.0
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq, 0.0, 0.0])   # at equilibrium
        bond = Bond(0, 1, r_eq, 50.0, 0.8)
        w.bonds.append(bond)
        w.bond_counts[0] = 1
        w.bond_counts[1] = 1

        chemistry._break_bonds(w)

        assert len(w.bonds) == 1


class TestFusion:
    def test_two_hydrogen_fuse_to_deuterium(self, cfg):
        cfg.chemistry.coulomb_barrier_scale = 1.0   # very low threshold
        w = World(cfg)

        r_eq = ELEMENTS['H'].covalent_radius * 2
        speed = 5000.0   # very high relative speed → high KE
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])

        chemistry._fuse(w)

        assert w.n == 1
        assert w.total_fusions == 1
        from sim.elements import ELEMENTS_LIST
        product_sym = ELEMENTS_LIST[w.elem_ids[0]].symbol
        assert product_sym == 'D'

    def test_fusion_conserves_momentum(self, cfg):
        cfg.chemistry.coulomb_barrier_scale = 1.0
        w = World(cfg)

        r_eq = ELEMENTS['H'].covalent_radius * 2
        m_h = ELEMENTS['H'].mass
        v = 5000.0
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[ v, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-v, 0.0, 0.0])

        # Equal-mass, opposite velocities → CoM velocity ≈ 0
        chemistry._fuse(w)

        assert w.n == 1
        np.testing.assert_allclose(w.velocities[0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_fusion_disabled_by_flag(self, cfg):
        cfg.chemistry.fusion_enabled = False
        w = World(cfg)

        r_eq = ELEMENTS['H'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])

        chemistry._fuse(w)

        assert w.n == 2   # no fusion
        assert w.total_fusions == 0

    def test_carbon_carbon_fuses_to_magnesium(self, cfg):
        """C+C → Mg-24 is real stellar carbon-burning (Burbidge et al. 1957).
        The Z+A search routes (Z=12, A=24) to Mg-24, which is in the table."""
        cfg.chemistry.coulomb_barrier_scale = 1.0    # trivial barrier in tests
        w = World(cfg)

        r_eq = ELEMENTS['C'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'C', [0.0,         0.0, 0.0], vel=[ speed, 0.0, 0.0])
        _place(w, 'C', [r_eq * 1.5,  0.0, 0.0], vel=[-speed, 0.0, 0.0])

        chemistry._fuse(w)

        assert w.n == 1
        from sim.elements import ELEMENTS_LIST
        assert ELEMENTS_LIST[w.elem_ids[0]].symbol == 'Mg'

    def test_pair_with_no_isotope_match_does_not_fuse(self, cfg):
        """H + Fe → (Z=27, A=57). No element with that (Z, A) in our table,
        and the β⁺ branches don't recover anything either → no fusion."""
        cfg.chemistry.coulomb_barrier_scale = 1.0
        w = World(cfg)
        r = ELEMENTS['H'].covalent_radius + ELEMENTS['Fe'].covalent_radius
        _place(w, 'H',  [0.0,       0.0, 0.0], vel=[ 5000.0, 0.0, 0.0])
        _place(w, 'Fe', [r * 1.5,   0.0, 0.0], vel=[-5000.0, 0.0, 0.0])
        chemistry._fuse(w)
        assert w.n == 2


class TestRadiationKick:
    def test_fusion_kicks_nearby_observer_outward(self, cfg):
        """A particle near a fusion site should receive an outward velocity kick."""
        cfg.chemistry.coulomb_barrier_scale = 1.0
        w = World(cfg)

        r_eq   = ELEMENTS['H'].covalent_radius * 2
        speed  = 5000.0
        # Two H atoms that will fuse
        _place(w, 'H', [0.0, 0.0, 0.0],       vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])
        # Observer sitting to the right, within radiation_radius
        observer_x = 200.0
        _place(w, 'C', [observer_x, 0.0, 0.0], vel=[0.0, 0.0, 0.0])

        v_before = w.velocities[2, 0]   # C is index 2
        chemistry._fuse(w)

        # After fusion n drops to 2 (H+H→D, C stays)
        # Find the C particle (the one with larger mass relative to element)
        assert w.n == 2
        # The surviving non-D particle should have been kicked outward (+x)
        from sim.elements import ELEMENTS_LIST
        c_idx = next(i for i in range(w.n) if ELEMENTS_LIST[w.elem_ids[i]].symbol == 'C')
        assert w.velocities[c_idx, 0] > v_before

    def test_observer_beyond_radius_not_kicked(self, cfg):
        """A particle beyond radiation_radius should not be affected."""
        cfg.chemistry.coulomb_barrier_scale = 1.0
        cfg.chemistry.radiation_radius    = 100.0   # small radius
        w = World(cfg)

        r_eq  = ELEMENTS['H'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'H', [0.0, 0.0, 0.0],        vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])
        # Observer far outside radiation_radius
        _place(w, 'C', [500.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])

        chemistry._fuse(w)

        from sim.elements import ELEMENTS_LIST
        c_idx = next(i for i in range(w.n) if ELEMENTS_LIST[w.elem_ids[i]].symbol == 'C')
        np.testing.assert_allclose(w.velocities[c_idx], [0.0, 0.0, 0.0], atol=1e-10)

    def test_kick_disabled_when_scale_zero(self, cfg):
        """Setting radiation_energy_scale=0 should produce no kick."""
        cfg.chemistry.coulomb_barrier_scale = 1.0
        cfg.chemistry.radiation_energy_scale = 0.0
        w = World(cfg)

        r_eq  = ELEMENTS['H'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'H', [0.0, 0.0, 0.0],        vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])
        _place(w, 'C', [200.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])

        chemistry._fuse(w)

        from sim.elements import ELEMENTS_LIST
        c_idx = next(i for i in range(w.n) if ELEMENTS_LIST[w.elem_ids[i]].symbol == 'C')
        np.testing.assert_allclose(w.velocities[c_idx], [0.0, 0.0, 0.0], atol=1e-10)
