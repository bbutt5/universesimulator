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
        cfg.chemistry.fusion_ke_threshold = 1.0   # very low threshold
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
        cfg.chemistry.fusion_ke_threshold = 1.0
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

    def test_non_fusable_elements_do_not_fuse(self, cfg):
        cfg.chemistry.fusion_ke_threshold = 1.0
        w = World(cfg)

        r_eq = ELEMENTS['C'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'C', [0.0, 0.0, 0.0], vel=[ speed, 0.0, 0.0])
        _place(w, 'C', [r_eq * 1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])

        chemistry._fuse(w)

        assert w.n == 2   # C has can_fuse=False
