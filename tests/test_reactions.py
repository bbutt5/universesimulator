"""Tests for sim/reactions.py — electronegativity-driven bond swaps.

The canonical example is halide displacement (HCl + F → HF + Cl).
F has higher electronegativity than Cl, so the Pauling formula gives
D(H-F) > D(H-Cl); the H atom switches partners and energy is released.

A physicist reading these tests can verify against the chemistry
handbook: the bond-energy ordering D(H-F) > D(H-Cl) is well-known and
the displacement reaction is in every introductory chem textbook.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS, pauling_bond_energy_kjmol
from sim.particle import Bond
from sim import reactions


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(
        ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float),
    )


def _bond(world, i: int, j: int, D_kjmol: float):
    """Add a Bond between i and j with realistic Morse parameters."""
    elem_i = ELEMENTS[world.element_of(i).symbol]
    elem_j = ELEMENTS[world.element_of(j).symbol]
    r_eq   = elem_i.covalent_radius + elem_j.covalent_radius
    # We don't care about the exact spring constant here; tests focus on
    # whether the swap fires and conserves momentum.
    world.bonds.append(Bond(i, j, r_eq, D_kjmol * 0.184, 0.8))
    world.bond_counts[i] += 1
    world.bond_counts[j] += 1


def _favourable_reactions_cfg(cfg):
    """Drop the activation-energy scale so any positive ΔE swap fires."""
    cfg.reactions.activation_energy_scale = 1.0   # P_swap ≈ 1 for any ΔE > 0
    return cfg


class TestHalideDisplacement:
    """HCl + F → HF + Cl is the textbook electronegativity-displacement
    reaction. F is more electronegative than Cl (3.98 vs 3.16), so the
    H atom binds more strongly to F. Real ΔH ≈ −180 kJ/mol."""

    def test_h_f_more_exothermic_than_h_cl(self):
        """Sanity check: confirm the Pauling formula agrees with the chemistry
        handbook that D(H-F) > D(H-Cl). If this fails the data table is wrong."""
        d_hf  = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['F'])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        assert d_hf > d_hcl

    def test_hcl_plus_f_becomes_hf_plus_cl(self, cfg):
        """Place H-Cl with F nearby (free valence). The reaction should swap
        H's partner from Cl to F."""
        _favourable_reactions_cfg(cfg)
        w = World(cfg)
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        _bond(w, h, cl, d_hcl)

        reactions.update(w)

        # The H-Cl bond is replaced by H-F
        assert w.total_reactions == 1
        assert len(w.bonds) == 1
        b = w.bonds[0]
        assert {b.i, b.j} == {h, f}
        # And the bond counts reflect the swap
        assert w.bond_counts[h]  == 1
        assert w.bond_counts[cl] == 0      # Cl released
        assert w.bond_counts[f]  == 1


class TestNoSwapWhenUnfavourable:
    def test_no_swap_when_alternative_weaker(self, cfg):
        """If the alternative partner offers a *weaker* bond, no swap."""
        _favourable_reactions_cfg(cfg)
        w = World(cfg)
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0])      # already bonded
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])      # weaker alternative
        d_hf = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['F'])
        _bond(w, h, f, d_hf)

        reactions.update(w)

        assert w.total_reactions == 0
        # Original bond intact
        assert len(w.bonds) == 1
        assert {w.bonds[0].i, w.bonds[0].j} == {h, f}

    def test_no_swap_when_partner_at_max_valence(self, cfg):
        """If the candidate partner has no free valence, swap is blocked."""
        _favourable_reactions_cfg(cfg)
        w = World(cfg)
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        _bond(w, h, cl, d_hcl)
        # Saturate F's valence (max_bonds=1 for halogen)
        w.bond_counts[f] = ELEMENTS['F'].max_bonds

        reactions.update(w)

        # F is saturated, so the swap can't proceed
        assert w.total_reactions == 0

    def test_no_swap_with_noble_gas(self, cfg):
        """Noble gases (max_bonds=0) can never participate as the new partner."""
        _favourable_reactions_cfg(cfg)
        w = World(cfg)
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])
        ne = _place(w, 'Ne', [ 70.0, 0.0, 0.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        _bond(w, h, cl, d_hcl)

        reactions.update(w)

        assert w.total_reactions == 0
        # Ne never gets a bond
        assert w.bond_counts[ne] == 0

    def test_no_reaction_radius_zero_disables_module(self, cfg):
        """Setting reaction_radius=0 should disable reactions entirely."""
        _favourable_reactions_cfg(cfg)
        cfg.reactions.reaction_radius = 0.0
        w = World(cfg)
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        _bond(w, h, cl, d_hcl)

        reactions.update(w)

        assert w.total_reactions == 0


class TestMomentumConservation:
    def test_total_momentum_unchanged_by_reaction(self, cfg):
        """The heat release kicks j and k apart; total momentum must stay put."""
        _favourable_reactions_cfg(cfg)
        w = World(cfg)
        # Add slight initial velocities so total momentum is non-zero
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0], vel=[1.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0], vel=[0.0, 2.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0], vel=[0.0, 0.0, 3.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        _bond(w, h, cl, d_hcl)

        # Total momentum before
        p_before = (w.masses[:3, None] * w.velocities[:3]).sum(axis=0)

        reactions.update(w)

        # Total momentum after
        p_after = (w.masses[:3, None] * w.velocities[:3]).sum(axis=0)

        assert w.total_reactions == 1
        np.testing.assert_allclose(p_after, p_before, rtol=1e-9, atol=1e-9)


class TestHeatRelease:
    def test_kinetic_energy_increases_by_delta_e(self, cfg):
        """The total KE of the j-k pair should rise by ΔE × heat_release_scale."""
        _favourable_reactions_cfg(cfg)
        cfg.reactions.heat_release_scale = 1.0   # 1 kJ/mol → 1 sim KE

        w = World(cfg)
        # Start everyone at rest so any KE gain is purely from the reaction
        h  = _place(w, 'H',  [  0.0, 0.0, 0.0])
        cl = _place(w, 'Cl', [100.0, 0.0, 0.0])
        f  = _place(w, 'F',  [ 70.0, 0.0, 0.0])
        d_hcl = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['Cl'])
        d_hf  = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['F'])
        delta_e = d_hf - d_hcl
        _bond(w, h, cl, d_hcl)

        ke_before = 0.5 * np.sum(w.masses[:3] * np.einsum('ij,ij->i', w.velocities[:3], w.velocities[:3]))

        reactions.update(w)

        ke_after = 0.5 * np.sum(w.masses[:3] * np.einsum('ij,ij->i', w.velocities[:3], w.velocities[:3]))
        delta_ke = ke_after - ke_before

        # Heat goes to the (j, k) = (cl, f) pair; KE gain should equal ΔE
        assert delta_ke == pytest.approx(delta_e, rel=1e-6)


class TestSafety:
    def test_no_bond_no_reaction(self, cfg):
        """With no bonds in the world, reactions.update is a no-op."""
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'F', [70.0, 0.0, 0.0])
        reactions.update(w)
        assert w.total_reactions == 0

    def test_no_reactions_cfg_no_op(self, cfg):
        """Missing reactions config block → silent no-op."""
        del cfg.reactions
        w = World(cfg)
        h = _place(w, 'H', [0.0, 0.0, 0.0])
        f = _place(w, 'F', [70.0, 0.0, 0.0])
        d_hf = pauling_bond_energy_kjmol(ELEMENTS['H'], ELEMENTS['F'])
        _bond(w, h, f, d_hf)
        reactions.update(w)
        assert w.total_reactions == 0
