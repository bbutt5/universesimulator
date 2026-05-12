"""Tests for bond-order semantics in formation, breaking, reactions.

Real chemistry distinguishes single (X–Y), double (X=Y), and triple
(X≡Y) bonds. The bond order determines:
  - how much valence each atom uses (bond_counts increments)
  - the dissociation energy (D_e scales by order-dependent factor)
  - the Morse force depth (proportional to D_e)

Reference ratios (CRC Handbook bond-dissociation energies):
  D(C=C) / D(C-C) ≈ 1.77
  D(C≡C) / D(C-C) ≈ 2.42
  D(N≡N) / D(N-N) ≈ 5.6  (atypical; N-N single is weak)
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS
from sim import chemistry, reactions
from sim.particle import Bond


def _place(world, sym, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(
        ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float),
    )


class TestBondOrderFormation:
    def test_h_h_is_single_bond_valence_limited(self, cfg):
        """H has max_bonds=1 each; the H-H bond can only be single, regardless
        of geometry."""
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert len(w.bonds) == 1
        assert w.bonds[0].order == 1
        assert w.bond_counts[0] == 1
        assert w.bond_counts[1] == 1

    def test_o_o_can_form_double_bond(self, cfg):
        """O has max_bonds=2 each; O=O (dioxygen) should form with order 2,
        consuming both atoms' valence in one bond."""
        w = World(cfg)
        r_eq = ELEMENTS['O'].covalent_radius * 2
        _place(w, 'O', [0.0, 0.0, 0.0])
        _place(w, 'O', [r_eq * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert len(w.bonds) == 1
        assert w.bonds[0].order == 2
        # Each atom uses 2 valence slots
        assert w.bond_counts[0] == 2
        assert w.bond_counts[1] == 2

    def test_n_n_can_form_triple_bond(self, cfg):
        """N has max_bonds=3; N≡N (dinitrogen) should form with order 3."""
        w = World(cfg)
        r_eq = ELEMENTS['N'].covalent_radius * 2
        _place(w, 'N', [0.0, 0.0, 0.0])
        _place(w, 'N', [r_eq * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert len(w.bonds) == 1
        assert w.bonds[0].order == 3
        assert w.bond_counts[0] == 3
        assert w.bond_counts[1] == 3

    def test_c_o_forms_double_bond(self, cfg):
        """C has 4 free, O has 2 free — bond order = min(4, 2, 3) = 2 (C=O)."""
        w = World(cfg)
        r_eq = ELEMENTS['C'].covalent_radius + ELEMENTS['O'].covalent_radius
        _place(w, 'C', [0.0, 0.0, 0.0])
        _place(w, 'O', [r_eq * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w)
        assert len(w.bonds) == 1
        assert w.bonds[0].order == 2

    def test_double_bond_stronger_than_single(self, cfg):
        """The Morse D_e of a double bond should be ~1.8× the single-bond D_e
        for the same pair, matching CRC handbook ratios."""
        # Single: H-H (D_e_single = E_scale × D_HH_kjmol × 1.0)
        w_h = World(cfg)
        r_h = ELEMENTS['H'].covalent_radius * 2
        _place(w_h, 'H', [0.0, 0.0, 0.0])
        _place(w_h, 'H', [r_h * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w_h)
        d_e_single = w_h.bonds[0].D_e

        # Double: O=O (D_e_double = E_scale × D_OO_kjmol × 1.8)
        w_o = World(cfg)
        r_o = ELEMENTS['O'].covalent_radius * 2
        _place(w_o, 'O', [0.0, 0.0, 0.0])
        _place(w_o, 'O', [r_o * 1.1, 0.0, 0.0])
        chemistry._form_bonds(w_o)
        d_e_double = w_o.bonds[0].D_e

        # Compute the underlying single-bond D_e for O-O for comparison
        from sim.elements import pauling_bond_energy_kjmol
        E_scale = cfg.chemistry.bond_energy_scale
        d_oo_single_baseline = E_scale * pauling_bond_energy_kjmol(
            ELEMENTS['O'], ELEMENTS['O'],
        )
        # The O=O bond should be 1.8× its underlying single-bond D
        assert d_e_double / d_oo_single_baseline == pytest.approx(1.8, rel=0.01)


class TestBondOrderBreaking:
    def test_double_bond_free_two_valence_slots(self, cfg):
        """When O=O breaks (stretched too far), each atom should regain both
        valence slots — not just one."""
        w = World(cfg)
        r_eq = ELEMENTS['O'].covalent_radius * 2
        _place(w, 'O', [0.0, 0.0, 0.0])
        # Place far apart so the bond will break
        _place(w, 'O', [r_eq * 5.0, 0.0, 0.0])
        # Manually create a double bond
        bond = Bond(0, 1, r_eq, 100.0, 0.8, order=2)
        w.bonds.append(bond)
        w.bond_counts[0] = 2
        w.bond_counts[1] = 2

        chemistry._break_bonds(w)

        assert len(w.bonds) == 0
        assert w.bond_counts[0] == 0
        assert w.bond_counts[1] == 0


class TestBondOrderReactions:
    def test_swap_preserves_bond_order(self, cfg):
        """When a reaction swap fires, the new bond has the same order as
        the old. So if the old was double, k must have ≥2 free valence."""
        cfg.reactions.activation_energy_scale = 1.0   # P_swap ≈ 1
        w = World(cfg)
        # Pre-existing O=O bond (order 2)
        _place(w, 'O', [0.0,  0.0, 0.0])
        _place(w, 'O', [60.0, 0.0, 0.0])
        # A nearby Si with full free valence (4) which CAN accept a double bond
        _place(w, 'Si', [80.0, 0.0, 0.0])
        bond = Bond(0, 1, 132.0, 50.0, 0.8, order=2)
        w.bonds.append(bond)
        w.bond_counts[0] = 2
        w.bond_counts[1] = 2

        reactions.update(w)

        # If the swap fires, the new bond should also be order 2
        if w.total_reactions > 0:
            assert w.bonds[0].order == 2

    def test_swap_blocked_by_insufficient_valence(self, cfg):
        """A double bond cannot swap to a partner with only 1 free valence
        (like H, max_bonds=1)."""
        cfg.reactions.activation_energy_scale = 1.0
        w = World(cfg)
        _place(w, 'O', [0.0,  0.0, 0.0])
        _place(w, 'O', [60.0, 0.0, 0.0])
        _place(w, 'H', [80.0, 0.0, 0.0])      # only 1 valence
        bond = Bond(0, 1, 132.0, 50.0, 0.8, order=2)
        w.bonds.append(bond)
        w.bond_counts[0] = 2
        w.bond_counts[1] = 2

        reactions.update(w)

        # H couldn't accept a double bond → no swap
        assert w.total_reactions == 0
