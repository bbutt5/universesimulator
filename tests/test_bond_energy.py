"""Tests for sim/elements.pauling_bond_energy_kjmol — Pauling bond-energy formula.

A physicist can compare each expected value to a chemistry handbook:
    D(A-B) = √(D_AA · D_BB) + 96 · (χ_A − χ_B)²        [kJ/mol]
(Pauling 1932; Atkins, Physical Chemistry §15.6)
"""

from __future__ import annotations
import math
import pytest

from sim.elements import ELEMENTS, pauling_bond_energy_kjmol


def _pauling(a, b):
    """Reference implementation of the formula — keeps test independent of the impl."""
    d_aa = a.bond_dissociation_self
    d_bb = b.bond_dissociation_self
    if d_aa <= 0 or d_bb <= 0:
        return 0.0
    return math.sqrt(d_aa * d_bb) + 96.0 * (a.electronegativity - b.electronegativity) ** 2


class TestPaulingFormula:
    def test_homonuclear_returns_self_bond_energy(self):
        """For X-X, Δχ = 0 and √(D_XX·D_XX) = D_XX."""
        H = ELEMENTS['H']
        assert pauling_bond_energy_kjmol(H, H) == pytest.approx(H.bond_dissociation_self)

    def test_heteronuclear_h_o_realistic(self):
        """H-O bond energy is ~459 kJ/mol experimentally (single bond).
        Pauling estimate from H (436, χ=2.20) and O (146, χ=3.44):
            √(436·146) + 96·1.24² ≈ 252.4 + 147.6 ≈ 400 kJ/mol
        (Pauling's formula systematically underestimates O-H; that's known.
        The order of magnitude is right and the formula is the textbook one.)"""
        H, O = ELEMENTS['H'], ELEMENTS['O']
        val = pauling_bond_energy_kjmol(H, O)
        assert 350.0 < val < 500.0    # real H-O ≈ 459 kJ/mol

    def test_polar_bond_stronger_than_geometric_mean(self):
        """Ionic-resonance term is strictly positive when electronegativities
        differ — heteronuclear bond is *always* ≥ geometric mean of the
        homonuclear bonds. That's the chemistry behind why polar bonds exist."""
        H, F = ELEMENTS['H'], ELEMENTS['F']
        d_hh, d_ff = H.bond_dissociation_self, F.bond_dissociation_self
        geo_mean   = math.sqrt(d_hh * d_ff)
        d_hf       = pauling_bond_energy_kjmol(H, F)
        assert d_hf > geo_mean

    def test_h_f_is_much_stronger_than_h_h(self):
        """H-F bond (Δχ = 1.78) is famously very strong.
        Real H-F ≈ 565 kJ/mol; Pauling estimate ≈ √(436·158) + 96·(3.98-2.20)²
        ≈ 262 + 304 ≈ 566 kJ/mol. Should be > H-H (436)."""
        H, F = ELEMENTS['H'], ELEMENTS['F']
        d_hf = pauling_bond_energy_kjmol(H, F)
        assert d_hf > ELEMENTS['H'].bond_dissociation_self

    def test_no_bond_with_noble_gas(self):
        """He, Ne, Ar have D(X-X) = 0 → formula returns 0 with any partner."""
        H = ELEMENTS['H']
        for noble in ('He', 'Ne', 'Ar'):
            assert pauling_bond_energy_kjmol(H, ELEMENTS[noble]) == 0.0

    def test_symmetric_a_b_equals_b_a(self):
        """D(A-B) must equal D(B-A) by the formula's structure."""
        for a, b in [('H', 'O'), ('C', 'N'), ('Fe', 'O'), ('Si', 'C')]:
            v_ab = pauling_bond_energy_kjmol(ELEMENTS[a], ELEMENTS[b])
            v_ba = pauling_bond_energy_kjmol(ELEMENTS[b], ELEMENTS[a])
            assert v_ab == pytest.approx(v_ba)


class TestBondEnergyOrdering:
    """Real-physics ordering: stronger bonds match higher electronegativity
    differences with high self-bond partners."""

    def test_c_c_stronger_than_c_n(self):
        """Real D(C-C) = 348 vs D(C-N) ≈ 305 kJ/mol — N has weaker self-bond."""
        cc = pauling_bond_energy_kjmol(ELEMENTS['C'], ELEMENTS['C'])
        cn = pauling_bond_energy_kjmol(ELEMENTS['C'], ELEMENTS['N'])
        assert cc > cn

    def test_n_n_weaker_than_c_n(self):
        """Δχ for C-N (0.49) pushes C-N above N-N's weak geometric mean."""
        nn = pauling_bond_energy_kjmol(ELEMENTS['N'], ELEMENTS['N'])
        cn = pauling_bond_energy_kjmol(ELEMENTS['C'], ELEMENTS['N'])
        assert cn > nn

    def test_extreme_polar_bonds_are_strong(self):
        """Cs-F-like extremes don't exist in our element set, but K-F should
        be very strong (Δχ ≈ 3.16). Use Na-F (Δχ = 3.05)."""
        na_f = pauling_bond_energy_kjmol(ELEMENTS['Na'], ELEMENTS['F'])
        # √(75·158) + 96·3.05² ≈ 109 + 893 ≈ 1002 kJ/mol
        assert na_f > 800.0    # genuinely ionic bond
