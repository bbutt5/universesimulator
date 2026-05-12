"""Tests for sim/molecules.py — connected-component recognition + Hill formulae.

The simulator's bond layer already produces bonded clusters; this module
tags each cluster with a chemistry-handbook formula (Hill ordered) and a
common name if known. A chemist should look at the tests below and find
them unambiguous: each canonical molecule (H₂, H₂O, CO₂, CH₄, NH₃, O₂, etc.)
gets the textbook formula string.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond
from sim.molecules import (
    identify_molecules, molecule_counts, molecule_of, Molecule,
    _formula_from_counts,
)


def _place(world, sym: str, pos):
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.zeros(3))


def _bond_pair(world, i: int, j: int):
    """Add a Bond between particles i and j with sensible default params."""
    elem_i = ELEMENTS[world.element_of(i).symbol]
    elem_j = ELEMENTS[world.element_of(j).symbol]
    r_eq = elem_i.covalent_radius + elem_j.covalent_radius
    world.bonds.append(Bond(i, j, r_eq, 50.0, 0.8))
    world.bond_counts[i] += 1
    world.bond_counts[j] += 1


class TestHillFormula:
    """Hill ordering: C first, H second, then alphabetical."""

    def test_water_is_h2o(self):
        assert _formula_from_counts({'H': 2, 'O': 1}) == 'H2O'

    def test_methane_is_ch4(self):
        assert _formula_from_counts({'H': 4, 'C': 1}) == 'CH4'

    def test_ammonia_is_nh3(self):
        assert _formula_from_counts({'H': 3, 'N': 1}) == 'NH3'

    def test_carbon_dioxide_is_co2(self):
        assert _formula_from_counts({'C': 1, 'O': 2}) == 'CO2'

    def test_ethanol_is_c2h6o(self):
        """Hill: C first, H second, then O — ethanol = C2H6O."""
        assert _formula_from_counts({'C': 2, 'H': 6, 'O': 1}) == 'C2H6O'

    def test_singletons_have_no_subscript(self):
        assert _formula_from_counts({'H': 1, 'F': 1}) == 'HF'
        assert _formula_from_counts({'C': 1, 'O': 1}) == 'CO'


class TestCanonicalMolecules:
    """Each canonical species gets the right formula AND name."""

    def test_h2_is_dihydrogen(self, cfg):
        w = World(cfg)
        i = _place(w, 'H', [0, 0, 0])
        j = _place(w, 'H', [62, 0, 0])
        _bond_pair(w, i, j)
        mols = identify_molecules(w)
        assert len(mols) == 1
        assert mols[0].formula == 'H2'
        assert mols[0].name == 'dihydrogen'
        assert mols[0].size == 2

    def test_h2o_is_water(self, cfg):
        w = World(cfg)
        o  = _place(w, 'O', [0,   0, 0])
        h1 = _place(w, 'H', [60,  0, 0])
        h2 = _place(w, 'H', [-60, 0, 0])
        _bond_pair(w, o, h1)
        _bond_pair(w, o, h2)
        mols = identify_molecules(w)
        assert len(mols) == 1
        assert mols[0].formula == 'H2O'
        assert mols[0].name == 'water'

    def test_co2_is_carbon_dioxide(self, cfg):
        w = World(cfg)
        c  = _place(w, 'C', [0,   0, 0])
        o1 = _place(w, 'O', [80,  0, 0])
        o2 = _place(w, 'O', [-80, 0, 0])
        _bond_pair(w, c, o1)
        _bond_pair(w, c, o2)
        mols = identify_molecules(w)
        assert mols[0].formula == 'CO2'
        assert mols[0].name == 'carbon dioxide'

    def test_ch4_is_methane(self, cfg):
        w = World(cfg)
        c = _place(w, 'C', [0, 0, 0])
        for k in range(4):
            h = _place(w, 'H', [60 * (k + 1), 0, 0])
            _bond_pair(w, c, h)
        mols = identify_molecules(w)
        assert mols[0].formula == 'CH4'
        assert mols[0].name == 'methane'

    def test_nh3_is_ammonia(self, cfg):
        w = World(cfg)
        n = _place(w, 'N', [0, 0, 0])
        for k in range(3):
            h = _place(w, 'H', [60 * (k + 1), 0, 0])
            _bond_pair(w, n, h)
        mols = identify_molecules(w)
        assert mols[0].formula == 'NH3'
        assert mols[0].name == 'ammonia'

    def test_o2_is_dioxygen(self, cfg):
        w = World(cfg)
        a = _place(w, 'O', [0,  0, 0])
        b = _place(w, 'O', [80, 0, 0])
        _bond_pair(w, a, b)
        mols = identify_molecules(w)
        assert mols[0].formula == 'O2'
        assert mols[0].name == 'dioxygen'

    def test_n2_is_dinitrogen(self, cfg):
        w = World(cfg)
        a = _place(w, 'N', [0,  0, 0])
        b = _place(w, 'N', [70, 0, 0])
        _bond_pair(w, a, b)
        mols = identify_molecules(w)
        assert mols[0].formula == 'N2'
        assert mols[0].name == 'dinitrogen'

    def test_hf_is_hydrogen_fluoride(self, cfg):
        w = World(cfg)
        h = _place(w, 'H', [0,  0, 0])
        f = _place(w, 'F', [60, 0, 0])
        _bond_pair(w, h, f)
        mols = identify_molecules(w)
        assert mols[0].formula == 'HF'
        assert mols[0].name == 'hydrogen fluoride'


class TestComponentSeparation:
    def test_two_separate_h2_tagged_separately(self, cfg):
        w = World(cfg)
        # Cluster A
        a1 = _place(w, 'H', [0,   0, 0]);   a2 = _place(w, 'H', [60,  0, 0])
        _bond_pair(w, a1, a2)
        # Cluster B (far away)
        b1 = _place(w, 'H', [500, 0, 0]);   b2 = _place(w, 'H', [560, 0, 0])
        _bond_pair(w, b1, b2)
        mols = identify_molecules(w)
        assert len(mols) == 2
        assert all(m.formula == 'H2' for m in mols)

    def test_lone_atoms_not_returned(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0,   0, 0])
        _place(w, 'O', [500, 0, 0])
        assert identify_molecules(w) == []

    def test_empty_world(self, cfg):
        assert identify_molecules(World(cfg)) == []

    def test_single_bonded_h2o_within_larger_cloud(self, cfg):
        """A bonded H₂O plus several lone H atoms scattered around →
        exactly one molecule (H₂O); lone atoms not reported."""
        w = World(cfg)
        # H₂O
        o  = _place(w, 'O', [0,   0, 0])
        h1 = _place(w, 'H', [60,  0, 0])
        h2 = _place(w, 'H', [-60, 0, 0])
        _bond_pair(w, o, h1)
        _bond_pair(w, o, h2)
        # Lone atoms far away
        for k in range(5):
            _place(w, 'H', [1000 + 200 * k, 0, 0])
        mols = identify_molecules(w)
        assert len(mols) == 1
        assert mols[0].formula == 'H2O'


class TestMoleculeCounts:
    def test_hud_census(self, cfg):
        w = World(cfg)
        # 3 × H₂
        for k in range(3):
            i = _place(w, 'H', [k * 300,      0, 0])
            j = _place(w, 'H', [k * 300 + 60, 0, 0])
            _bond_pair(w, i, j)
        # 1 × H₂O
        o  = _place(w, 'O', [0,   500, 0])
        h1 = _place(w, 'H', [60,  500, 0])
        h2 = _place(w, 'H', [-60, 500, 0])
        _bond_pair(w, o, h1)
        _bond_pair(w, o, h2)
        counts = molecule_counts(w)
        assert counts.get('H2') == 3
        assert counts.get('H2O') == 1


class TestMoleculeOf:
    def test_returns_containing_molecule(self, cfg):
        w = World(cfg)
        o  = _place(w, 'O', [0,   0, 0])
        h1 = _place(w, 'H', [60,  0, 0])
        h2 = _place(w, 'H', [-60, 0, 0])
        _bond_pair(w, o, h1)
        _bond_pair(w, o, h2)
        mol = molecule_of(w, h1)
        assert mol is not None
        assert mol.formula == 'H2O'

    def test_returns_none_for_lone_atom(self, cfg):
        w = World(cfg)
        lone = _place(w, 'H', [0, 0, 0])
        assert molecule_of(w, lone) is None
