"""Tests for Phase-3 composition tracking.

The composition array on World stores, per particle, the mass of each
element it has absorbed (or has been since birth). A pristine atom has
all its mass in its own element entry; an accreted body has a mix
summed through history; a fusion product is a fresh single-element
slate.

Sanity invariants verified here:
  * composition[i].sum() ≈ masses[i]  (mass conservation)
  * accretion sums the two reactant compositions
  * fusion resets to a pure-product composition
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS, SYMBOL_TO_ID, ELEMENTS_LIST
from sim import accretion, chemistry


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(
        ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float),
    )


def _comp_sym(world, i: int, sym: str) -> float:
    """Mass of element sym held by particle i."""
    return float(world.composition[i, SYMBOL_TO_ID[sym]])


class TestCompositionInitialisation:
    def test_fresh_atom_is_pure(self, cfg):
        w = World(cfg)
        idx = _place(w, 'H', [0, 0, 0])
        # All mass is in H
        assert _comp_sym(w, idx, 'H') == pytest.approx(ELEMENTS['H'].mass)
        # Sum equals mass
        assert w.composition[idx].sum() == pytest.approx(w.masses[idx])
        # No other element present
        nonzero = np.nonzero(w.composition[idx])[0]
        assert nonzero.tolist() == [SYMBOL_TO_ID['H']]

    def test_multiple_atoms_independent(self, cfg):
        w = World(cfg)
        _place(w, 'H',  [0,    0, 0])
        _place(w, 'Fe', [100,  0, 0])
        _place(w, 'O',  [200,  0, 0])
        # Each particle's composition is its own element only
        assert _comp_sym(w, 0, 'H')  == pytest.approx(ELEMENTS['H'].mass)
        assert _comp_sym(w, 0, 'Fe') == 0.0
        assert _comp_sym(w, 1, 'Fe') == pytest.approx(ELEMENTS['Fe'].mass)
        assert _comp_sym(w, 1, 'H')  == 0.0


class TestCompositionAccretion:
    def test_two_bodies_compositions_sum(self, cfg):
        """When Fe + C accrete, the merged body's composition contains both
        elements' original masses."""
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'C',  [50.0, 0.0, 0.0])
        accretion.update(w)
        assert w.n == 1
        # Combined: both Fe and C present with their original masses
        assert _comp_sym(w, 0, 'Fe') == pytest.approx(ELEMENTS['Fe'].mass)
        assert _comp_sym(w, 0, 'C')  == pytest.approx(ELEMENTS['C'].mass)
        # Sum equals total mass
        assert w.composition[0].sum() == pytest.approx(w.masses[0])

    def test_repeated_accretion_accumulates(self, cfg):
        """Successive merges build up the composition record correctly."""
        w = World(cfg)
        # Three Fe particles
        for x in [0.0, 50.0, 100.0]:
            _place(w, 'Fe', [x, 0.0, 0.0])
        accretion.update(w)   # likely merges two
        accretion.update(w)   # may merge the resulting body with the third
        # All Fe mass is preserved in the final body's composition
        if w.n == 1:
            assert _comp_sym(w, 0, 'Fe') == pytest.approx(3 * ELEMENTS['Fe'].mass)
            assert w.composition[0].sum() == pytest.approx(w.masses[0])

    def test_mixed_accretion_preserves_identities(self, cfg):
        """A planet built from Fe + Si + O retains all three in its
        composition, not just the dominant element."""
        w = World(cfg)
        # Heavier elements drift together
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'Si', [40.0, 0.0, 0.0])
        _place(w, 'O',  [80.0, 0.0, 0.0])
        for _ in range(3):
            accretion.update(w)
        if w.n == 1:
            for sym in ('Fe', 'Si', 'O'):
                assert _comp_sym(w, 0, sym) > 0.0


class TestCompositionFusion:
    def test_fusion_product_is_pure(self, cfg):
        """A D nucleus formed from H+H fusion has composition entirely in
        D — not lingering hydrogen-record from the reactants. Fusion is a
        nuclear rebirth, not a chemical merger."""
        cfg.chemistry.coulomb_barrier_scale = 1.0  # easy fusion in tests
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        speed = 5000.0
        _place(w, 'H', [0.0,      0.0, 0.0], vel=[ speed, 0.0, 0.0])
        _place(w, 'H', [r_eq*1.5, 0.0, 0.0], vel=[-speed, 0.0, 0.0])
        chemistry._fuse(w)
        # Expect H+H → D with pure-D composition
        assert w.n == 1
        assert ELEMENTS_LIST[w.elem_ids[0]].symbol == 'D'
        assert _comp_sym(w, 0, 'D') == pytest.approx(w.masses[0])
        assert _comp_sym(w, 0, 'H') == 0.0  # fusion erased the H history


class TestCompositionMassConservation:
    def test_composition_sums_match_masses(self, cfg):
        """For every live particle in any state, composition[i].sum() must
        equal masses[i] within floating-point tolerance."""
        w = World(cfg)
        for _ in range(50):
            w.step(0.02)
        n = w.n
        if n > 0:
            sums = w.composition[:n].sum(axis=1)
            np.testing.assert_allclose(sums, w.masses[:n], rtol=1e-6, atol=1e-6)


class TestCompositionRemovePreserves:
    def test_remove_swaps_composition(self, cfg):
        """remove_particle uses swap-with-last; composition must move with
        the swapped row so the surviving particles keep their identity."""
        w = World(cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0])
        _place(w, 'C',  [100., 0.0, 0.0])
        _place(w, 'O',  [200., 0.0, 0.0])
        # Remove the middle particle (C)
        w.remove_particle(1)
        assert w.n == 2
        # The element previously at index 2 (O) now sits at index 1
        symbols_now = {ELEMENTS_LIST[w.elem_ids[i]].symbol for i in range(w.n)}
        assert symbols_now == {'Fe', 'O'}
        # And each particle's composition matches its current element
        for i in range(w.n):
            sym = ELEMENTS_LIST[w.elem_ids[i]].symbol
            assert _comp_sym(w, i, sym) == pytest.approx(ELEMENTS[sym].mass)
