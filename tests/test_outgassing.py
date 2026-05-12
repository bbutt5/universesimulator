"""Tests for sim/outgassing.py — Jeans-escape style volatile release.

Hot accreted bodies should shed light elements (H, He) back into the
surrounding gas phase, mirroring real stellar winds and atmospheric
escape. Mass is strictly conserved: emitted-atom mass = body-mass-loss."""

from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from sim.world import World
from sim.elements import ELEMENTS, ELEMENTS_LIST, SYMBOL_TO_ID
from sim import outgassing


def _viewer_cfg(cfg):
    """Outgassing reads thresholds from renderer config; supply them."""
    cfg.renderer = SimpleNamespace(
        planet_mass_threshold     = 10.0,
        star_mass_threshold       = 100.0,
        hot_temperature_threshold = 500000.0,
        reference_temperature_K   = 20000.0,
    )
    return cfg


def _make_hot_body(world, sym, ratio, vel=None):
    """Place a single atom and inflate it to a body with mass = ratio × atom."""
    pos = np.zeros(3)
    if vel is None:
        vel = np.zeros(3)
    idx = world.add_particle(ELEMENTS[sym], pos, vel)
    elem = ELEMENTS[sym]
    world.masses[idx] = elem.mass * ratio
    world.composition[idx, :] = 0.0
    world.composition[idx, SYMBOL_TO_ID[sym]] = elem.mass * ratio
    return idx


class TestNoOpCases:
    def test_no_config_block(self, cfg):
        _viewer_cfg(cfg)
        del cfg.outgassing
        w = World(cfg)
        outgassing.update(w)              # silent no-op
        assert w.total_outgassing == 0

    def test_zero_rate_disables(self, cfg):
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 0.0
        w = World(cfg)
        _make_hot_body(w, 'Fe', ratio=200.0)
        # Add hydrogen mass to the body's composition so there's something to lose
        w.composition[0, SYMBOL_TO_ID['H']] = 100.0 * ELEMENTS['H'].mass
        outgassing.update(w)
        assert w.total_outgassing == 0

    def test_cold_body_does_not_outgas(self, cfg):
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0           # max probability
        cfg.outgassing.temperature_threshold_K = 1e7  # impossibly high
        w = World(cfg)
        _make_hot_body(w, 'Fe', ratio=200.0)
        w.composition[0, SYMBOL_TO_ID['H']] = 100.0 * ELEMENTS['H'].mass
        outgassing.update(w)
        assert w.total_outgassing == 0

    def test_atom_below_planet_threshold_does_not_outgas(self, cfg):
        """A solo Fe atom (no accumulated mass) isn't a body — nothing happens."""
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0
        cfg.outgassing.temperature_threshold_K = 100.0
        w = World(cfg)
        w.add_particle(ELEMENTS['Fe'], np.zeros(3), np.zeros(3))
        outgassing.update(w)
        assert w.total_outgassing == 0


class TestOutgassingFires:
    def test_hot_body_with_hydrogen_releases_hydrogen(self, cfg):
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0          # max probability
        cfg.outgassing.temperature_threshold_K = 100.0
        w = World(cfg)
        # Big stellar-mass body with H in its composition
        _make_hot_body(w, 'Fe', ratio=500.0)
        # Give it a substantial H stock
        h_initial = 1000.0 * ELEMENTS['H'].mass
        w.composition[0, SYMBOL_TO_ID['H']] = h_initial
        w.masses[0] += h_initial    # keep composition.sum() == mass invariant

        outgassing.update(w)

        assert w.total_outgassing == 1
        assert w.n == 2
        # The new particle is hydrogen
        assert ELEMENTS_LIST[w.elem_ids[1]].symbol == 'H'
        # Body's H stock dropped by one H atom mass
        delta = h_initial - w.composition[0, SYMBOL_TO_ID['H']]
        assert delta == pytest.approx(ELEMENTS['H'].mass)


class TestMassConservation:
    def test_total_mass_conserved_across_emission(self, cfg):
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0
        cfg.outgassing.temperature_threshold_K = 100.0
        w = World(cfg)
        _make_hot_body(w, 'Fe', ratio=500.0)
        h_initial = 1000.0 * ELEMENTS['H'].mass
        w.composition[0, SYMBOL_TO_ID['H']] = h_initial
        w.masses[0] += h_initial

        total_before = w.masses[:w.n].sum()
        outgassing.update(w)
        total_after  = w.masses[:w.n].sum()
        assert total_after == pytest.approx(total_before, rel=1e-12)

    def test_composition_invariant_holds_after_emission(self, cfg):
        """After outgassing, composition[i].sum() must still equal masses[i]
        for every live particle (the standing project invariant)."""
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0
        cfg.outgassing.temperature_threshold_K = 100.0
        w = World(cfg)
        _make_hot_body(w, 'Fe', ratio=500.0)
        h_initial = 1000.0 * ELEMENTS['H'].mass
        w.composition[0, SYMBOL_TO_ID['H']] = h_initial
        w.masses[0] += h_initial

        outgassing.update(w)

        for i in range(w.n):
            comp_sum = w.composition[i].sum()
            assert comp_sum == pytest.approx(w.masses[i], rel=1e-9, abs=1e-9)


class TestNoLightElements:
    def test_pure_iron_body_does_not_outgas(self, cfg):
        """A body made entirely of heavy elements can't emit volatiles."""
        _viewer_cfg(cfg)
        cfg.outgassing.rate_per_step = 1.0
        cfg.outgassing.temperature_threshold_K = 100.0
        w = World(cfg)
        # Pure-Fe body — no H, no He, nothing light
        _make_hot_body(w, 'Fe', ratio=500.0)
        outgassing.update(w)
        assert w.total_outgassing == 0
