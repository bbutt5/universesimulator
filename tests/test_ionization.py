"""Tests for the ionisation lifecycle in sim/world.py.

Covers the third Phase-1 acceptance criterion: a very high-temperature
region should ionise into plasma, with hysteresis on recombination."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond
from sim import chemistry


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestIonizationThreshold:
    def test_slow_particle_not_ionized(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[10.0, 0.0, 0.0])
        w._update_ionization()
        assert not w.ionized[0]

    def test_fast_particle_ionizes(self, cfg):
        w = World(cfg)
        # KE threshold = 1e5; H mass = 1.008; v required ≈ sqrt(2 × 1e5 / 1.008) ≈ 446
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]

    def test_skipped_when_thermal_cfg_absent(self, cfg):
        del cfg.thermal
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[5000.0, 0.0, 0.0])
        w._update_ionization()   # must not raise
        assert not w.ionized[0]


class TestHysteresis:
    def test_ionized_stays_ionized_when_slowing_through_window(self, cfg):
        """Once ionised, a particle should remain ionised until KE drops
        below the lower recombination threshold."""
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]
        # Slow down to KE between recomb and ion thresholds:
        # recomb_thresh = 1e4 → v ≈ 141; choose v=300 (KE ≈ 4.5e4)
        w.velocities[0] = np.array([300.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0], "should remain ionised (hysteresis)"

    def test_ionized_recombines_below_lower_threshold(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]
        # Drop well below recomb_threshold = 1e4 → v < 141; use v=50
        w.velocities[0] = np.array([50.0, 0.0, 0.0])
        w._update_ionization()
        assert not w.ionized[0]


class TestIonizationBreaksBonds:
    def test_ionising_a_bonded_atom_breaks_its_bond(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0])
        _place(w, 'H', [62.0, 0.0, 0.0])
        w.bonds.append(Bond(0, 1, 62.0, 50.0, 0.8))
        w.bond_counts[0] = 1
        w.bond_counts[1] = 1

        w.ionized[0] = True
        chemistry._break_bonds(w)
        assert len(w.bonds) == 0
        assert w.bond_counts[0] == 0
        assert w.bond_counts[1] == 0

    def test_ionised_atom_cannot_form_new_bond(self, cfg):
        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0,        0.0, 0.0])
        _place(w, 'H', [r_eq * 1.2, 0.0, 0.0])
        w.ionized[0] = True
        chemistry._form_bonds(w)
        assert len(w.bonds) == 0
