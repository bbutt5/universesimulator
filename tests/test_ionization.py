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
        """With H IE = 13.598 eV and scale = 7355, threshold ≈ 100k sim KE.
        H mass = 1.008 amu → required v ≈ sqrt(2 × 100k / 1.008) ≈ 446 SU/s."""
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]

    def test_skipped_when_thermal_cfg_absent(self, cfg):
        del cfg.thermal
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[5000.0, 0.0, 0.0])
        w._update_ionization()   # must not raise
        assert not w.ionized[0]


class TestPerElementPhysics:
    """The signature physics test: each element ionises at its own measured
    threshold, derived from NIST first-IE data (eV)."""

    def test_helium_harder_to_ionize_than_hydrogen(self, cfg):
        """At a kinetic energy that ionises H, He must still be neutral
        because its first IE is higher (24.587 eV vs 13.598 eV).

        Use KE = 150,000 sim units (between H and He thresholds with the
        conftest scale of 7355 eV⁻¹):
          - H threshold = 13.598 × 7355 ≈ 100,000  → H ionises
          - He threshold = 24.587 × 7355 ≈ 180,838 → He remains neutral
        """
        ke_target = 150_000.0
        m_h, m_he = ELEMENTS['H'].mass, ELEMENTS['He'].mass
        v_h  = float(np.sqrt(2 * ke_target / m_h))
        v_he = float(np.sqrt(2 * ke_target / m_he))

        w = World(cfg)
        _place(w, 'H',  [0.0,   0.0, 0.0], vel=[v_h,  0.0, 0.0])
        _place(w, 'He', [100.0, 0.0, 0.0], vel=[v_he, 0.0, 0.0])
        w._update_ionization()

        assert w.ionized[0],     "H should ionise at KE > 100k"
        assert not w.ionized[1], "He should NOT ionise at KE < 181k (NIST IE)"

    def test_helium_ionizes_above_its_own_threshold(self, cfg):
        """Increase He's energy past its real first-IE → ionises."""
        # Push KE = 200,000 → above He threshold of ~181k
        ke_target = 200_000.0
        m_he = ELEMENTS['He'].mass
        v_he = float(np.sqrt(2 * ke_target / m_he))

        w = World(cfg)
        _place(w, 'He', [0.0, 0.0, 0.0], vel=[v_he, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]

    def test_alkali_metals_ionize_more_easily_than_noble_gases(self, cfg):
        """Na (5.139 eV) should ionise at a KE that leaves Ar (15.760 eV)
        neutral — physics, not handwaving."""
        # KE between Na and Ar thresholds: Na ≈ 37,800; Ar ≈ 115,900.
        ke_target = 60_000.0
        m_na, m_ar = ELEMENTS['Na'].mass, ELEMENTS['Ar'].mass
        v_na = float(np.sqrt(2 * ke_target / m_na))
        v_ar = float(np.sqrt(2 * ke_target / m_ar))

        w = World(cfg)
        _place(w, 'Na', [0.0,   0.0, 0.0], vel=[v_na, 0.0, 0.0])
        _place(w, 'Ar', [200.0, 0.0, 0.0], vel=[v_ar, 0.0, 0.0])
        w._update_ionization()

        assert w.ionized[0],     "Na (IE 5.14 eV) should ionise"
        assert not w.ionized[1], "Ar (IE 15.76 eV) should remain neutral"


class TestHysteresis:
    def test_ionized_stays_ionized_when_slowing_through_window(self, cfg):
        """Once ionised, a particle should remain ionised until KE drops
        below the lower recombination threshold."""
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]
        # H ion-thresh ≈ 100k, recomb ≈ 10k. Pick KE ≈ 50k → v ≈ 315.
        w.velocities[0] = np.array([315.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0], "should remain ionised (hysteresis)"

    def test_ionized_recombines_below_lower_threshold(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[600.0, 0.0, 0.0])
        w._update_ionization()
        assert w.ionized[0]
        # Drop well below recomb threshold (≈ 10k for H) → v < 141; use v=50
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
