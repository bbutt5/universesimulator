"""Tests for sim/nuclear.gamow_factor — quantum tunnelling below the
classical Coulomb barrier.

Reference values from Clayton, "Principles of Stellar Evolution and
Nucleosynthesis" §4.3 and standard nuclear-physics tables.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.nuclear import gamow_factor, gamow_factor_sim, AMU_TO_MEV


class TestGamowProperties:
    def test_zero_energy_zero_probability(self):
        """No kinetic energy → no tunnelling. (Limit of exp(−∞)→0.)"""
        assert gamow_factor(1, 1, 1, 1, rel_ke_mev=0.0) == 0.0

    def test_monotonic_increase_with_energy(self):
        """Higher CM kinetic energy → larger tunnelling probability."""
        a = gamow_factor(1, 1, 1, 1, rel_ke_mev=1e-3)   # 1 keV
        b = gamow_factor(1, 1, 1, 1, rel_ke_mev=1e-2)   # 10 keV
        c = gamow_factor(1, 1, 1, 1, rel_ke_mev=1e-1)   # 100 keV
        assert 0.0 < a < b < c < 1.0

    def test_higher_charge_suppresses_tunnelling(self):
        """At fixed CM energy, larger Z₁Z₂ → tighter exponential suppression."""
        e_cm = 0.001                                     # 1 keV (stellar core)
        p_hh   = gamow_factor(1, 1,  1, 1, e_cm)        # Z₁Z₂ = 1
        p_hehe = gamow_factor(2, 4,  2, 4, e_cm)        # Z₁Z₂ = 4
        p_fefe = gamow_factor(26, 56, 26, 56, e_cm)     # Z₁Z₂ = 676
        assert p_hh > p_hehe > p_fefe
        # Fe+Fe at solar-core energy is effectively zero
        assert p_fefe < 1e-100

    def test_solar_core_h_h(self):
        """H + H tunnelling at the solar core (~1.5 keV) gives a finite,
        very small probability — this is what powers the Sun.

        Gamow energy for H+H: E_G = (2πα·1·1)² · 2 · (0.5 amu·c²)
                                  ≈ 0.49 MeV
        At E_cm = 1.5 keV = 1.5e-3 MeV:
            sqrt(E_G / E_cm) = sqrt(326) ≈ 18.1
            P ≈ exp(-18.1) ≈ 1.4e-8
        """
        p = gamow_factor(1, 1, 1, 1, rel_ke_mev=1.5e-3)
        assert 1e-12 < p < 1e-5     # ballpark — matches order of magnitude


class TestGamowInSimUnits:
    def test_sim_to_mev_zero_disables(self):
        """sim_to_mev=0 means we have no unit-conversion — return 0."""
        assert gamow_factor_sim(1, 1, 1, 1, rel_ke_sim=1.0, sim_to_mev=0.0) == 0.0

    def test_matches_direct_call(self):
        """gamow_factor_sim(KE, scale) == gamow_factor(KE * scale)."""
        ke_sim = 100.0
        scale = 0.001    # 1 sim_KE = 1 keV
        a = gamow_factor_sim(1, 1, 1, 1, rel_ke_sim=ke_sim, sim_to_mev=scale)
        b = gamow_factor(1, 1, 1, 1, rel_ke_mev=ke_sim * scale)
        assert a == pytest.approx(b, rel=1e-12)
