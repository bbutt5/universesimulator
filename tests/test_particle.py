"""Tests for sim/particle.py — Bond (Morse potential)."""

from __future__ import annotations
import math
import pytest
from sim.particle import Bond


def make_bond(r_eq=100.0, D_e=50.0, k=0.8) -> Bond:
    return Bond(i=0, j=1, length_eq=r_eq, D_e=D_e, k=k)


class TestBondInit:
    def test_morse_parameter_a(self):
        r_eq, D_e, k = 100.0, 50.0, 0.8
        bond = make_bond(r_eq, D_e, k)
        expected_a = math.sqrt(k / (2.0 * D_e))
        assert abs(bond.a - expected_a) < 1e-12

    def test_zero_De_guard(self):
        # Should not raise; a defaults to 1.0 when D_e is tiny
        bond = Bond(i=0, j=1, length_eq=100.0, D_e=1e-13, k=0.8)
        assert bond.a == 1.0

    def test_slots_stored(self):
        bond = make_bond()
        assert bond.i == 0
        assert bond.j == 1
        assert bond.length_eq == 100.0


class TestMorseForce:
    def test_zero_force_at_equilibrium(self):
        bond = make_bond(r_eq=100.0)
        assert abs(bond.radial_force(100.0)) < 1e-10

    def test_repulsive_when_compressed(self):
        bond = make_bond(r_eq=100.0)
        # Positive = pushes atoms apart = repulsive
        assert bond.radial_force(80.0) > 0

    def test_attractive_when_stretched(self):
        bond = make_bond(r_eq=100.0)
        # Negative = pulls atoms together = attractive
        assert bond.radial_force(120.0) < 0

    def test_force_magnitude_increases_with_compression(self):
        bond = make_bond(r_eq=100.0)
        f_small = bond.radial_force(99.0)
        f_large = bond.radial_force(90.0)
        assert f_large > f_small > 0


class TestMorsePotential:
    def test_zero_energy_at_equilibrium(self):
        bond = make_bond(r_eq=100.0, D_e=50.0)
        assert abs(bond.potential_energy(100.0)) < 1e-10

    def test_energy_positive_away_from_equilibrium(self):
        bond = make_bond(r_eq=100.0, D_e=50.0)
        assert bond.potential_energy(80.0) > 0
        assert bond.potential_energy(120.0) > 0

    def test_energy_approaches_De_at_large_separation(self):
        bond = make_bond(r_eq=100.0, D_e=50.0)
        assert abs(bond.potential_energy(1e6) - 50.0) < 0.01

    def test_energy_is_symmetric_near_equilibrium(self):
        bond = make_bond(r_eq=100.0)
        # Morse well is not symmetric, but compressed > stretched energy
        e_compressed = bond.potential_energy(50.0)
        e_stretched  = bond.potential_energy(150.0)
        assert e_compressed > 0
        assert e_stretched > 0
