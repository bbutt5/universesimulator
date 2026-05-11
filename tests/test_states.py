"""Tests for sim/states.py — state classification (solid/liquid/gas/plasma)."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.particle import Bond
from sim.states import (
    classify, state_counts,
    STATE_SOLID, STATE_LIQUID, STATE_GAS, STATE_PLASMA,
)


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


def _grid_cluster(world, sym: str, side: int = 3, spacing: float = 80.0,
                  velocity: np.ndarray | None = None):
    """Place a side³ cube of atoms; defaults to at rest."""
    if velocity is None:
        velocity = np.zeros(3)
    for x in range(side):
        for y in range(side):
            for z in range(side):
                _place(world, sym,
                       [x * spacing, y * spacing, z * spacing],
                       vel=velocity.copy())


class TestClassify:
    def test_empty_world(self, cfg):
        w = World(cfg)
        s = classify(w)
        assert s.shape == (0,)

    def test_single_particle_is_gas(self, cfg):
        w = World(cfg)
        _place(w, 'He', [0.0, 0.0, 0.0])
        s = classify(w)
        assert s[0] == STATE_GAS

    def test_isolated_pair_is_gas(self, cfg):
        """Two atoms within cutoff but with no other neighbours → still gas."""
        w = World(cfg)
        _place(w, 'He', [0.0,   0.0, 0.0])
        _place(w, 'He', [100.0, 0.0, 0.0])
        s = classify(w)
        # Density is just 1 neighbour each → gas
        assert all(code == STATE_GAS for code in s)

    def test_bonded_pair_is_solid(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0])
        _place(w, 'H', [62.0, 0.0, 0.0])
        w.bonds.append(Bond(0, 1, 62.0, 50.0, 0.8))
        w.bond_counts[0] = 1
        w.bond_counts[1] = 1
        s = classify(w)
        # Both bonded → both classified as solid regardless of density
        assert s[0] == STATE_SOLID
        assert s[1] == STATE_SOLID


class TestCondensedPhases:
    def test_cold_dense_cluster_has_solid_atoms(self, cfg):
        """A 3x3x3 cube of cold He atoms — interior atoms have ≥6 neighbours,
        which should classify them as solid."""
        w = World(cfg)
        _grid_cluster(w, 'He', side=3, spacing=80.0)
        s = classify(w)
        # The single interior atom (index 13 in a 3³ grid) should be solid
        assert (s == STATE_SOLID).any()

    def test_hot_dense_cluster_has_liquid_or_gas_atoms(self, cfg):
        """Same dense cube, but each atom moving fast and randomly →
        local relative KE high → solid criterion fails → liquid/gas."""
        rng = np.random.default_rng(7)
        # Heat well above the bond-velocity threshold so |Δv| > v_thresh
        v_thresh = cfg.chemistry.bond_velocity_threshold
        sigma    = v_thresh * 2.0
        w = World(cfg)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    v = rng.normal(0.0, sigma, 3)
                    _place(w, 'He', [x * 80.0, y * 80.0, z * 80.0], vel=v)
        s = classify(w)
        # No interior atom should be classified as solid
        assert not (s == STATE_SOLID).any()
        # And the dense interior should be liquid (not gas)
        assert (s == STATE_LIQUID).any()

    def test_sparse_cluster_is_all_gas(self, cfg):
        """Wide spacing → very few neighbours within cutoff → gas."""
        w = World(cfg)
        cutoff = cfg.thermal.vdw_cutoff
        _grid_cluster(w, 'He', side=2, spacing=cutoff * 1.2)  # beyond cutoff
        s = classify(w)
        assert all(code == STATE_GAS for code in s)


class TestPlasma:
    def test_ionized_particle_is_plasma(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0])
        w.ionized[0] = True
        s = classify(w)
        assert s[0] == STATE_PLASMA

    def test_ionization_overrides_bonded_solid(self, cfg):
        """Ionising a bonded atom: classifier returns plasma (bonds will be
        broken on the next step by chemistry)."""
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0])
        _place(w, 'H', [62.0, 0.0, 0.0])
        w.bonds.append(Bond(0, 1, 62.0, 50.0, 0.8))
        w.bond_counts[0] = 1
        w.bond_counts[1] = 1
        w.ionized[0] = True
        s = classify(w)
        assert s[0] == STATE_PLASMA
        assert s[1] == STATE_SOLID  # the other is still bonded


class TestStateCounts:
    def test_counts_sum_to_n(self, cfg):
        w = World(cfg)
        _grid_cluster(w, 'He', side=3, spacing=80.0)
        counts = state_counts(w)
        assert sum(counts.values()) == w.n

    def test_returns_all_four_keys(self, cfg):
        w = World(cfg)
        counts = state_counts(w)
        assert set(counts.keys()) == {'solid', 'liquid', 'gas', 'plasma'}
