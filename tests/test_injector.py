"""Tests for sim/injector.py — Injector class."""

from __future__ import annotations
import numpy as np
import pytest
from sim.injector import Injector
from sim.world import World
from sim.elements import ELEMENTS


@pytest.fixture
def injector(cfg):
    return Injector(cfg)


class TestDistribution:
    def test_weights_sum_to_one(self, injector):
        _, weights = injector._get_distribution()
        assert abs(weights.sum() - 1.0) < 1e-12

    def test_symbols_are_valid_elements(self, injector):
        symbols, _ = injector._get_distribution()
        for sym in symbols:
            assert sym in ELEMENTS

    def test_distribution_cached(self, injector):
        result1 = injector._get_distribution()
        result2 = injector._get_distribution()
        assert result1 is result2   # same object (cached)


class TestSeeds:
    def test_seeds_generated_correctly(self, cfg):
        inj = Injector(cfg)
        seeds = inj._get_seeds()
        n_seeds = cfg.cosmology.n_density_seeds
        assert seeds.shape == (n_seeds, 3)

    def test_seeds_cached(self, injector):
        s1 = injector._get_seeds()
        s2 = injector._get_seeds()
        assert s1 is s2

    def test_seeds_within_unit_sphere(self, injector):
        seeds = injector._get_seeds()
        norms = np.linalg.norm(seeds, axis=1)
        assert np.all(norms <= 1.0)

    def test_no_seeds_when_disabled(self, cfg):
        cfg.cosmology.n_density_seeds = 0
        inj = Injector(cfg)
        seeds = inj._get_seeds()
        assert seeds.shape == (0, 3)


class TestPositionSampling:
    def test_position_inside_sphere(self, injector):
        radius = 500.0
        for _ in range(50):
            pos = injector._sample_position(radius)
            assert np.linalg.norm(pos) <= radius + 1e-9

    def test_uniform_sphere_inside_sphere(self, injector):
        radius = 200.0
        for _ in range(20):
            pos = injector._uniform_sphere_volume(radius)
            assert np.linalg.norm(pos) <= radius + 1e-9


class TestVelocitySampling:
    def test_velocity_within_max_without_hubble(self, cfg):
        cfg.cosmology.hubble_initial = 0.0
        inj = Injector(cfg)
        v_max = cfg.simulation.injection_velocity_max
        for _ in range(50):
            pos = np.array([1.0, 0.0, 0.0])
            vel = inj._sample_velocity(pos)
            assert np.linalg.norm(vel) <= v_max + 1e-9

    def test_hubble_adds_outward_velocity(self, cfg):
        cfg.cosmology.hubble_initial = 1.0
        cfg.simulation.injection_velocity_max = 0.0   # disable thermal
        inj = Injector(cfg)
        pos = np.array([10.0, 0.0, 0.0])
        vel = inj._sample_velocity(pos)
        # With H=1.0 and pos=[10,0,0], Hubble adds [10,0,0] to vel
        assert vel[0] == pytest.approx(10.0)

    def test_no_hubble_when_disabled(self, cfg):
        cfg.cosmology.hubble_initial = 0.0
        cfg.simulation.injection_velocity_max = 0.0
        inj = Injector(cfg)
        pos = np.array([100.0, 0.0, 0.0])
        vel = inj._sample_velocity(pos)
        np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-12)


class TestInject:
    def test_inject_adds_particles(self, cfg):
        w = World(cfg)
        inj = w.injector
        inj.inject(w, dt=1.0)   # rate=10, dt=1.0 → should inject 10
        assert w.n == 10

    def test_inject_respects_max_particles(self, cfg):
        cfg.simulation.max_particles = 5
        w = World(cfg)
        w.injector.inject(w, dt=100.0)
        assert w.n <= 5

    def test_inject_does_nothing_when_full(self, cfg):
        cfg.simulation.max_particles = 0
        w = World(cfg)
        w.injector.inject(w, dt=1.0)
        assert w.n == 0

    def test_injected_symbols_are_valid(self, cfg):
        w = World(cfg)
        w.injector.inject(w, dt=1.0)
        from sim.elements import ELEMENTS_LIST
        for i in range(w.n):
            sym = ELEMENTS_LIST[w.elem_ids[i]].symbol
            assert sym in ELEMENTS
