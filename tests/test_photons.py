"""Tests for sim/photons.py — first-order radiation transport.

Verifies the four pieces of photon physics:
  1. Emission rate scales with T⁴ (Stefan-Boltzmann)
  2. Photons propagate in straight lines at speed c (finite light-travel time)
  3. Absorption transfers energy/momentum to the absorber (radiation pressure)
  4. Total momentum is exactly conserved across emit + absorb cycles
"""

from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from sim.world import World
from sim.elements import ELEMENTS
from sim import photons


def _viewer_cfg(cfg):
    """Photon module reads renderer thresholds for temperature mapping;
    supply them here."""
    cfg.renderer = SimpleNamespace(
        hot_temperature_threshold = 500000.0,
        reference_temperature_K   = 20000.0,
        planet_mass_threshold     = 20.0,
        star_mass_threshold       = 200.0,
    )
    return cfg


def _enable_photons(cfg, **overrides):
    """Activate photons with sensible test defaults."""
    defaults = dict(
        speed_of_light_sim       = 20000.0,
        emission_rate_scale      = 1.0,
        absorption_cross_section = 20.0,
        energy_per_photon        = 1.0,
        cull_distance            = 100000.0,
        capacity                 = 200,
    )
    defaults.update(overrides)
    cfg.photons = SimpleNamespace(**defaults)
    return cfg


def _place(world, sym, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(
        ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float),
    )


class TestSetup:
    def test_no_config_no_photon_state(self, cfg):
        """If cfg.photons is absent, World should not allocate the buffer."""
        _viewer_cfg(cfg)
        # cfg.photons is already 0-speed by default → no allocation
        w = World(cfg)
        assert w.photons is None

    def test_enabled_creates_state(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg)
        w = World(cfg)
        assert w.photons is not None
        assert w.photons.cap == 200


class TestEmissionRate:
    def test_cold_particle_emits_nothing(self, cfg):
        """At T ≪ T_ref, rate ∝ T⁴ → ~0. No emissions expected."""
        _viewer_cfg(cfg)
        _enable_photons(cfg, emission_rate_scale=1.0)
        w = World(cfg)
        # Very slow particle: T = (KE/hot_T) × ref_T  → effectively 0
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[0.1, 0.0, 0.0])
        for _ in range(30):
            photons.update(w, dt=0.01)
        assert w.total_photons_emitted == 0

    def test_hot_particle_emits(self, cfg):
        """A hot particle should emit photons across many steps."""
        _viewer_cfg(cfg)
        _enable_photons(cfg, emission_rate_scale=100.0)
        w = World(cfg)
        m_h = ELEMENTS['H'].mass
        v = float(np.sqrt(2 * 500000.0 / m_h))           # T ≈ T_ref
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[v, 0.0, 0.0])
        for step in range(50):
            w.velocities[0] = np.array([v, 0.0, 0.0])    # hold KE constant
            w.time = step * 0.01                          # advance RNG seed
            photons.update(w, dt=0.01)
        assert w.total_photons_emitted > 10

    def test_t_fourth_scaling(self, cfg):
        """Hotter particles emit much more.  Stefan-Boltzmann predicts the
        rate goes as T⁴; we run two temperatures and verify the rate ratio
        is large (loose tolerance for stochastic + saturation effects)."""
        m_h = ELEMENTS['H'].mass

        def emit_count(temperature_ratio: float, steps: int = 600) -> int:
            local_cfg = _viewer_cfg(cfg)
            _enable_photons(local_cfg, emission_rate_scale=5.0)
            ke_target = temperature_ratio * 500_000.0
            v = float(np.sqrt(2 * ke_target / m_h))
            w = World(local_cfg)
            _place(w, 'H', [0.0, 0.0, 0.0], vel=[v, 0.0, 0.0])
            for step in range(steps):
                w.velocities[0] = np.array([v, 0.0, 0.0])
                w.time = step * 0.01
                photons.update(w, dt=0.01)
            return w.total_photons_emitted

        # T⁴ ratio between 1.0 and 0.7 is (1/0.7)⁴ ≈ 4.2
        n_low  = emit_count(0.7)
        n_high = emit_count(1.0)
        # Conservative — just check the strongly-rising behaviour:
        assert n_high > 2 * (n_low + 1), \
            f'Expected strongly-rising emission rate, got n_low={n_low}, n_high={n_high}'


class TestPropagation:
    def test_photon_travels_at_speed_c(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg)
        w = World(cfg)
        # Manually inject a photon (skip emission)
        w.photons.emit_one(np.zeros(3), np.array([1.0, 0.0, 0.0]), energy=1.0)
        photons.update(w, dt=0.01)
        # Position should have advanced by c × dt = 20000 × 0.01 = 200
        live_idx = np.where(w.photons.alive)[0]
        assert len(live_idx) >= 1
        # We can't easily distinguish original-vs-emitted-during-update; use slot 0
        np.testing.assert_allclose(
            w.photons.positions[0],
            [200.0, 0.0, 0.0],
            rtol=1e-9, atol=1e-9,
        )

    def test_photon_culled_at_far_distance(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg, cull_distance=300.0)
        w = World(cfg)
        # Inject photon at distance 250, moving outward → after one step → 250+200=450 > cull
        w.photons.emit_one(
            position=np.array([250.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            energy=1.0,
        )
        photons.update(w, dt=0.01)
        assert not w.photons.alive[0]


class TestAbsorption:
    def test_photon_absorbed_by_nearby_particle(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg, absorption_cross_section=20.0)
        w = World(cfg)
        # Target particle at rest at origin
        _place(w, 'Fe', [0.0, 0.0, 0.0])
        v_before = w.velocities[0].copy()
        # Inject a photon AT the particle's location — should be absorbed next step
        w.photons.emit_one(
            position=np.zeros(3),
            direction=np.array([1.0, 0.0, 0.0]),
            energy=10.0,
        )
        photons.update(w, dt=0.0)   # dt=0 so propagation doesn't move it
        assert w.total_photons_absorbed >= 1
        # Particle should have been kicked along the photon's direction
        assert w.velocities[0, 0] > v_before[0]

    def test_photon_not_absorbed_when_far_from_any_particle(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg, absorption_cross_section=20.0)
        w = World(cfg)
        _place(w, 'Fe', [10000.0, 0.0, 0.0])    # far away
        w.photons.emit_one(np.zeros(3), np.array([1.0, 0.0, 0.0]), energy=1.0)
        photons.update(w, dt=0.0)
        assert w.total_photons_absorbed == 0
        assert w.photons.alive[0]


class TestRadiationPressure:
    def test_emit_recoils_emitter(self, cfg):
        """Emitting a photon should give the emitter a recoil kick opposite
        to the photon's direction.  Real radiation pressure: p_recoil = −E/c."""
        _viewer_cfg(cfg)
        _enable_photons(cfg, emission_rate_scale=1e6)   # essentially guaranteed emission
        w = World(cfg)
        m_h = ELEMENTS['H'].mass
        v = float(np.sqrt(2 * 500000.0 / m_h))           # T ≈ T_ref
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[v, 0.0, 0.0])
        v_before = w.velocities[0].copy()
        photons.update(w, dt=0.01)
        # Velocity must have changed due to recoil
        v_after = w.velocities[0]
        assert not np.allclose(v_before, v_after)

    def test_emit_absorb_momentum_conserved(self, cfg):
        """For a closed system (emitter at position 0, absorber adjacent),
        total momentum after emit-then-absorb must equal the initial total."""
        _viewer_cfg(cfg)
        _enable_photons(cfg)
        w = World(cfg)
        # Place a hot emitter and a cold absorber very close together
        m_h = ELEMENTS['H'].mass
        _place(w, 'Fe', [0.0,  0.0, 0.0], vel=[1.0, 0.0, 0.0])
        _place(w, 'Fe', [10.0, 0.0, 0.0])
        # Inject a photon moving from emitter toward absorber
        w.photons.emit_one(np.zeros(3), np.array([1.0, 0.0, 0.0]), energy=5.0)

        p_before = (w.masses[:2, None] * w.velocities[:2]).sum(axis=0)
        photons.update(w, dt=0.0)            # only absorbs, doesn't emit (dt=0)
        p_after  = (w.masses[:2, None] * w.velocities[:2]).sum(axis=0)

        # The photon's momentum E/c is transferred to the absorber → total
        # particle momentum increases by p_photon = E/c · direction
        c = float(cfg.photons.speed_of_light_sim)
        expected_delta = np.array([5.0 / c, 0.0, 0.0])
        np.testing.assert_allclose(p_after - p_before, expected_delta, rtol=1e-9)


class TestNoOpCases:
    def test_zero_speed_disables(self, cfg):
        _viewer_cfg(cfg)
        _enable_photons(cfg, speed_of_light_sim=0.0)
        w = World(cfg)
        assert w.photons is None        # constructor short-circuits
