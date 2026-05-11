"""Tests for sim/diagnostics.py — conservation-law diagnostics.

These validate that the velocity-Verlet integrator does what it claims:
total linear momentum is exactly conserved when no clamp fires, and
total energy oscillates but does not drift.  If these tests start
failing, the integrator implementation is wrong (not the test).
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS
from sim import diagnostics


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


@pytest.fixture
def no_fusion_cfg(cfg):
    """Disable fusion + accretion + bonds so we test pure gravity dynamics."""
    cfg.chemistry.fusion_enabled = False
    cfg.chemistry.bond_velocity_threshold = 0.0   # nothing slow enough
    cfg.accretion.accretion_radius = 0.0          # no merges
    # Keep ions away from each other so VdW + thermal pressure stay quiet
    return cfg


class TestKineticEnergy:
    def test_zero_for_at_rest_particles(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0])
        _place(w, 'H', [100.0, 0.0, 0.0])
        assert diagnostics.kinetic_energy(w) == 0.0

    def test_single_particle_one_half_mv_squared(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[10.0, 0.0, 0.0])
        m = ELEMENTS['H'].mass
        assert diagnostics.kinetic_energy(w) == pytest.approx(0.5 * m * 100.0)

    def test_empty_world_zero(self, cfg):
        w = World(cfg)
        assert diagnostics.kinetic_energy(w) == 0.0


class TestGravitationalPE:
    def test_two_particles_classic_formula(self, cfg):
        """PE = -G·m₁·m₂ / √(r² + ε²) — matches the softened gravity force."""
        w = World(cfg)
        r = 100.0
        _place(w, 'Fe', [0.0, 0.0, 0.0])
        _place(w, 'Fe', [r,   0.0, 0.0])
        G   = cfg.physics.gravity_constant
        eps = cfg.physics.softening_length
        m   = ELEMENTS['Fe'].mass
        expected = -G * m * m / np.sqrt(r * r + eps * eps)
        assert diagnostics.gravitational_pe(w) == pytest.approx(expected)

    def test_empty_world_zero(self, cfg):
        w = World(cfg)
        assert diagnostics.gravitational_pe(w) == 0.0


class TestMomentumConservation:
    def test_isolated_pair_conserves_momentum_after_step(self, no_fusion_cfg):
        """Velocity-Verlet on an isolated pair should be momentum-exact
        (force is symmetric → ∫F dt sums to zero)."""
        w = World(no_fusion_cfg)
        _place(w, 'Fe', [0.0,  0.0, 0.0], vel=[ 1.0, 0.5, 0.0])
        _place(w, 'Fe', [200.0, 0.0, 0.0], vel=[-1.0, 0.5, 0.0])
        # Stop the injector from adding extra particles each step
        w.cfg.simulation.injection_rate = 0

        p0 = diagnostics.linear_momentum(w).copy()
        for _ in range(50):
            w.step(0.01)
        p1 = diagnostics.linear_momentum(w)

        # Symmetric gravity: should match to numerical noise
        np.testing.assert_allclose(p0, p1, atol=1e-8)
        assert w.total_velocity_clamps == 0, "Clamp should not fire in this test"


class TestEnergyDrift:
    def test_isolated_pair_no_long_term_drift(self, no_fusion_cfg):
        """Velocity-Verlet is symplectic: energy oscillates but doesn't drift.
        Over 200 steps with reasonable dt, drift should be < 1%."""
        cfg = no_fusion_cfg
        cfg.simulation.injection_rate = 0
        cfg.physics.softening_length = 50.0   # generous softening for stability
        w = World(cfg)
        _place(w, 'Fe', [-200.0, 0.0, 0.0], vel=[ 0.0,  0.5, 0.0])
        _place(w, 'Fe', [ 200.0, 0.0, 0.0], vel=[ 0.0, -0.5, 0.0])

        e0 = diagnostics.total_energy(w)
        for _ in range(200):
            w.step(0.005)
        e1 = diagnostics.total_energy(w)

        drift = abs(e1 - e0) / abs(e0)
        assert drift < 0.05, f"Energy drift {drift*100:.2f}% too large"


class TestClampCounter:
    def test_starts_at_zero(self, cfg):
        w = World(cfg)
        assert w.total_velocity_clamps == 0

    def test_increments_when_clamp_fires(self, cfg):
        cfg.simulation.injection_rate = 0
        cfg.physics.max_velocity = 1.0    # force the clamp to fire
        w = World(cfg)
        _place(w, 'Fe', [0.0, 0.0, 0.0], vel=[100.0, 0.0, 0.0])
        _place(w, 'Fe', [50.0, 0.0, 0.0], vel=[-100.0, 0.0, 0.0])
        w.step(0.01)
        assert w.total_velocity_clamps >= 1


class TestConservationReport:
    def test_report_includes_all_fields(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[10.0, 0.0, 0.0])
        _place(w, 'H', [100.0, 0.0, 0.0])
        report = diagnostics.conservation_report(w)
        assert set(report.keys()) >= {'kinetic', 'potential', 'total', '|momentum|', 'clamps'}

    def test_drift_pct_when_reference_given(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[10.0, 0.0, 0.0])
        _place(w, 'H', [100.0, 0.0, 0.0])
        e0 = diagnostics.total_energy(w)
        report = diagnostics.conservation_report(w, reference_energy=e0)
        assert report['drift_pct'] == pytest.approx(0.0)
