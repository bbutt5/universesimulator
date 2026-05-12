"""Tests for sim/extinction.py — Beer-Lambert extinction along the line of sight.

These validate the physics of dark nebulae: when cold gas sits between an
emitter and the observer, the emitter's apparent flux drops by exp(−τ)
where τ accumulates from each intervening absorber. A physicist can check
the law directly against Rybicki & Lightman Ch. 1.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.extinction import compute_visibility, per_particle_optical_depth


class TestEdgeCases:
    def test_empty_world(self):
        v = compute_visibility(np.zeros((0, 3)), np.zeros(0), np.zeros(3))
        assert v.shape == (0,)

    def test_single_particle_no_obstruction(self):
        """A lone particle has no blockers → visibility = 1."""
        v = compute_visibility(
            np.array([[100.0, 0.0, 0.0]]),
            np.array([0.5]),
            np.zeros(3),
        )
        np.testing.assert_array_equal(v, 1.0)


class TestBeerLambert:
    def test_clear_line_of_sight_returns_near_one(self):
        """Two well-separated particles → both essentially fully visible.
        The kernel has a smooth Gaussian falloff so far-off particles still
        contribute *something* (physically correct — diffuse gas absorbs
        a little even off-axis); we just check the attenuation is tiny."""
        positions = np.array([
            [100.0,    0.0, 0.0],
            [100.0,  500.0, 0.0],   # well off to the side
        ])
        v = compute_visibility(positions, np.array([1.0, 1.0]), np.zeros(3))
        np.testing.assert_allclose(v, 1.0, atol=1e-3)

    def test_blocker_in_front_dims_background(self):
        """Camera at origin → blocker at (50, 0, 0) → source at (100, 0, 0).
        The source is behind the blocker; its visibility drops below 1."""
        positions = np.array([
            [100.0, 0.0, 0.0],     # source (i=0)
            [ 50.0, 0.0, 0.0],     # blocker (i=1)
        ])
        # Give the blocker substantial optical depth, source none
        optical_depths = np.array([0.0, 2.0])
        v = compute_visibility(positions, optical_depths, np.zeros(3), kernel_sigma=30.0)
        assert v[0] < 1.0     # source is dimmed
        # exp(-tau · K(0)) = exp(-2 · 1) = exp(-2) ≈ 0.135
        assert v[0] == pytest.approx(np.exp(-2.0), rel=0.01)
        assert v[1] == pytest.approx(1.0, rel=1e-6)   # nothing in front of blocker

    def test_no_blocker_behind_emitter(self):
        """A particle BEHIND the emitter (further from camera) cannot
        block it — geometry only attenuates the FARTHER particle."""
        positions = np.array([
            [100.0, 0.0, 0.0],     # nearer to camera
            [200.0, 0.0, 0.0],     # farther — behind the first
        ])
        # Both potentially opaque; only the farther one should be dimmed
        v = compute_visibility(positions, np.array([1.0, 1.0]), np.zeros(3))
        assert v[0] == pytest.approx(1.0, rel=1e-6)   # nearer one: nothing in front
        assert v[1] < 1.0                              # farther one: blocked by the nearer

    def test_perpendicular_distance_falloff(self):
        """A blocker offset perpendicular to the line of sight attenuates
        less than one directly on the line."""
        # Source at (100, 0, 0); two blockers at z = 50: one on the line, one offset
        cam = np.zeros(3)

        v_on_line = compute_visibility(
            np.array([
                [100.0, 0.0, 0.0],    # source
                [ 50.0, 0.0, 0.0],    # blocker on the line
            ]),
            np.array([0.0, 1.0]),
            cam, kernel_sigma=20.0,
        )
        v_offset = compute_visibility(
            np.array([
                [100.0,  0.0, 0.0],    # source
                [ 50.0, 50.0, 0.0],    # blocker 50 SU off the line
            ]),
            np.array([0.0, 1.0]),
            cam, kernel_sigma=20.0,
        )
        # Off-line blocker absorbs much less → source brighter
        assert v_offset[0] > v_on_line[0]
        assert v_on_line[0] < 1.0

    def test_two_blockers_sum_tau(self):
        """Optical depth from two blockers should add (multiplicative
        visibility: exp(−τ1 − τ2))."""
        positions = np.array([
            [100.0, 0.0, 0.0],   # source
            [ 30.0, 0.0, 0.0],   # blocker 1
            [ 60.0, 0.0, 0.0],   # blocker 2
        ])
        optical_depths = np.array([0.0, 1.0, 1.0])
        v = compute_visibility(positions, optical_depths, np.zeros(3), kernel_sigma=30.0)
        # tau ≈ 1·K(0) + 1·K(0) = 2 → visibility ≈ exp(-2) ≈ 0.135
        assert v[0] == pytest.approx(np.exp(-2.0), rel=0.05)


class TestPerParticleOpticalDepth:
    def test_cold_particle_is_opaque(self):
        """Below cold_threshold, optical depth is near base_opacity × size."""
        T = np.array([100.0])           # cold
        sizes = np.array([10.0])
        opacity = per_particle_optical_depth(T, sizes, cold_threshold_K=3000.0, base_opacity=0.05)
        # cold_fraction → 1; expect 0.05 · 10 · 1 = 0.5
        assert opacity[0] == pytest.approx(0.5, rel=0.01)

    def test_hot_particle_is_transparent(self):
        """Well above cold_threshold, optical depth → 0 (hot gas is in
        emission, not absorption)."""
        T = np.array([30000.0])         # very hot
        sizes = np.array([10.0])
        opacity = per_particle_optical_depth(T, sizes, cold_threshold_K=3000.0, base_opacity=0.05)
        # cold_fraction at T=30000 is 1/(1 + 100) ≈ 0.0099; expect very small
        assert opacity[0] < 0.05    # well below the cold value

    def test_monotonic_in_temperature(self):
        """Increasing T strictly decreases the optical-depth contribution."""
        T = np.array([100.0, 1000.0, 5000.0, 20000.0])
        sizes = np.full(4, 10.0)
        opacity = per_particle_optical_depth(T, sizes, 3000.0, 0.05)
        # strictly decreasing
        assert np.all(np.diff(opacity) < 0.0)

    def test_proportional_to_size(self):
        """Bigger particles block more light (proportional to cross-section)."""
        T = np.array([100.0, 100.0])
        sizes = np.array([5.0, 50.0])
        opacity = per_particle_optical_depth(T, sizes, 3000.0, 0.05)
        assert opacity[1] == pytest.approx(opacity[0] * 10.0, rel=1e-6)


class TestPhysicalProperties:
    def test_visibility_always_in_zero_one(self):
        """exp(−τ) ∈ (0, 1] for τ ≥ 0 — a sanity check on the law."""
        rng = np.random.default_rng(0)
        positions = rng.normal(0, 200, (20, 3))
        optical_depths = np.abs(rng.normal(0.5, 0.2, 20))
        cam = np.array([500.0, 0.0, 0.0])
        v = compute_visibility(positions, optical_depths, cam)
        assert np.all((v > 0.0) & (v <= 1.0))

    def test_zero_optical_depth_means_full_visibility(self):
        positions = np.array([[100.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        v = compute_visibility(positions, np.zeros(2), np.zeros(3))
        np.testing.assert_allclose(v, 1.0, rtol=1e-6)
