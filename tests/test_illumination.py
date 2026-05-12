"""Tests for sim/illumination.py — inverse-square illumination of cold
gas by hot emitters.

These validate the *physics* of reflection nebulae: gas near a hot star
receives flux proportional to L_star / (4πr²), with the star's blackbody
colour. A physicist reading the tests can check each formula directly.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.illumination import compute_received_rgb_flux


def _trivial_rgb_fn(T_kelvin: np.ndarray) -> np.ndarray:
    """For these tests, ignore real blackbody curves and return constant
    white RGB. Decouples the inverse-square geometry from colour mapping."""
    return np.ones((len(T_kelvin), 3), dtype=np.float32)


class TestInverseSquareLaw:
    def test_no_emitters_returns_zero(self):
        """All particles cold → no illumination anywhere."""
        pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        T   = np.array([100.0, 100.0])     # both cold (below 2000 K threshold)
        flux = compute_received_rgb_flux(pos, T, _trivial_rgb_fn, reference_T_K=20000.0)
        np.testing.assert_array_equal(flux, 0.0)

    def test_single_emitter_one_receiver(self):
        """Receiver at distance r from a hot emitter receives L / (4πr²)."""
        pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        T   = np.array([20000.0, 100.0])   # emitter (= ref T) + cold receiver
        flux = compute_received_rgb_flux(pos, T, _trivial_rgb_fn, reference_T_K=20000.0)

        # L_emitter = (20000/20000)⁴ = 1, r² = 100² = 1e4
        # Expected flux at receiver: 1 / (4π·1e4) ≈ 7.96e-6
        expected = 1.0 / (4.0 * np.pi * 1e4)
        assert flux[1, 0] == pytest.approx(expected, rel=1e-6)
        # Emitter itself: excluded from self-illumination
        np.testing.assert_array_equal(flux[0], 0.0)

    def test_inverse_square_falloff(self):
        """Doubling the distance quarters the flux."""
        pos = np.array([
            [0.0,    0.0, 0.0],   # emitter
            [100.0,  0.0, 0.0],   # receiver at r
            [200.0,  0.0, 0.0],   # receiver at 2r
            [400.0,  0.0, 0.0],   # receiver at 4r
        ])
        T = np.array([20000.0, 100.0, 100.0, 100.0])
        flux = compute_received_rgb_flux(pos, T, _trivial_rgb_fn, 20000.0)
        # flux[1] : flux[2] : flux[3] = 1 : 1/4 : 1/16
        assert flux[1, 0] == pytest.approx(4.0 * flux[2, 0], rel=1e-6)
        assert flux[2, 0] == pytest.approx(4.0 * flux[3, 0], rel=1e-6)

    def test_T_to_fourth_luminosity_scaling(self):
        """Doubling the emitter's temperature multiplies its luminosity by 16
        (Stefan-Boltzmann: L ∝ T⁴)."""
        pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        T_low  = np.array([10000.0, 100.0])
        T_high = np.array([20000.0, 100.0])   # 2× temperature
        flux_low  = compute_received_rgb_flux(pos, T_low,  _trivial_rgb_fn, 20000.0)
        flux_high = compute_received_rgb_flux(pos, T_high, _trivial_rgb_fn, 20000.0)
        # 2⁴ = 16× more flux at receiver
        assert flux_high[1, 0] == pytest.approx(16.0 * flux_low[1, 0], rel=1e-6)


class TestMultipleEmitters:
    def test_flux_sums_linearly_over_emitters(self):
        """Two identical emitters at the same distance produce 2× the flux
        of a single emitter — linearity of radiative transfer."""
        pos_1emitter = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        T_1emitter   = np.array([20000.0, 100.0])
        flux_one = compute_received_rgb_flux(pos_1emitter, T_1emitter, _trivial_rgb_fn, 20000.0)

        # Two emitters equidistant from a single receiver in the middle
        pos_2emitters = np.array([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        T_2emitters   = np.array([20000.0, 20000.0, 100.0])
        flux_two = compute_received_rgb_flux(pos_2emitters, T_2emitters, _trivial_rgb_fn, 20000.0)

        # The receiver index 2 should see 2× the single-emitter case
        assert flux_two[2, 0] == pytest.approx(2.0 * flux_one[1, 0], rel=1e-6)


class TestColorWeighting:
    def test_received_color_is_emitters_color(self):
        """A blue emitter illuminates surrounding gas blue; red emitter, red."""
        def red_rgb_fn(T):
            return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(T), 1))
        def blue_rgb_fn(T):
            return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(T), 1))

        pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        T   = np.array([20000.0, 100.0])

        flux_red = compute_received_rgb_flux(pos, T, red_rgb_fn, 20000.0)
        flux_blue = compute_received_rgb_flux(pos, T, blue_rgb_fn, 20000.0)
        # Receiver gets only red flux from red emitter, only blue from blue
        assert flux_red[1, 0] > 0 and flux_red[1, 2] == 0
        assert flux_blue[1, 0] == 0 and flux_blue[1, 2] > 0


class TestSelfIllumination:
    def test_emitter_does_not_illuminate_itself(self):
        """A hot particle's *own* emission is rendered elsewhere; the
        illumination function must exclude self-pairs to avoid double-counting."""
        pos = np.array([[0.0, 0.0, 0.0]])
        T   = np.array([20000.0])
        flux = compute_received_rgb_flux(pos, T, _trivial_rgb_fn, 20000.0)
        np.testing.assert_array_equal(flux, 0.0)


class TestEdgeCases:
    def test_empty_world(self):
        flux = compute_received_rgb_flux(
            np.zeros((0, 3)), np.zeros(0), _trivial_rgb_fn, 20000.0,
        )
        assert flux.shape == (0, 3)

    def test_softening_prevents_divergence(self):
        """Overlapping receiver and emitter shouldn't crash with 1/0."""
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # same position
        T   = np.array([20000.0, 100.0])
        flux = compute_received_rgb_flux(pos, T, _trivial_rgb_fn, 20000.0)
        assert np.all(np.isfinite(flux))


class TestReinhardToneMapping:
    """The renderer applies L → eL / (1 + eL) per channel. Confirm the
    physically-required properties: identity at low flux, asymptote at high."""

    @staticmethod
    def _reinhard(rgb, exposure):
        e = exposure * rgb
        return e / (1.0 + e)

    def test_zero_flux_maps_to_zero(self):
        out = self._reinhard(np.zeros((1, 3)), exposure=1.0)
        np.testing.assert_array_equal(out, 0.0)

    def test_low_flux_is_nearly_linear(self):
        """At L << 1/exposure the curve is L · exposure (no compression)."""
        rgb = np.array([[0.001, 0.001, 0.001]])
        out = self._reinhard(rgb, exposure=1.0)
        np.testing.assert_allclose(out, rgb, rtol=0.01)

    def test_high_flux_asymptotes_to_one(self):
        """L → ∞ produces output approaching 1 (no clip required)."""
        rgb = np.array([[1e6, 1e6, 1e6]])
        out = self._reinhard(rgb, exposure=1.0)
        np.testing.assert_array_less(out, 1.0)
        np.testing.assert_array_less(0.999, out)

    def test_monotonic(self):
        """Higher input always yields higher output."""
        a = self._reinhard(np.array([0.5]), 1.0).item()
        b = self._reinhard(np.array([1.0]), 1.0).item()
        c = self._reinhard(np.array([5.0]), 1.0).item()
        assert a < b < c
