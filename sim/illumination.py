"""
Inverse-square illumination — gas lit by nearby stars.

Each "hot" particle is treated as a point source emitting thermal radiation
isotropically. Stefan-Boltzmann gives its luminosity:

    L = σ · A · T⁴      (proportional to T⁴ in our normalised units)

The flux received by another particle at distance r from that source is
the standard inverse-square law:

    Φ = L / (4 π r²)

We sum this contribution over all sources to get the total illumination on
each particle, then split it by source colour (real Planck blackbody RGB)
to produce the per-particle "lit by external starlight" colour.

This is the missing physics behind every reflection nebula — the blue cocoon
around the Pleiades, the pink rim around H II regions, the colourful gas
swirling around protostars in star-forming clouds. The light wasn't emitted
by the gas itself; it was emitted by nearby hot stars and scatters off the
gas back toward the observer.

A first-order approximation: we do not ray-march through intervening gas
(no extinction yet — that's a future item). Each pair is treated as an
unoccluded inverse-square path. This is exactly right at low optical
depths and a defensible approximation at moderate ones.
"""

from __future__ import annotations
import numpy as np

# Softening: minimum distance² (sim units²) to avoid divergence when a
# receiver overlaps an emitter. Same role as gravity softening.
_R_SOFTENING_SQ = 1.0


def compute_received_rgb_flux(
    positions: np.ndarray,         # (N, 3)
    temperatures_K: np.ndarray,    # (N,)
    blackbody_rgb_fn,              # callable: T_K array → (N, 3) RGB
    reference_T_K: float,
    emitter_T_threshold_K: float = 2000.0,
) -> np.ndarray:
    """Per-particle total RGB flux received from all hot emitters.

    Returns an (N, 3) array of incoming flux × emitter blackbody RGB,
    summed over every emitter j with T_j > ``emitter_T_threshold_K``.
    The flux from emitter j to receiver i is

        Φ_ij  =  (T_j / T_ref)⁴  /  (4 π r_ij²)

    where (T_j / T_ref)⁴ is the dimensionless Stefan-Boltzmann luminosity.

    Self-pairs are excluded — a particle does not illuminate itself.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    emitter_mask = temperatures_K > emitter_T_threshold_K
    if not emitter_mask.any():
        return np.zeros((n, 3), dtype=np.float32)

    em_idx = np.where(emitter_mask)[0]
    em_pos = positions[em_idx]                           # (M, 3)
    em_T   = temperatures_K[em_idx]                      # (M,)

    # Stefan-Boltzmann: luminosity ∝ T⁴
    em_L   = (em_T / reference_T_K) ** 4                 # (M,) dimensionless
    em_rgb = blackbody_rgb_fn(em_T).astype(np.float32)   # (M, 3)

    # Pairwise squared distances (N, M)
    diff   = positions[:, None, :] - em_pos[None, :, :]
    r_sq   = np.einsum('nmk,nmk->nm', diff, diff)
    r_sq   = np.maximum(r_sq, _R_SOFTENING_SQ)

    # Inverse-square flux at each receiver from each emitter
    flux   = em_L[None, :] / (4.0 * np.pi * r_sq)        # (N, M)

    # Exclude self-illumination (a particle does not illuminate itself)
    if em_idx.size > 0:
        receiver_idx_grid = np.arange(n)[:, None]        # (N, 1)
        is_self           = receiver_idx_grid == em_idx[None, :]
        flux              = np.where(is_self, 0.0, flux)

    # Flux-weighted RGB: each emitter contributes its own Planck colour
    received_rgb = (flux[:, :, None] * em_rgb[None, :, :]).sum(axis=1)  # (N, 3)
    return received_rgb.astype(np.float32)
