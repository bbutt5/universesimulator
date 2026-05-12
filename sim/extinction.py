"""
Beer-Lambert extinction — cold gas absorbs / scatters light along the line
of sight to the observer.

The Beer-Lambert law states that a beam of flux ``F`` traversing a medium
loses intensity according to

    F_observed = F_emitted · exp(−τ)

where ``τ`` (the optical depth) is the integrated opacity along the path:

    τ = ∫ n(s) · σ ds       (n = number density, σ = cross-section)

For our discrete N-body simulator we approximate the line integral by
summing over individual particles that lie near the line of sight, each
contributing its own optical-depth coefficient weighted by a Gaussian
profile of its perpendicular distance from the ray.

The visible-physics consequence is the missing third illumination
mechanism (after self-emission and reflection):

  · Dark nebulae    — cold dense gas blocks light from emitters behind it
                       (Horsehead, Coalsack, Snake Nebula)
  · Star occlusion  — a foreground gas cloud dims a star behind it
  · Limb darkening  — a stellar limb appears redder/dimmer than its centre

We do *not* model wavelength-dependent extinction (cold gas absorbs blue
more than red — "reddening"); that would require a per-channel τ. The
greyscale Beer-Lambert formulation here is the first-order correct form.
A wavelength-dependent extension is a future refinement (tracked on
issue #12 as a sub-item).

Reference: Rybicki & Lightman, "Radiative Processes in Astrophysics",
Chapter 1.
"""

from __future__ import annotations
import numpy as np


def compute_visibility(
    positions: np.ndarray,                # (N, 3) particle world positions
    optical_depths: np.ndarray,           # (N,)   per-particle τ contribution
    camera_position: np.ndarray,          # (3,)   observer in world coords
    kernel_sigma: float = 30.0,           # perpendicular-distance falloff (SU)
) -> np.ndarray:
    """Per-particle exp(−τ) visibility factor along the line to ``camera_position``.

    For each particle ``i``, the optical depth ``τ_i`` is

        τ_i = Σ_{j ≠ i ; j between cam and i} optical_depths[j] · K(d_perp_j)

    with the perpendicular-distance kernel

        K(d) = exp(− d² / σ²)

    so a particle whose line of sight passes near a dense absorber sees
    its flux substantially reduced; one in a clear sight-line sees no
    extinction.
    """
    n = positions.shape[0]
    if n < 2:
        return np.ones(n, dtype=np.float32)

    # Vectors from camera to each particle
    rel = positions - camera_position[None, :]          # (N, 3)
    norms = np.linalg.norm(rel, axis=1)                 # (N,)
    norms = np.maximum(norms, 1e-6)
    units = rel / norms[:, None]                        # (N, 3)

    # For each pair (i, j): project j's camera-relative vector onto i's view ray.
    # proj[i, j] = (camera→j) · unit_camera→i
    proj = rel @ units.T                                # (N, N)  (rows = j, cols = i)
    proj = proj.T                                       # transpose → rows = i, cols = j

    # j is between camera and i  ⟺  0 < proj[i, j] < |camera→i|
    between = (proj > 0.0) & (proj < norms[:, None])

    # Perpendicular distance² from j to the i-ray:
    #   |camera→j|² = proj[i,j]² + d_perp²
    rel_sq = (rel ** 2).sum(axis=1)                     # (N,)
    perp_sq = np.maximum(rel_sq[None, :] - proj * proj, 0.0)

    # Gaussian kernel
    kernel = np.exp(-perp_sq / (kernel_sigma * kernel_sigma))
    kernel = np.where(between, kernel, 0.0)
    np.fill_diagonal(kernel, 0.0)                       # exclude self

    # Sum optical-depth contributions over blockers
    tau = (kernel * optical_depths[None, :]).sum(axis=1)
    return np.exp(-tau).astype(np.float32)


def per_particle_optical_depth(
    temperatures_K: np.ndarray,           # (N,)
    sizes: np.ndarray,                    # (N,) rendered size — proxy for cross-section
    cold_threshold_K: float = 3000.0,
    base_opacity: float = 0.05,
) -> np.ndarray:
    """How opaque each particle is for Beer-Lambert purposes.

    Only cold particles absorb — hot gas / plasma is largely transparent
    to its own emission band (the gas is in an emission rather than
    absorption regime).  Real interstellar dust absorption coefficients
    peak around 100 K; we approximate with a sigmoid falloff above
    ``cold_threshold_K``.

    The base coefficient is per-particle and scales linearly with the
    rendered cross-section (sizes), giving big bodies and dense clusters
    larger optical depths than individual atoms.
    """
    n = len(temperatures_K)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    # 1.0 when T → 0, falling to ~0 well above cold_threshold
    cold_fraction = 1.0 / (1.0 + (temperatures_K / cold_threshold_K) ** 2)
    return (base_opacity * sizes * cold_fraction).astype(np.float32)
