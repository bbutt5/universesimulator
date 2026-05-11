"""
Physics: gravitational N-body + Morse bond forces.

Gravity has two backends, chosen at import time:
  * **numba** (preferred if installed) — parallel JIT compilation. Each
    thread runs the full inner loop in compiled native code; comfortably
    handles N=5000+ in real time on a laptop.
  * **NumPy** (fallback) — symmetric O(N²/2) using Newton's 3rd law and
    vectorised slicing.  Pure-stdlib path; ~10–50× slower than numba at
    N=1000+.

Both implementations agree to within floating-point noise — verified by
``tests/test_physics_gravity.py``.

Bond forces — fully vectorised
--------------------------------
No Python loop over bonds.  Uses the correct Morse potential sign:

    F_on_i = dV/dr · r̂_{i→j}     (positive dV/dr = attractive when r > re)
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Optional numba JIT backend
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:                                # pragma: no cover
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Gravity — NumPy backend (always available)
# ---------------------------------------------------------------------------

def add_gravity_numpy(world, forces: np.ndarray) -> None:
    """Symmetric O(N²/2) NumPy gravity. Newton's 3rd law applied for free."""
    n = world.n
    if n < 2:
        return

    G      = world.cfg.physics.gravity_constant
    eps_sq = world.cfg.physics.softening_length ** 2
    pos    = world.positions[:n]
    m      = world.masses[:n]

    for i in range(n - 1):
        diff    = pos[i + 1:] - pos[i]
        dist_sq = np.einsum('jk,jk->j', diff, diff) + eps_sq
        dist    = np.sqrt(dist_sq)
        F_over_d = G * m[i] * m[i + 1:] / (dist_sq * dist)
        F = F_over_d[:, np.newaxis] * diff
        forces[i]      += F.sum(axis=0)
        forces[i + 1:n] -= F


# ---------------------------------------------------------------------------
# Gravity — numba backend (compiled, parallel)
# ---------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(parallel=True, cache=True, fastmath=True)
    def _gravity_kernel(pos, masses, G, eps_sq, out):
        n = pos.shape[0]
        for i in prange(n):
            fx = 0.0
            fy = 0.0
            fz = 0.0
            xi = pos[i, 0]
            yi = pos[i, 1]
            zi = pos[i, 2]
            mi = masses[i]
            for j in range(n):
                if i == j:
                    continue
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                dz = pos[j, 2] - zi
                d2 = dx * dx + dy * dy + dz * dz + eps_sq
                r  = d2 ** 0.5
                f  = G * mi * masses[j] / (d2 * r)
                fx += f * dx
                fy += f * dy
                fz += f * dz
            out[i, 0] += fx
            out[i, 1] += fy
            out[i, 2] += fz

    def add_gravity_numba(world, forces: np.ndarray) -> None:
        n = world.n
        if n < 2:
            return
        G      = world.cfg.physics.gravity_constant
        eps_sq = world.cfg.physics.softening_length ** 2
        _gravity_kernel(world.positions[:n], world.masses[:n],
                        float(G), float(eps_sq), forces[:n])

else:                                              # pragma: no cover
    add_gravity_numba = None


# Default dispatch — numba when available, NumPy otherwise.
add_gravity = add_gravity_numba if HAS_NUMBA else add_gravity_numpy


# ---------------------------------------------------------------------------
# Bond forces
# ---------------------------------------------------------------------------

def add_bond_forces(world, forces: np.ndarray) -> None:
    """Vectorised Morse bond forces — no Python loop over bonds."""
    bonds = world.bonds
    if not bonds:
        return

    nb    = len(bonds)
    ii    = np.fromiter((b.i         for b in bonds), dtype=np.int32,   count=nb)
    jj    = np.fromiter((b.j         for b in bonds), dtype=np.int32,   count=nb)
    r_eq  = np.fromiter((b.length_eq for b in bonds), dtype=np.float64, count=nb)
    D_e   = np.fromiter((b.D_e       for b in bonds), dtype=np.float64, count=nb)
    a_arr = np.fromiter((b.a         for b in bonds), dtype=np.float64, count=nb)

    pos    = world.positions
    dr     = pos[jj] - pos[ii]
    r      = np.linalg.norm(dr, axis=1)

    valid  = r > 1e-10
    r_safe = np.where(valid, r, 1.0)

    x      = r_safe - r_eq
    exp_t  = np.exp(-a_arr * x)
    # dV/dr > 0 when r > re  →  force on i points toward j  (attractive) ✓
    dVdr   = 2.0 * D_e * a_arr * (1.0 - exp_t) * exp_t

    unit   = dr / r_safe[:, np.newaxis]
    F_vec  = dVdr[:, np.newaxis] * unit
    F_vec[~valid] = 0.0

    np.add.at(forces, ii,  F_vec)
    np.add.at(forces, jj, -F_vec)
