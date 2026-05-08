"""
Physics: gravitational N-body + Morse bond forces.

Gravity — symmetric O(N²/2)
----------------------------
Each pair (i, j) is processed exactly once.  Newton's 3rd law gives the force
on j for free from the force on i, halving both flops AND memory vs the full
(N×N) matrix approach.

For each particle i we vectorise over all j > i:
  - diff   = pos[j:] − pos[i]              shape (n−i−1, 3)
  - forces[i]   +=  Σ_j  F_ij
  - forces[j:]  −=  F_ij          (Newton's 3rd law, one scatter)

Numba upgrade path
------------------
  pip install numba
Then swap in the @njit(parallel=True) version (see comments below) for
10-50× speed and N > 5000 at 60 fps.

Bond forces — fully vectorised
--------------------------------
No Python loop over bonds.  Uses the correct Morse potential sign:

    F_on_i = dV/dr · r̂_{i→j}     (positive dV/dr = attractive when r > re)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

def add_gravity(world, forces: np.ndarray):
    """Symmetric O(N²/2) gravity using Newton's 3rd law."""
    n = world.n
    if n < 2:
        return

    G      = world.cfg.physics.gravity_constant
    eps_sq = world.cfg.physics.softening_length ** 2
    pos    = world.positions[:n]
    m      = world.masses[:n]

    for i in range(n - 1):
        diff    = pos[i + 1:] - pos[i]                     # (n−i−1, 3)
        dist_sq = np.einsum('jk,jk->j', diff, diff) + eps_sq
        dist    = np.sqrt(dist_sq)

        # Force magnitude / distance (avoids separate division for unit vector)
        F_over_d = G * m[i] * m[i + 1:] / (dist_sq * dist)   # (n−i−1,)

        F = F_over_d[:, np.newaxis] * diff  # (n−i−1, 3) — force on i toward each j

        forces[i]      += F.sum(axis=0)     # aggregate force on i
        forces[i + 1:n] -= F               # equal and opposite on each j


# ---------------------------------------------------------------------------
# Bond forces
# ---------------------------------------------------------------------------

def add_bond_forces(world, forces: np.ndarray):
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


# ---------------------------------------------------------------------------
# Numba upgrade (uncomment + pip install numba for 10-50× faster gravity)
# ---------------------------------------------------------------------------
# from numba import njit, prange
#
# @njit(parallel=True, cache=True)
# def _gravity_numba(pos, masses, G, eps_sq, out):
#     n = len(pos)
#     for i in prange(n):
#         fx = fy = fz = 0.0
#         xi, yi, zi, mi = pos[i,0], pos[i,1], pos[i,2], masses[i]
#         for j in range(n):
#             if i == j: continue
#             dx = pos[j,0]-xi; dy = pos[j,1]-yi; dz = pos[j,2]-zi
#             d2 = dx*dx + dy*dy + dz*dz + eps_sq
#             r  = d2**0.5
#             f  = G * mi * masses[j] / (d2 * r)
#             fx += f*dx; fy += f*dy; fz += f*dz
#         out[i,0] += fx; out[i,1] += fy; out[i,2] += fz
#
# def add_gravity(world, forces):
#     if world.n < 2: return
#     _gravity_numba(world.positions[:world.n], world.masses[:world.n],
#                    world.cfg.physics.gravity_constant,
#                    world.cfg.physics.softening_length**2, forces)
