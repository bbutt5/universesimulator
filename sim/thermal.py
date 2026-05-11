"""
Thermal pressure — prevents complete gravitational collapse.

Each close particle pair contributes a short-range repulsive force
proportional to their relative kinetic energy (a local temperature proxy).
This mimics ideal-gas pressure and creates hydrostatic equilibrium in
dense, hot regions such as stellar cores.

The coupling is purely emergent:
  - Cold dust cloud : relative KE ≈ 0  →  pressure ≈ 0  →  gravity wins  →  collapse
  - Hot stellar core: high relative KE  →  strong pressure  →  stable equilibrium
  - Fusion kick raises local KE  →  raises pressure  →  opposes over-compression

Force law (Newton's 3rd law pair):
    F_ij = k_p × μ|Δv|² / r²  ×  r̂_{i→j}   (repulsive on i, equal/opposite on j)

where μ = reduced mass, so the pressure scales with local kinetic temperature
and is independent of the arbitrary choice of reference frame.
"""

from __future__ import annotations
import numpy as np

from sim.spatial import build_grid, neighbors


def add_thermal_pressure(world, forces: np.ndarray) -> None:
    """
    Add short-range thermal pressure forces for all particle pairs within
    pressure_cutoff.  Requires world.cfg.thermal to be present.
    """
    cfg = getattr(world.cfg, 'thermal', None)
    if cfg is None:
        return

    n = world.n
    if n < 2:
        return

    cutoff    = float(cfg.pressure_cutoff)
    k_p       = float(cfg.pressure_constant)
    plasma_k  = float(getattr(cfg, 'plasma_repulsion_factor', 1.0))
    cutoff_sq = cutoff ** 2

    pos     = world.positions[:n]
    vel     = world.velocities[:n]
    m       = world.masses[:n]
    ionized = world.ionized[:n]

    grid = build_grid(pos, cutoff)

    for i in range(n):
        for j in neighbors(i, pos, grid, cutoff):
            if j <= i:
                continue

            dr   = pos[j] - pos[i]
            r_sq = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue

            r      = np.sqrt(r_sq)
            dv     = vel[i] - vel[j]
            mu     = m[i] * m[j] / (m[i] + m[j])
            rel_ke = 0.5 * mu * float(np.dot(dv, dv))

            if rel_ke < 1e-12:
                continue

            unit  = dr / r
            mult  = plasma_k if (ionized[i] and ionized[j]) else 1.0
            F_vec = (mult * k_p * rel_ke / r_sq) * unit
            forces[i] -= F_vec   # repel i away from j
            forces[j] += F_vec   # repel j away from i
