"""
State classifier — assign each particle to solid / liquid / gas / plasma.

The four classical elements (earth / water / air / fire) emerge purely
from local conditions, with no hardcoded transitions:

  PLASMA  — particle is ionised (very high KE stripped its electrons)
  SOLID   — has at least one covalent bond, OR is densely surrounded by
            cold neighbours (lattice-like)
  LIQUID  — densely surrounded but moving too much for stable bonds
  GAS     — few or no nearby particles

The decision uses local-frame kinetic energy (relative motion to
neighbours) rather than absolute velocity, which is the physical
definition of temperature.
"""

from __future__ import annotations
import numpy as np

from sim.spatial import build_grid, neighbors

STATE_SOLID  = 0
STATE_LIQUID = 1
STATE_GAS    = 2
STATE_PLASMA = 3
STATE_NAMES  = ('solid', 'liquid', 'gas', 'plasma')

_DENSE_NEIGHBOURS  = 6      # ≥ this many close neighbours → condensed phase
_LIQUID_NEIGHBOURS = 3      # ≥ this many → at least liquid; below → gas


def classify(world) -> np.ndarray:
    """Return an int array of state codes for each live particle."""
    n = world.n
    states = np.full(n, STATE_GAS, dtype=np.int32)
    if n == 0:
        return states

    ionized = world.ionized[:n].copy()
    states[ionized] = STATE_PLASMA

    cfg      = getattr(world.cfg, 'thermal', None)
    cutoff   = float(getattr(cfg, 'vdw_cutoff', 200.0)) if cfg else 200.0
    chem_cfg = getattr(world.cfg, 'chemistry', None)
    v_thresh = float(getattr(chem_cfg, 'bond_velocity_threshold', 1000.0)) \
               if chem_cfg else 1000.0
    v_thresh_sq = v_thresh * v_thresh
    cutoff_sq   = cutoff * cutoff

    pos         = world.positions[:n]
    vel         = world.velocities[:n]
    bond_counts = world.bond_counts[:n]

    grid = build_grid(pos, cutoff)

    for i in range(n):
        if ionized[i]:
            continue

        close = 0
        ke_sum = 0.0
        for j in neighbors(i, pos, grid, cutoff):
            if j == i:
                continue
            dr   = pos[j] - pos[i]
            r_sq = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq:
                continue
            close += 1
            dv = vel[i] - vel[j]
            ke_sum += float(np.dot(dv, dv))

        density = close + int(bond_counts[i])
        local_T = ke_sum / max(close, 1)        # mean |Δv|²
        bonded  = bond_counts[i] > 0

        if bonded or (density >= _DENSE_NEIGHBOURS and local_T < v_thresh_sq):
            states[i] = STATE_SOLID
        elif density >= _LIQUID_NEIGHBOURS:
            states[i] = STATE_LIQUID
        else:
            states[i] = STATE_GAS

    return states


def state_counts(world) -> dict[str, int]:
    """Return {name: count} for the four phases."""
    states = classify(world)
    return {name: int(np.count_nonzero(states == code))
            for code, name in enumerate(STATE_NAMES)}
