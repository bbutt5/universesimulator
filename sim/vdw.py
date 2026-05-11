"""
Van der Waals — weak attractive force between cold non-bonded atoms.

Real atoms feel a long-range induced-dipole attraction. Without it, inert
gases (He, Ne, Ar) and any non-bonded neutrals just drift past each other
forever — there is no mechanism to condense them. With it, cold dense
clouds collapse into liquid and solid phases, giving us the "earth" and
"water" states of matter without hardcoding them.

Force law:
    F_ij = +k_vdw / r²  ×  r̂_{i→j}   (attractive, magnitude scales as 1/r²)

Inhibitors (force is skipped) when:
  - r is inside the covalent regime (handled by bonds / thermal pressure)
  - the pair already has a covalent bond
  - either particle is ionised (plasma)

The 1/r² scaling deliberately mirrors thermal_pressure so the two compete
directly:
  cold pair  →  pressure ≈ 0, VdW pulls them together  → condensation
  hot pair   →  pressure dominates, atoms repel        → vapourisation
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS_LIST
from sim.spatial import build_grid, neighbors


def add_vdw_forces(world, forces: np.ndarray) -> None:
    cfg = getattr(world.cfg, 'thermal', None)
    if cfg is None:
        return

    k_vdw  = float(getattr(cfg, 'vdw_strength', 0.0))
    cutoff = float(getattr(cfg, 'vdw_cutoff',   0.0))
    if k_vdw <= 0.0 or cutoff <= 0.0 or world.n < 2:
        return

    n         = world.n
    pos       = world.positions[:n]
    ionized   = world.ionized[:n]
    cutoff_sq = cutoff * cutoff

    bonded = {(min(b.i, b.j), max(b.i, b.j)) for b in world.bonds}

    grid = build_grid(pos, cutoff)

    for i in range(n):
        if ionized[i]:
            continue
        elem_i = ELEMENTS_LIST[world.elem_ids[i]]
        rc_i   = elem_i.covalent_radius

        for j in neighbors(i, pos, grid, cutoff):
            if j <= i or ionized[j]:
                continue
            if (i, j) in bonded:
                continue

            elem_j = ELEMENTS_LIST[world.elem_ids[j]]
            r_min  = rc_i + elem_j.covalent_radius      # covalent regime

            dr   = pos[j] - pos[i]
            r_sq = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq or r_sq < r_min * r_min:
                continue

            r     = np.sqrt(r_sq)
            unit  = dr / r
            F_mag = k_vdw / r_sq

            forces[i] += F_mag * unit       # pull i toward j
            forces[j] -= F_mag * unit       # Newton's 3rd
