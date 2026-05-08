"""
Gravitational accretion — merges gravitationally bound inert particles.

Two non-fusable particles merge when they are:
  1. Within accretion_radius of each other
  2. Gravitationally bound: relative KE < mutual gravitational PE

The merged body inherits the heavier particle's element identity and
the combined mass of both.  Momentum is conserved exactly.

Why only non-fusable elements?
  H, D, He can_fuse=True  →  they fuse instead (handled by chemistry.py)
  C, O, Si, Fe can_fuse=False  →  they accrete into rocky/metallic bodies

This separation naturally produces:
  - Stellar cores from fusable light elements (H/He clouds collapsing and fusing)
  - Rocky planets from heavy refractory elements (Si, Fe, O accreting)
  - No structure is hardcoded; the mass threshold for "planet" vs "star" appearance
    is set in renderer settings only for visualisation.
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS_LIST
from sim.spatial import build_grid, neighbors


def update(world) -> None:
    """Detect and apply all gravitational accretion events this step."""
    cfg = getattr(world.cfg, 'accretion', None)
    if cfg is None:
        return
    if world.n < 2:
        return

    G       = world.cfg.physics.gravity_constant
    cutoff  = float(cfg.accretion_radius)
    cutoff_sq = cutoff ** 2

    n   = world.n
    pos = world.positions[:n]
    vel = world.velocities[:n]
    m   = world.masses[:n]

    # Only inert (non-fusable) particles accrete; fusable ones fuse via chemistry
    can_accrete = np.array(
        [not ELEMENTS_LIST[world.elem_ids[i]].can_fuse for i in range(n)],
        dtype=bool,
    )

    grid = build_grid(pos, cutoff)

    used:   set[int] = set()
    merges: list[tuple[int, int]] = []

    for i in range(n):
        if i in used or not can_accrete[i]:
            continue
        for j in neighbors(i, pos, grid, cutoff):
            if j <= i or j in used or not can_accrete[j]:
                continue

            dr   = pos[j] - pos[i]
            r_sq = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue

            r       = np.sqrt(r_sq)
            dv      = vel[j] - vel[i]
            mu      = m[i] * m[j] / (m[i] + m[j])
            rel_ke  = 0.5 * mu * float(np.dot(dv, dv))
            grav_pe = G * m[i] * m[j] / r   # positive magnitude

            if rel_ke < grav_pe:   # bound: gravitational PE exceeds kinetic energy
                merges.append((i, j))
                used.add(i)
                used.add(j)
                break

    # Apply largest-index-first so swap-with-last removal stays valid
    merges.sort(key=lambda e: max(e[0], e[1]), reverse=True)

    for i, j in merges:
        m_i     = world.masses[i]
        m_j     = world.masses[j]
        m_total = m_i + m_j

        com_pos = (m_i * world.positions[i] + m_j * world.positions[j]) / m_total
        com_vel = (m_i * world.velocities[i] + m_j * world.velocities[j]) / m_total

        # Dominant element: whichever particle carries more mass
        dominant_elem = ELEMENTS_LIST[world.elem_ids[i] if m_i >= m_j else world.elem_ids[j]]

        a, b = (i, j) if i < j else (j, i)
        world.remove_particle(b)
        world.remove_particle(a)

        new_idx = world.add_particle(dominant_elem, com_pos, com_vel)
        world.masses[new_idx] = m_total   # override with accumulated mass
        world.total_accretions += 1
        world.total_injected  -= 1        # undo the add_particle injection counter
