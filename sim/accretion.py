"""
Gravitational accretion — merges gravitationally bound particles.

A pair merges when they are:
  1. Within accretion_radius of each other
  2. Gravitationally bound: relative KE < mutual gravitational PE
  3. *Not* energetic enough to clear their Coulomb barrier this step
     (otherwise they will fuse, not accrete)

Condition (3) replaces the old ``can_fuse`` boolean.  It is real physics:
a pair of nuclei that clears its V_C produces a fusion event; a bound
pair below V_C drops into a stable gravitationally-bound configuration.
The split between "stars" (light, hot, fusion-dominated regions) and
"planets" (heavy, cold, accretion-dominated regions) is therefore
emergent — it follows from the Z₁·Z₂ scaling of the Coulomb barrier
and the local kinetic-energy distribution.

The merged body inherits the heavier particle's element identity and the
combined mass.  Momentum is conserved exactly.
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS_LIST, ATOMIC_NUMBERS_Z, MASS_NUMBERS_A
from sim.nuclear import coulomb_barrier_sim
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
    if cutoff <= 0.0:
        return            # accretion disabled
    cutoff_sq = cutoff ** 2

    n   = world.n
    pos = world.positions[:n]
    vel = world.velocities[:n]
    m   = world.masses[:n]

    # Pull the Coulomb-barrier scale so accretion can defer to fusion when
    # the pair clears its V_C this step.  If fusion is disabled or no scale
    # is set, accretion proceeds purely on the binding-energy criterion.
    chem_cfg = getattr(world.cfg, 'chemistry', None)
    fusion_on    = bool(getattr(chem_cfg, 'fusion_enabled', False)) if chem_cfg else False
    coulomb_scale = float(getattr(chem_cfg, 'coulomb_barrier_scale', 0.0)) if chem_cfg else 0.0
    z_all = ATOMIC_NUMBERS_Z[world.elem_ids[:n]]
    a_all = MASS_NUMBERS_A[world.elem_ids[:n]]

    grid = build_grid(pos, cutoff)

    used:   set[int] = set()
    merges: list[tuple[int, int]] = []

    for i in range(n):
        if i in used:
            continue
        for j in neighbors(i, pos, grid, cutoff):
            if j <= i or j in used:
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

            if rel_ke >= grav_pe:
                continue   # unbound — will fly apart, no merge

            # Coulomb-barrier check: if this pair could fuse this step,
            # defer to chemistry (no merge). Real physics — the same pair
            # cannot be simultaneously fusing and accreting.
            if fusion_on and coulomb_scale > 0.0:
                v_c = coulomb_barrier_sim(
                    z_all[i], a_all[i], z_all[j], a_all[j], coulomb_scale,
                )
                if rel_ke >= v_c:
                    continue   # fusion regime — let _fuse handle it

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
