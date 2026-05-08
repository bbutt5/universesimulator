"""
Chemistry: covalent bond formation/breaking + nuclear fusion + radiation kicks.

Bond formation
--------------
Two atoms form a bond when:
  1. Distance < bond_formation_factor * (rc_i + rc_j)
  2. Neither has reached its max_bonds valence
  3. Relative speed is below bond_velocity_threshold (slow enough to "stick")
  4. They are not already bonded

Bond breaking
-------------
A bond breaks when the current length exceeds bond_break_factor * eq_length.

Nuclear fusion
--------------
Two atoms fuse when:
  1. A reaction product exists in FUSION_REACTIONS
  2. Both atoms have can_fuse=True
  3. Reduced-mass kinetic energy ½μv_rel² > fusion_ke_threshold * threshold_factor
  4. They are within 2*(rc_i + rc_j) of each other

On fusion both reactants are removed and a new product particle is inserted
at the centre-of-mass with momentum-conserved velocity.

Radiation kick
--------------
Each fusion event emits an outward velocity impulse to nearby particles,
falling off as 1/r.  This is the stellar feedback mechanism: fusion heats
the surrounding gas, raises thermal pressure, and opposes further collapse.
The magnitude is controlled by chemistry.radiation_energy_scale.
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS, ELEMENTS_LIST, fusion_product
from sim.particle import Bond
from sim.spatial import build_grid, neighbors


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def update(world) -> None:
    """Called once per physics step after forces & integration."""
    _break_bonds(world)
    _fuse(world)
    _form_bonds(world)


# ------------------------------------------------------------------
# Bond breaking
# ------------------------------------------------------------------

def _break_bonds(world) -> None:
    cfg       = world.cfg.chemistry
    pos       = world.positions
    threshold = cfg.bond_break_factor

    keep = []
    for bond in world.bonds:
        i, j = bond.i, bond.j
        r = float(np.linalg.norm(pos[j] - pos[i]))
        if r > threshold * bond.length_eq:
            world.bond_counts[i] = max(0, world.bond_counts[i] - 1)
            world.bond_counts[j] = max(0, world.bond_counts[j] - 1)
        else:
            keep.append(bond)
    world.bonds = keep


# ------------------------------------------------------------------
# Bond formation
# ------------------------------------------------------------------

def _form_bonds(world) -> None:
    if world.n < 2:
        return

    cfg         = world.cfg.chemistry
    pos         = world.positions[:world.n]
    vel         = world.velocities[:world.n]
    bond_counts = world.bond_counts

    bonded = {(min(b.i, b.j), max(b.i, b.j)) for b in world.bonds}

    # Cell size from elements ACTUALLY in the simulation — avoids O(N²) degenerate case
    # when the global max radius (e.g. K=203 pm) would put every particle in one cell.
    present_ids  = set(int(world.elem_ids[i]) for i in range(world.n))
    bondable_rc  = [ELEMENTS_LIST[eid].covalent_radius
                    for eid in present_ids if ELEMENTS_LIST[eid].max_bonds > 0]
    if not bondable_rc:
        return

    max_rc    = max(bondable_rc)
    cell_size = cfg.bond_formation_factor * 2.0 * max_rc
    grid      = build_grid(pos, cell_size)

    D_e_scale = cfg.bond_de_scale
    k         = cfg.bond_spring_constant
    v_thresh  = cfg.bond_velocity_threshold

    for i in range(world.n):
        elem_i = ELEMENTS_LIST[world.elem_ids[i]]
        if elem_i.max_bonds == 0 or bond_counts[i] >= elem_i.max_bonds:
            continue

        for j in neighbors(i, pos, grid, cell_size):
            if j <= i:
                continue
            key = (i, j)
            if key in bonded:
                continue

            elem_j = ELEMENTS_LIST[world.elem_ids[j]]
            if elem_j.max_bonds == 0 or bond_counts[j] >= elem_j.max_bonds:
                continue

            dr   = pos[j] - pos[i]
            r    = float(np.linalg.norm(dr))
            r_eq = elem_i.covalent_radius + elem_j.covalent_radius
            if r > cfg.bond_formation_factor * r_eq:
                continue

            rel_v = float(np.linalg.norm(vel[i] - vel[j]))
            if rel_v > v_thresh:
                continue

            D_e = D_e_scale * (elem_i.de_relative * elem_j.de_relative) ** 0.5
            if D_e < 1e-12:
                continue

            bond = Bond(i, j, r_eq, D_e, k)
            world.bonds.append(bond)
            world.bond_counts[i] += 1
            world.bond_counts[j] += 1
            world.total_bonds_formed += 1
            bonded.add(key)

            if bond_counts[i] >= elem_i.max_bonds:
                break


# ------------------------------------------------------------------
# Nuclear fusion  (vectorised distance + KE screening)
# ------------------------------------------------------------------

def _fuse(world) -> None:
    """
    Vectorised fusion pass:
      1. NumPy screens all fusable pairs for distance + KE in one shot.
      2. Python loop only over the handful of pairs that actually pass.
      3. After structural changes, radiation kicks applied to survivors.
    """
    if not world.cfg.chemistry.fusion_enabled or world.n < 2:
        return

    n   = world.n
    cfg = world.cfg.chemistry

    can_fuse = np.array([ELEMENTS_LIST[world.elem_ids[i]].can_fuse for i in range(n)])
    fi_all   = np.where(can_fuse)[0]
    nf       = len(fi_all)
    if nf < 2:
        return

    pos_f  = world.positions[fi_all]
    vel_f  = world.velocities[fi_all]
    mass_f = world.masses[fi_all]
    rc_f   = np.array([ELEMENTS_LIST[world.elem_ids[fi_all[k]]].covalent_radius
                       for k in range(nf)])
    tf_f   = np.array([ELEMENTS_LIST[world.elem_ids[fi_all[k]]].fusion_threshold_factor
                       for k in range(nf)])

    diff    = pos_f[:, np.newaxis, :] - pos_f[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)

    max_dist_sq = (2.0 * (rc_f[:, np.newaxis] + rc_f[np.newaxis, :])) ** 2
    close       = np.triu(dist_sq <= max_dist_sq, k=1)
    pi, pj      = np.where(close)
    if len(pi) == 0:
        return

    rel_vel     = vel_f[pi] - vel_f[pj]
    mu          = mass_f[pi] * mass_f[pj] / (mass_f[pi] + mass_f[pj])
    rel_ke      = 0.5 * mu * np.einsum('ij,ij->i', rel_vel, rel_vel)
    ke_thresh   = cfg.fusion_ke_threshold
    thresh_pairs = ke_thresh * (tf_f[pi] + tf_f[pj]) * 0.5
    hot          = np.where(rel_ke >= thresh_pairs)[0]
    if len(hot) == 0:
        return

    hot = hot[np.argsort(-rel_ke[hot])]

    used:   set[int] = set()
    events: list[tuple[int, int, str]] = []

    for h in hot:
        i = int(fi_all[pi[h]])
        j = int(fi_all[pj[h]])
        if i in used or j in used:
            continue
        sym_i = ELEMENTS_LIST[world.elem_ids[i]].symbol
        sym_j = ELEMENTS_LIST[world.elem_ids[j]].symbol
        prod  = fusion_product(sym_i, sym_j)
        if prod is None or prod not in ELEMENTS:
            continue
        events.append((i, j, prod))
        used.add(i)
        used.add(j)

    events.sort(key=lambda e: max(e[0], e[1]), reverse=True)

    fusion_sites: list[np.ndarray] = []

    for i, j, product_sym in events:
        m_i     = world.masses[i]
        m_j     = world.masses[j]
        m_total = m_i + m_j

        com_pos = (m_i * world.positions[i] + m_j * world.positions[j]) / m_total
        com_vel = (m_i * world.velocities[i] + m_j * world.velocities[j]) / m_total

        fusion_sites.append(com_pos.copy())

        a, b = (i, j) if i < j else (j, i)
        world.remove_particle(b)
        world.remove_particle(a)

        prod_elem = ELEMENTS[product_sym]
        new_idx   = world.add_particle(prod_elem, com_pos, com_vel)
        world.total_fusions += 1
        world.masses[new_idx] = m_total   # conserve mass

    if fusion_sites:
        _apply_radiation_kicks(world, fusion_sites)


# ------------------------------------------------------------------
# Radiation kicks (outward impulse from each fusion site)
# ------------------------------------------------------------------

def _apply_radiation_kicks(world, fusion_sites: list[np.ndarray]) -> None:
    """
    Outward velocity impulse from each fusion event, falling off as 1/r.

    Mimics the radiation pressure that makes stellar cores self-regulating:
    more fusion → hotter surroundings → higher thermal pressure → expansion.
    The impulse is applied as a direct velocity change (not a force), so it
    acts instantaneously on this step's particle set.
    """
    cfg    = world.cfg.chemistry
    radius = float(getattr(cfg, 'radiation_radius', 0.0))
    k_rad  = float(getattr(cfg, 'radiation_energy_scale', 0.0))

    if radius <= 0.0 or k_rad <= 0.0 or world.n == 0:
        return

    pos = world.positions[:world.n]   # live slice after structural changes

    for com_pos in fusion_sites:
        r_vecs = pos - com_pos                            # (n, 3)
        dists  = np.linalg.norm(r_vecs, axis=1)          # (n,)
        mask   = (dists > 1.0) & (dists < radius)
        if not np.any(mask):
            continue

        d_safe  = np.where(mask, dists, 1.0)[:, np.newaxis]  # (n, 1)
        delta_v = k_rad * r_vecs / d_safe ** 2                # (n, 3)  magnitude ∝ 1/r
        delta_v[~mask] = 0.0

        world.velocities[:world.n] += delta_v
