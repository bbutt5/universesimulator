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
  1. They are within 2*(rc_i + rc_j) of each other
  2. Their reduced-mass relative kinetic energy ½μv_rel² exceeds the
     classical Coulomb barrier V_C = scale·Z₁·Z₂ / (r₀·(A₁^⅓+A₂^⅓))
     (see sim/nuclear.py).  Gamow tunnelling is not yet modelled.
  3. A reaction product exists in FUSION_REACTIONS (phenomenological
     lookup — to be replaced by Q-value calculation in task 4 of #11).

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

from sim.elements import (
    ELEMENTS, ELEMENTS_LIST, SYMBOL_TO_ID,
    ATOMIC_NUMBERS_Z, MASS_NUMBERS_A,
    pauling_bond_energy_kjmol,
)
from sim.nuclear import coulomb_barriers_matrix, find_fusion_product, gamow_factor_sim
from sim.particle import Bond
from sim.spatial import build_grid, neighbors

# Bond-order multipliers on the single-bond dissociation energy.
# Real chemistry: D(C=C)/D(C-C) ≈ 1.77; D(C≡C)/D(C-C) ≈ 2.42 (CRC Handbook).
# These are pair-averaged values; individual pairs vary (D(N≡N)/D(N-N) ≈ 5.6
# because N-N single bond is unusually weak), but the multipliers below are
# representative within ~20% for typical organic / inorganic combinations.
_BOND_ORDER_D_FACTOR = {1: 1.0, 2: 1.8, 3: 2.4}


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
    ionized   = world.ionized
    threshold = cfg.bond_break_factor

    keep = []
    for bond in world.bonds:
        i, j   = bond.i, bond.j
        order  = bond.order                 # 1 / 2 / 3 — free up matching valence
        # Ionisation strips electrons → covalent bond cannot survive
        if ionized[i] or ionized[j]:
            world.bond_counts[i] = max(0, world.bond_counts[i] - order)
            world.bond_counts[j] = max(0, world.bond_counts[j] - order)
            continue
        r = float(np.linalg.norm(pos[j] - pos[i]))
        if r > threshold * bond.length_eq:
            world.bond_counts[i] = max(0, world.bond_counts[i] - order)
            world.bond_counts[j] = max(0, world.bond_counts[j] - order)
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

    # Pauling formula → kJ/mol → sim energy via bond_energy_scale (issue #11
    # — this remaining scale is just a unit conversion, real chemistry sits
    # in pauling_bond_energy_kjmol).
    E_scale  = cfg.bond_energy_scale
    k        = cfg.bond_spring_constant
    v_thresh = cfg.bond_velocity_threshold

    ionized = world.ionized

    for i in range(world.n):
        elem_i = ELEMENTS_LIST[world.elem_ids[i]]
        if elem_i.max_bonds == 0 or bond_counts[i] >= elem_i.max_bonds:
            continue
        if ionized[i]:
            continue

        for j in neighbors(i, pos, grid, cell_size):
            if j <= i:
                continue
            key = (i, j)
            if key in bonded:
                continue
            if ionized[j]:
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

            # Bond order: take as much as both atoms can support, up to triple.
            free_i = elem_i.max_bonds - bond_counts[i]
            free_j = elem_j.max_bonds - bond_counts[j]
            order  = min(free_i, free_j, 3)
            if order <= 0:
                continue

            # Pauling: D(A-B) = √(D_AA · D_BB) + 96·(χ_A − χ_B)²  [kJ/mol]
            # Then scale by bond-order factor (real D(C=C)/D(C-C) ≈ 1.77 etc.)
            D_kjmol_single = pauling_bond_energy_kjmol(elem_i, elem_j)
            D_e = E_scale * D_kjmol_single * _BOND_ORDER_D_FACTOR[order]
            if D_e < 1e-12:
                continue

            bond = Bond(i, j, r_eq, D_e, k, order=order)
            world.bonds.append(bond)
            world.bond_counts[i] += order
            world.bond_counts[j] += order
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
      1. Every pair of nearby particles is a fusion candidate (no boolean
         flag) — what matters is whether they clear the classical Coulomb
         barrier in the CM frame, V_C = scale·Z₁·Z₂ / (r₀·(A₁^⅓+A₂^⅓)).
         (Gamow tunnelling is not yet modelled — tracked on issue #11.)
      2. Pairs that pass the barrier check are sorted by excess KE and
         processed greedily (each particle fuses at most once per step).
      3. Product symbol is looked up in FUSION_REACTIONS — still
         phenomenological until task 4 derives it from Q-values.
      4. After structural changes, radiation kicks are applied to survivors.
    """
    if not world.cfg.chemistry.fusion_enabled or world.n < 2:
        return

    n   = world.n
    cfg = world.cfg.chemistry

    # Per-particle nuclear data (real Z, A from the periodic table)
    e_ids  = world.elem_ids[:n]
    z_all  = ATOMIC_NUMBERS_Z[e_ids]
    a_all  = MASS_NUMBERS_A[e_ids]

    pos_f  = world.positions[:n]
    vel_f  = world.velocities[:n]
    mass_f = world.masses[:n]
    rc_f   = np.array([ELEMENTS_LIST[e_ids[k]].covalent_radius for k in range(n)])

    diff    = pos_f[:, np.newaxis, :] - pos_f[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)

    max_dist_sq = (2.0 * (rc_f[:, np.newaxis] + rc_f[np.newaxis, :])) ** 2
    close       = np.triu(dist_sq <= max_dist_sq, k=1)
    pi, pj      = np.where(close)
    if len(pi) == 0:
        return

    rel_vel = vel_f[pi] - vel_f[pj]
    mu      = mass_f[pi] * mass_f[pj] / (mass_f[pi] + mass_f[pj])
    rel_ke  = 0.5 * mu * np.einsum('ij,ij->i', rel_vel, rel_vel)

    # Classical Coulomb barrier per candidate pair (sim units).
    # Form: V_C = scale·Z₁·Z₂ / (r₀·(A₁^⅓+A₂^⅓))
    coulomb_scale = float(getattr(cfg, 'coulomb_barrier_scale', 0.0))
    if coulomb_scale <= 0.0:
        return
    barriers = coulomb_barriers_matrix(z_all, a_all, coulomb_scale)
    v_c      = barriers[pi, pj]

    # Two regimes:
    #   1. Above the classical Coulomb barrier — fusion proceeds with P=1.
    #   2. Below the barrier — fusion proceeds via Gamow quantum tunnelling
    #      with probability ∝ exp(−√(E_G/E_cm)).  Stellar fusion is
    #      dominated by this regime (the Sun's core is ~1.5 keV, the H+H
    #      barrier is 0.6 MeV).
    sim_to_mev = float(getattr(cfg, 'gamow_sim_to_mev', 0.0))
    tunnel_strength = float(getattr(cfg, 'gamow_strength', 1.0))

    candidate_mask = rel_ke >= v_c
    if sim_to_mev > 0.0 and tunnel_strength > 0.0:
        # Stochastic accept for sub-barrier pairs (Gamow tunnelling)
        rng = np.random.default_rng(int(world.time * 1e6) + 1)
        for h in np.where(rel_ke < v_c)[0]:
            i_local = pi[h]
            j_local = pj[h]
            p_tunnel = gamow_factor_sim(
                z_all[i_local], a_all[i_local],
                z_all[j_local], a_all[j_local],
                float(rel_ke[h]), sim_to_mev,
            )
            if rng.random() < tunnel_strength * p_tunnel:
                candidate_mask[h] = True

    over_barrier = np.where(candidate_mask)[0]
    if len(over_barrier) == 0:
        return

    # Process most-energetic pairs first so the biggest fusions happen
    over_barrier = over_barrier[np.argsort(-rel_ke[over_barrier])]

    used:   set[int] = set()
    events: list[tuple[int, int, str]] = []

    # Q-value unit conversion (amu → sim KE) for energy-conservation gate.
    q_scale_sim_per_amu = float(getattr(cfg, 'q_value_scale', 1.0))

    for h in over_barrier:
        i = int(pi[h])
        j = int(pj[h])
        if i in used or j in used:
            continue
        elem_i = ELEMENTS_LIST[world.elem_ids[i]]
        elem_j = ELEMENTS_LIST[world.elem_ids[j]]

        # Emergent product routing: find the (Z, A)-conserving product with
        # the largest Q that satisfies rel_KE + Q ≥ 0. β⁺ branches (Z-1, Z-2)
        # are allowed for reactions like H+H → D and Si+Si → Fe that emit
        # positrons in nature. No FUSION_REACTIONS lookup table — the product
        # falls out of conservation laws and measured isotope masses.
        prod_elem, _q_amu = find_fusion_product(
            elem_i.Z, elem_i.A, elem_j.Z, elem_j.A,
            elem_i.mass, elem_j.mass,
            rel_ke=float(rel_ke[h]),
            q_scale=q_scale_sim_per_amu,
        )
        if prod_elem is None:
            continue   # no allowed product

        events.append((i, j, prod_elem.symbol))
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
        # Composition: fusion is a nuclear rebirth — the product is *pure*
        # of its element, with mass equal to the conserved m_total (which
        # differs slightly from prod_elem.mass by the small Q-value defect).
        prod_id = SYMBOL_TO_ID[product_sym]
        world.composition[new_idx, :]       = 0.0
        world.composition[new_idx, prod_id] = m_total

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
