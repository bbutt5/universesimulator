"""
Phase-4 chemical reactions — electronegativity-driven bond swaps.

The classic textbook example is the halide-displacement reaction

    HCl + F → HF + Cl

Hydrogen's covalent partner can switch from chlorine to fluorine because
the H–F bond is *stronger* than H–Cl: fluorine's higher electronegativity
gives a larger ionic-resonance term in Pauling's formula, so the new
bond is more thermodynamically stable. The system loses potential energy
and gains kinetic — the reaction is exothermic.

Algorithm (per simulation step)
-------------------------------
For each existing covalent bond (i — j), look at every atom k within
``reaction_radius`` of i that is:

    * not already bonded to i
    * has free valence (bond_count[k] < max_bonds)
    * not a noble gas

Compute the bond energy that *would* form between i and k via Pauling:

    D(i-k) = √(D_ii · D_kk)  +  96 · (χ_i − χ_k)²       [kJ/mol]

If D(i-k) > D(i-j), the swap is exothermic with energy gain
``ΔE = D(i-k) − D(i-j)``. The reaction proceeds with probability

    P_swap = 1 − exp(−ΔE / activation_energy_scale)

(Arrhenius-like activation: small gains face a higher rejection rate.)
When accepted: the bond (i, j) is replaced by (i, k); j loses one bond,
k gains one. The released energy is converted into relative kinetic
energy of the (j, k) pair along their connecting axis, distributed
inversely to mass so that total momentum is exactly conserved.

What this is and isn't
----------------------
This is **real chemistry** at the single-bond level — Pauling 1932's
electronegativity-driven bond stability, with the Arrhenius rate model
from chemical kinetics. ΔE is the actual mass-defect energy released.

It does *not* yet model:
    * Multi-bond rearrangements like 2H₂ + O₂ → 2H₂O (needs simultaneous
      bond breaks/forms across multiple molecules)
    * Stereochemistry / molecular geometry (we have no bond angles)
    * Catalytic surfaces (could be added by inflating the swap probability
      near solid bodies)

These are follow-up refinements, tracked on issue #4.
"""

from __future__ import annotations
import math
import numpy as np

from sim.elements import ELEMENTS_LIST, pauling_bond_energy_kjmol
from sim.particle import Bond
from sim.spatial import build_grid, neighbors


def update(world) -> None:
    """Detect and apply favourable single-bond swap reactions this step."""
    cfg = getattr(world.cfg, 'reactions', None)
    if cfg is None:
        return
    if not world.bonds or world.n < 3:
        return

    radius             = float(getattr(cfg, 'reaction_radius',           150.0))
    activation_scale   = float(getattr(cfg, 'activation_energy_scale',   200.0))
    heat_release_scale = float(getattr(cfg, 'heat_release_scale',          0.5))
    rng_seed_step      = int(getattr(cfg, 'rng_seed_step',                 0))

    if radius <= 0.0:
        return

    chem_cfg = getattr(world.cfg, 'chemistry', None)
    if chem_cfg is None:
        return
    energy_scale  = float(getattr(chem_cfg, 'bond_energy_scale',     0.184))
    spring_const  = float(getattr(chem_cfg, 'bond_spring_constant',  0.8))

    # Deterministic RNG keyed off simulation time so step results are
    # reproducible run-to-run. The user can disable by setting rng_seed_step.
    rng = np.random.default_rng(int(world.time * 1e6) + rng_seed_step)

    pos       = world.positions[:world.n]
    elem_ids  = world.elem_ids[:world.n]
    grid      = build_grid(pos, radius)

    # Build a set of currently-bonded pairs for quick membership checks.
    bonded_pairs = {(min(b.i, b.j), max(b.i, b.j)) for b in world.bonds}

    consumed: set[int] = set()
    new_bonds: list[Bond] = []

    for bond_idx, bond in enumerate(world.bonds):
        if bond_idx in consumed:
            continue
        i, j = bond.i, bond.j
        if i >= world.n or j >= world.n:
            continue

        elem_i = ELEMENTS_LIST[elem_ids[i]]
        elem_j = ELEMENTS_LIST[elem_ids[j]]
        d_ij   = pauling_bond_energy_kjmol(elem_i, elem_j)
        if d_ij <= 0.0:
            continue

        # Search for the most-exothermic alternative partner k near i.
        best: tuple[int, float, float] | None = None       # (k, delta_e, d_ik)
        for k in neighbors(i, pos, grid, radius):
            if k in (i, j):
                continue
            pair = (min(i, k), max(i, k))
            if pair in bonded_pairs:
                continue

            elem_k = ELEMENTS_LIST[elem_ids[k]]
            if elem_k.max_bonds == 0 or world.bond_counts[k] >= elem_k.max_bonds:
                continue

            d_ik = pauling_bond_energy_kjmol(elem_i, elem_k)
            delta_e = d_ik - d_ij
            if delta_e > 0.0 and (best is None or delta_e > best[1]):
                best = (k, delta_e, d_ik)

        if best is None:
            continue
        k, delta_e, d_ik = best

        # Arrhenius-like acceptance
        p_swap = 1.0 - math.exp(-delta_e / activation_scale)
        if rng.random() >= p_swap:
            continue

        # ---- Perform the swap ----------------------------------------
        # Break i—j: j loses a bond. k gains one. i is unchanged.
        world.bond_counts[j] = max(0, world.bond_counts[j] - 1)
        world.bond_counts[k] += 1

        r_eq    = elem_i.covalent_radius + ELEMENTS_LIST[elem_ids[k]].covalent_radius
        d_e_sim = energy_scale * d_ik
        new_bonds.append(Bond(i, k, r_eq, d_e_sim, spring_const))
        consumed.add(bond_idx)

        bonded_pairs.discard((min(i, j), max(i, j)))
        bonded_pairs.add(pair)

        # ---- Heat release: ΔE → relative KE of (j, k), momentum-conserving --
        heat_sim = delta_e * heat_release_scale
        v_jk = world.positions[k] - world.positions[j]
        d_jk = float(np.linalg.norm(v_jk))
        if d_jk < 1e-9:
            # Particles coincident — pick a random direction
            r = rng.standard_normal(3); v_jk = r; d_jk = float(np.linalg.norm(r))
            d_jk = max(d_jk, 1e-9)
        unit_jk = v_jk / d_jk

        m_j   = float(world.masses[j])
        m_k   = float(world.masses[k])
        mu_jk = m_j * m_k / (m_j + m_k)
        v_rel = math.sqrt(2.0 * heat_sim / max(mu_jk, 1e-9))

        # Push j and k apart in their CoM frame — total momentum exactly preserved.
        share_k = m_j / (m_j + m_k)
        share_j = m_k / (m_j + m_k)
        world.velocities[j] -= v_rel * share_j * unit_jk
        world.velocities[k] += v_rel * share_k * unit_jk

        world.total_reactions += 1

    # Rebuild the bond list: kept survivors + freshly-formed swap products
    survivors = [b for idx, b in enumerate(world.bonds) if idx not in consumed]
    survivors.extend(new_bonds)
    world.bonds = survivors
