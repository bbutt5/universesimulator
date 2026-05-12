"""
Van der Waals — London-dispersion attraction between cold non-bonded atoms.

The London dispersion force between two neutral atoms comes from
instantaneous-dipole fluctuations. The classic quantum-mechanical result
(F. London, Trans. Faraday Soc. 33, 8, 1937) is:

    U(r)  =  − C₆ / r⁶

with the London coefficient

    C₆  =  (3/2) · α₁ α₂  ·  I₁ · I₂ / (I₁ + I₂)

where α₁, α₂ are the **measured static dipole polarisabilities** of the two
atoms (Å³, CRC Handbook §10) and I₁, I₂ are their first ionisation
energies (eV, NIST ASD). Both are real measured data; the formula is the
textbook London result.

Force on particle i from particle j:

    F_i  =  − dU/dr · ∇_i r  =  (6 C₆ / r⁷) · r̂_{i→j}

i.e. attractive 1/r⁷ falloff — much steeper than the previous
phenomenological 1/r² form. Without it, inert atoms (He, Ne, Ar) and
non-bonded neutrals would drift through each other forever. With it,
cold dense clouds condense into liquid and solid phases (water, earth).

Inhibitors (force is skipped) when:
  - r is inside the covalent regime (handled by bonds / thermal pressure)
  - the pair already has a covalent bond
  - either particle is ionised (plasma — no induced-dipole interaction)

Sim-units calibration
---------------------
``vdw_dispersion_scale`` is the only calibration constant: it converts
the natural-units London C₆ (eV·Å⁶ multiplied by the simulator's length
unit conversion to pm⁶) into the simulator's force units. Same role as
the other unit-conversion scales in the project — present only because
the simulator runs in custom time-scaled units.
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS_LIST
from sim.spatial import build_grid, neighbors


def add_vdw_forces(world, forces: np.ndarray) -> None:
    cfg = getattr(world.cfg, 'thermal', None)
    if cfg is None:
        return

    dispersion_scale = float(getattr(cfg, 'vdw_dispersion_scale', 0.0))
    cutoff           = float(getattr(cfg, 'vdw_cutoff',           0.0))
    # Backwards-compatible fallback: if the new scale isn't configured but
    # the old vdw_strength is, use that — no behaviour change for legacy
    # configs that haven't been updated yet.
    if dispersion_scale <= 0.0:
        legacy_strength = float(getattr(cfg, 'vdw_strength', 0.0))
        if legacy_strength > 0.0 and cutoff > 0.0 and world.n >= 2:
            _add_vdw_legacy(world, forces, legacy_strength, cutoff)
        return
    if cutoff <= 0.0 or world.n < 2:
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
        alpha_i = elem_i.polarisability
        I_i     = elem_i.ionization_energy
        if alpha_i <= 0.0 or I_i <= 0.0:
            continue                       # no induced-dipole interaction

        for j in neighbors(i, pos, grid, cutoff):
            if j <= i or ionized[j]:
                continue
            if (i, j) in bonded:
                continue

            elem_j = ELEMENTS_LIST[world.elem_ids[j]]
            alpha_j = elem_j.polarisability
            I_j     = elem_j.ionization_energy
            if alpha_j <= 0.0 or I_j <= 0.0:
                continue

            r_min  = rc_i + elem_j.covalent_radius     # covalent regime
            dr     = pos[j] - pos[i]
            r_sq   = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq or r_sq < r_min * r_min:
                continue

            r     = np.sqrt(r_sq)
            unit  = dr / r

            # London C₆ coefficient — real formula with measured data
            c6 = 1.5 * alpha_i * alpha_j * I_i * I_j / (I_i + I_j)

            # F = 6 C₆ / r⁷ (attractive, toward j)
            F_mag = dispersion_scale * 6.0 * c6 / (r_sq ** 3 * r)

            forces[i] += F_mag * unit      # attract i toward j
            forces[j] -= F_mag * unit      # Newton's 3rd


def _add_vdw_legacy(world, forces: np.ndarray, k_vdw: float, cutoff: float) -> None:
    """Legacy phenomenological 1/r² form, kept for backward compatibility
    with configs that haven't been migrated to ``vdw_dispersion_scale``."""
    n         = world.n
    pos       = world.positions[:n]
    ionized   = world.ionized[:n]
    cutoff_sq = cutoff * cutoff
    bonded    = {(min(b.i, b.j), max(b.i, b.j)) for b in world.bonds}
    grid      = build_grid(pos, cutoff)

    for i in range(n):
        if ionized[i]:
            continue
        elem_i = ELEMENTS_LIST[world.elem_ids[i]]
        rc_i   = elem_i.covalent_radius
        for j in neighbors(i, pos, grid, cutoff):
            if j <= i or ionized[j] or (i, j) in bonded:
                continue
            elem_j = ELEMENTS_LIST[world.elem_ids[j]]
            r_min  = rc_i + elem_j.covalent_radius
            dr     = pos[j] - pos[i]
            r_sq   = float(np.dot(dr, dr))
            if r_sq >= cutoff_sq or r_sq < r_min * r_min:
                continue
            r     = np.sqrt(r_sq)
            unit  = dr / r
            F_mag = k_vdw / r_sq
            forces[i] += F_mag * unit
            forces[j] -= F_mag * unit
