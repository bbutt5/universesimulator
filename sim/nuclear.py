"""
Nuclear physics — fusion gating via the classical Coulomb barrier.

The energy two nuclei must overcome to fuse is the electrostatic repulsion
between their positively-charged protons. At nuclear contact distance:

    V_C = k_e · Z₁·Z₂·e² / r_nuc

where the contact radius scales with mass number as

    r_nuc = r_0 · (A₁^(1/3) + A₂^(1/3))      r_0 ≈ 1.2 fm  (Krane 1988)

In MeV with the standard combination k_e·e² ≈ 1.44 MeV·fm:

    V_C [MeV] = 1.44 · Z₁ · Z₂ / r_nuc[fm]

Reference values (V_C in MeV, computed from the formula above):
    H + H   →   0.60       Si + Si →  35.6
    He + He →   1.92       Fe + Fe → 105
    C + C   →   8.18

These set the *relative* difficulty of fusing different nuclei, and that
ordering is the important emergent property: hydrogen fuses easiest, the
chain becomes progressively harder, and the climb up to iron is steep.

What this module does NOT include
---------------------------------
Real stellar fusion is dominated by **quantum tunnelling through** the
Coulomb barrier (Gamow peak), not by classical *over-the-barrier*
encounters: solar-core temperatures correspond to ~1 keV CM energy, far
below the 0.6 MeV H+H barrier. Including Gamow requires an additional
unit-conversion calibration (sim_KE → MeV) and a tunnelling-probability
gate. Tracked as a follow-up on issue #11. Without it the simulator's
"fusion regime" sits at higher effective temperatures than a real star.

Sim-units calibration
---------------------
`coulomb_barrier_scale` (in sim KE × fm) is the only remaining knob. Its
physical meaning is `k_e·e² × (sim_KE / energy)` — present only because
the simulation runs in custom time-scaled units. With consistent SI units
it would vanish.
"""

from __future__ import annotations
import numpy as np

# Nuclear-radius scaling constant. Empirical value from electron-scattering
# experiments on stable nuclei. Krane, "Introductory Nuclear Physics" §3.1.
_R0_FM = 1.2


def coulomb_barrier_sim(
    z1: float, a1: float, z2: float, a2: float, scale: float,
) -> float:
    """Classical Coulomb barrier between nuclei (Z₁,A₁) and (Z₂,A₂), in sim units.

        V_C = scale · Z₁·Z₂ / r_nuc[fm]
        r_nuc = r₀ · (A₁^(1/3) + A₂^(1/3)),   r₀ = 1.2 fm

    `scale` is the simulator-specific unit conversion (sim_KE × fm).
    """
    r_nuc_fm = _R0_FM * (a1 ** (1.0 / 3.0) + a2 ** (1.0 / 3.0))
    return scale * z1 * z2 / r_nuc_fm


def coulomb_barriers_matrix(
    z: np.ndarray, a: np.ndarray, scale: float,
) -> np.ndarray:
    """Vectorised pairwise Coulomb barriers for all elements in z, a arrays.

    Returns an NxN symmetric matrix V[i,j] = V_C(z[i], a[i], z[j], a[j])."""
    r_nuc = _R0_FM * (a[:, None] ** (1.0 / 3.0) + a[None, :] ** (1.0 / 3.0))
    return scale * z[:, None] * z[None, :] / r_nuc
