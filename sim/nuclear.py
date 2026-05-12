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

# Lazy import to avoid circular dependency at module load
def _isotopes_by_za():
    from sim.elements import ISOTOPES_BY_ZA
    return ISOTOPES_BY_ZA

# Nuclear-radius scaling constant. Empirical value from electron-scattering
# experiments on stable nuclei. Krane, "Introductory Nuclear Physics" §3.1.
_R0_FM = 1.2

# CODATA: 1 atomic mass unit (amu) in MeV/c²
AMU_TO_MEV = 931.494

# Fine-structure constant (dimensionless, CODATA)
_ALPHA_FS = 1.0 / 137.035999


def gamow_factor(z1: float, a1: float, z2: float, a2: float,
                 rel_ke_mev: float) -> float:
    """Gamow tunnelling probability for two nuclei to fuse below the classical
    Coulomb barrier.

        P_tunnel  ∝  exp(−√(E_G / E_cm))

    where E_G is the Gamow energy

        E_G  =  (2 π α Z₁ Z₂)² · 2 μ c²

    with α the fine-structure constant, μ the reduced mass (in MeV/c²), and
    E_cm the centre-of-mass kinetic energy (in MeV).

    Reference: Clayton, "Principles of Stellar Evolution and Nucleosynthesis"
    §4.3; the standard WKB approximation for nuclear barrier penetration.

    Real stellar fusion is dominated by this mechanism: the Sun's core is at
    ~1.5 keV but the H+H Coulomb barrier is 0.6 MeV (400× higher) — fusion
    proceeds via the long tunnelling tail.
    """
    if rel_ke_mev <= 0.0:
        return 0.0
    # Reduced mass in MeV/c² (1 amu·c² = 931.494 MeV)
    mu_mev = (a1 * a2 / (a1 + a2)) * AMU_TO_MEV
    # Gamow energy E_G = 2π² (α·Z₁·Z₂)² · μc² (Clayton §4.3)
    e_g = 2.0 * (np.pi ** 2) * (_ALPHA_FS * z1 * z2) ** 2 * mu_mev
    # Tunnelling probability ∝ exp(−√(E_G / E_cm))
    return float(np.exp(-np.sqrt(e_g / rel_ke_mev)))


def gamow_factor_sim(z1: float, a1: float, z2: float, a2: float,
                     rel_ke_sim: float, sim_to_mev: float) -> float:
    """Gamow tunnelling probability in simulator units.

    Converts the simulator's reduced-mass KE into MeV using ``sim_to_mev``
    (the only calibration knob, present because the sim runs in custom
    time-scaled units), then evaluates the Gamow factor above.
    """
    if sim_to_mev <= 0.0:
        return 0.0
    return gamow_factor(z1, a1, z2, a2, rel_ke_sim * sim_to_mev)


def fusion_q_amu(m_a: float, m_b: float, m_product: float) -> float:
    """Q-value (released rest-mass energy) of A + B → C, in amu.

        Q = (m_A + m_B - m_C) · c²

    A positive Q means the reaction releases energy (exothermic). With the
    exact AME 2020 isotope masses on each Element, this is the *real*
    Q-value — no semi-empirical mass formula approximation needed.

    Examples (handbook values):
        H + H → D    : Q = +1.44  MeV   (pp-chain step 1, β⁺ branch)
        D + D → He   : Q = +23.85 MeV
        H + D → He3  : Q = +5.49  MeV
        He+He → Be8  : Q = −0.092 MeV   (resonance; needs KE assist)
        Be8+He → C   : Q = +7.37  MeV   (triple-α completion)
        Si + Si → Fe : Q = +17.6  MeV
    """
    return m_a + m_b - m_product


# ---------------------------------------------------------------------------
# Fusion product search — emergent routing, no hand-curated table
# ---------------------------------------------------------------------------
# Strategy:
#   1. Strict Z and A conservation first: search for an element with
#      Z = Z₁+Z₂ and A = A₁+A₂.
#   2. If that product is unavailable (or energetically forbidden), allow
#      β⁺ branches in which 1 or 2 positrons + neutrinos are emitted.
#      Each positron carries away one unit of positive charge, so the
#      product Z drops by ``delta_z`` while A is preserved.
#         delta_z = 1  :  pp-chain step 1 (H + H → D + e⁺ + ν)
#         delta_z = 2  :  silicon burning net (Si + Si → Ni-56 → Fe-56 + 2β⁺)
#   3. Among all feasible products (Z conserved or up to 2 β⁺ branches),
#      pick the one with the largest Q (most exothermic). Reject any
#      where rel_KE + Q < 0 — that violates energy conservation.
#
# This is *emergent routing*: any pair of nuclei in the table can fuse if
# a product exists with the right (Z, A) and the energetics allow. The
# previous hand-curated FUSION_REACTIONS dict has been removed.

def find_fusion_product(
    z1: int, a1: int,
    z2: int, a2: int,
    m_a: float, m_b: float,
    rel_ke: float,
    q_scale: float,
    max_beta_plus: int = 2,
) -> tuple[object, float]:
    """Return (Element, Q_amu) for the best fusion product, or (None, 0.0).

    All arguments use the simulator's units convention: nucleon counts as
    integers, masses in amu, kinetic energy in sim KE units, q_scale
    converts amu → sim KE.

    The "best" product is the one with the largest Q-value that satisfies
    energy conservation (rel_KE + Q · scale ≥ 0).  Direct Z/A conservation
    is preferred (smaller delta_z); β⁺ branches are considered only when
    no direct product exists or the direct product is energetically
    forbidden.
    """
    isotopes = _isotopes_by_za()
    z_total = int(z1 + z2)
    a_total = int(a1 + a2)

    best_q = -float('inf')
    best_elem = None
    best_delta_z = -1

    for delta_z in range(max_beta_plus + 1):
        z_try = z_total - delta_z
        elem  = isotopes.get((z_try, a_total))
        if elem is None:
            continue
        q_amu = m_a + m_b - elem.mass
        # Energy conservation: kinetic energy must compensate any
        # endothermic mass defect
        if rel_ke + q_amu * q_scale < 0.0:
            continue
        # Prefer most exothermic (highest Q). Break ties by smallest delta_z.
        better = (
            q_amu > best_q
            or (q_amu == best_q and (best_delta_z < 0 or delta_z < best_delta_z))
        )
        if better:
            best_q = q_amu
            best_elem = elem
            best_delta_z = delta_z

    if best_elem is None:
        return None, 0.0
    return best_elem, best_q


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
