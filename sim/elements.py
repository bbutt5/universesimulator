"""
Periodic table data for the simulation.

All atomic data here is real measured / standard reference data. Where
values differ across sources, NIST is preferred so a physicist reading
this file can verify each row against the standard databases.

Sources
-------
- Atomic masses (amu):                Atomic Mass Evaluation 2020 (Wang et al.,
                                      Chinese Physics C 45, 030003) — exact
                                      dominant-isotope masses, not natural
                                      abundance averages. Needed for accurate
                                      nuclear Q-values.
- Covalent radii (pm):                Cordero et al. 2008, Dalton Trans.
- Electronegativity (Pauling scale):  Pauling 1932 / Allen revision
- First ionization energies (eV):     NIST Atomic Spectra Database
                                      https://physics.nist.gov/asd
- Homonuclear bond dissociation
  energies D(X-X) (kJ/mol):           CRC Handbook of Chemistry and Physics,
                                      95th ed. (2014), Section 9; Luo, JR
                                      "Comprehensive Handbook of Chemical
                                      Bond Energies" (CRC Press, 2007)
- Static dipole polarisability (Å³)   CRC Handbook 95th ed. §10
                                      ("Atomic and Molecular Polarisabilities")

The covalent radius is used directly as the simulation length unit (pm).

Heteronuclear bond energies are computed (not stored) from Pauling's formula:
    D(A-B) = √(D(A-A) · D(B-B)) + 96 · (χ_A − χ_B)²    [kJ/mol]
which is real chemistry (Atkins, Physical Chemistry; Pauling 1932). The
fusion gate is the classical Coulomb barrier from real Z, A values (see
sim/nuclear.py); the old can_fuse boolean has been removed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    Z: int                          # atomic number (proton count)
    A: int                          # mass number (nucleon count) — explicit so isotopes are unambiguous
    mass: float                     # exact isotope mass (amu, AME 2020)
    covalent_radius: float          # pm (Cordero 2008) — used as SU directly
    electronegativity: float        # Pauling scale (0 = noble gas / not applicable)
    valence_electrons: int          # outer shell electrons
    max_bonds: int                  # max covalent bonds this atom can form
    color: tuple[float, float, float]            # CPK colour (RGB 0-1) for rendering
    bond_dissociation_self: float   # D(X-X) homonuclear bond energy in kJ/mol (CRC Handbook).
                                    # 0 for noble gases (no covalent self-bond).
    ionization_energy: float        # first ionization energy in eV (NIST ASD)
    polarisability: float = 0.0     # static dipole polarisability in Å³ (CRC Handbook)
                                    # Drives the London-dispersion VdW force.


# ---------------------------------------------------------------------------
# Element table — every numerical column is real measured data.
# Ionization energies are NIST Atomic Spectra Database first-IE values.
# ---------------------------------------------------------------------------
ELEMENTS: dict[str, Element] = {
    # Each entry uses the dominant isotope's mass (AME 2020). 'He3' and
    # 'Be8' are nuclear intermediates (transients in the stellar fusion
    # chain); they have no cosmic abundance and only appear as fusion
    # products.
    # symbol  name            Z   A     mass         rc    EN    val bnd  RGB                  D(X-X)/(kJ·mol⁻¹)  IE/eV   α/Å³
    'H':   Element('H',   'Hydrogen',     1,  1,  1.00783,    31, 2.20,  1,  1, (0.90, 0.90, 0.90), 436.0, 13.598,  0.667),
    'D':   Element('D',   'Deuterium',    1,  2,  2.01410,    31, 2.20,  1,  1, (0.75, 0.75, 1.00), 443.0, 13.602,  0.667),
    'He3': Element('He3', 'Helium-3',     2,  3,  3.01603,    28, 0.00,  2,  0, (0.80, 1.00, 0.95),   0.0, 24.586,  0.205),
    'He':  Element('He',  'Helium-4',     2,  4,  4.00260,    28, 0.00,  2,  0, (0.85, 1.00, 1.00),   0.0, 24.587,  0.205),
    'Li':  Element('Li',  'Lithium-7',    3,  7,  7.01600,   128, 0.98,  1,  1, (0.80, 0.50, 1.00), 110.0,  5.392, 24.300),
    'Be8': Element('Be8', 'Beryllium-8',  4,  8,  8.00531,    96, 0.00,  2,  0, (1.00, 1.00, 0.50),   0.0,  9.323,  5.600),
    'Be':  Element('Be',  'Beryllium-9',  4,  9,  9.01218,    96, 1.57,  2,  2, (0.76, 1.00, 0.00), 208.0,  9.323,  5.600),
    'B':   Element('B',   'Boron-11',     5, 11, 11.00931,    84, 2.04,  3,  3, (1.00, 0.71, 0.71), 290.0,  8.298,  3.030),
    'C':   Element('C',   'Carbon-12',    6, 12, 12.00000,    77, 2.55,  4,  4, (0.50, 0.50, 0.50), 348.0, 11.260,  1.760),
    'N':   Element('N',   'Nitrogen-14',  7, 14, 14.00307,    71, 3.04,  5,  3, (0.19, 0.31, 0.97), 167.0, 14.534,  1.100),
    'O':   Element('O',   'Oxygen-16',    8, 16, 15.99491,    66, 3.44,  6,  2, (1.00, 0.13, 0.13), 146.0, 13.618,  0.802),
    'F':   Element('F',   'Fluorine-19',  9, 19, 18.99840,    57, 3.98,  7,  1, (0.56, 0.82, 0.56), 158.0, 17.423,  0.557),
    'Ne':  Element('Ne',  'Neon-20',     10, 20, 19.99244,    58, 0.00,  8,  0, (0.70, 0.89, 0.96),   0.0, 21.565,  0.396),
    'Na':  Element('Na',  'Sodium-23',   11, 23, 22.98977,   166, 0.93,  1,  1, (0.67, 0.36, 0.95),  75.0,  5.139, 24.100),
    'Mg':  Element('Mg',  'Magnesium-24',12, 24, 23.98504,   141, 1.31,  2,  2, (0.54, 1.00, 0.00), 130.0,  7.646, 10.600),
    'Al':  Element('Al',  'Aluminum-27', 13, 27, 26.98154,   121, 1.61,  3,  3, (0.75, 0.65, 0.65), 186.0,  5.986,  6.800),
    'Si':  Element('Si',  'Silicon-28',  14, 28, 27.97693,   111, 1.90,  4,  4, (0.94, 0.78, 0.63), 222.0,  8.152,  5.380),
    'P':   Element('P',   'Phosphorus-31',15, 31, 30.97376,  107, 2.19,  5,  3, (1.00, 0.50, 0.00), 209.0, 10.487,  3.630),
    'S':   Element('S',   'Sulfur-32',   16, 32, 31.97207,   105, 2.58,  6,  2, (1.00, 1.00, 0.19), 226.0, 10.360,  2.900),
    'Cl':  Element('Cl',  'Chlorine-35', 17, 35, 34.96885,   102, 3.16,  7,  1, (0.12, 0.94, 0.12), 242.0, 12.968,  2.180),
    # Ar/Ti are the alpha-chain isotopes (³⁶Ar, ⁴⁴Ti), not the natural-abundance
    # ones (⁴⁰Ar, ⁴⁸Ti). The alpha chain Z+Z, A+A conservation requires it:
    # S-32 + α → ³⁶Ar (Q = +6.64 MeV) and Ca-40 + α → ⁴⁴Ti (Q = +5.13 MeV).
    # ⁴⁴Ti is unstable (T½ = 60 yr) but real stars produce it copiously.
    'Ar':  Element('Ar',  'Argon-36',    18, 36, 35.96755,   106, 0.00,  8,  0, (0.50, 0.82, 0.89),   0.0, 15.760,  1.640),
    'K':   Element('K',   'Potassium-39',19, 39, 38.96371,   203, 0.82,  1,  1, (0.56, 0.25, 0.83),  49.0,  4.341, 43.400),
    'Ca':  Element('Ca',  'Calcium-40',  20, 40, 39.96259,   176, 1.00,  2,  2, (0.24, 1.00, 0.00), 105.0,  6.113, 22.800),
    'Ti':  Element('Ti',  'Titanium-44', 22, 44, 43.95969,   136, 1.54,  4,  6, (0.75, 0.76, 0.78), 158.0,  6.828, 14.600),
    'Fe':  Element('Fe',  'Iron-56',     26, 56, 55.93494,   132, 1.83,  2,  6, (0.88, 0.40, 0.20), 118.0,  7.902,  8.400),
    'Ni':  Element('Ni',  'Nickel-58',   28, 58, 57.93534,   124, 1.91,  2,  6, (0.31, 0.82, 0.31), 207.0,  7.640,  6.800),
}

# ---------------------------------------------------------------------------
# Ordered list + index for fast lookup by integer ID (stored in numpy array)
# ---------------------------------------------------------------------------
ELEMENTS_LIST: list[Element] = list(ELEMENTS.values())
SYMBOL_TO_ID: dict[str, int] = {e.symbol: i for i, e in enumerate(ELEMENTS_LIST)}

# Vectorised lookups indexed by elem_id — used in tight inner loops so the
# per-particle physics checks can stay numpy-vectorised.
IONIZATION_ENERGIES_EV: np.ndarray = np.array(
    [e.ionization_energy for e in ELEMENTS_LIST], dtype=np.float64,
)
ATOMIC_NUMBERS_Z: np.ndarray = np.array(
    [e.Z for e in ELEMENTS_LIST], dtype=np.float64,
)
MASS_NUMBERS_A: np.ndarray = np.array(
    [e.A for e in ELEMENTS_LIST], dtype=np.float64,
)
ISOTOPE_MASSES_AMU: np.ndarray = np.array(
    [e.mass for e in ELEMENTS_LIST], dtype=np.float64,
)

# (Z, A) → Element lookup for the fusion product-search code in
# sim/nuclear.find_fusion_product. Avoids the previous hand-curated
# FUSION_REACTIONS table; products are now found by Z+A conservation
# (with optional β⁺ branches) rather than a fixed routing.
ISOTOPES_BY_ZA: dict[tuple[int, int], Element] = {
    (e.Z, e.A): e for e in ELEMENTS_LIST
}


def get(symbol: str) -> Element | None:
    return ELEMENTS.get(symbol)


def id_of(symbol: str) -> int:
    return SYMBOL_TO_ID[symbol]


# ---------------------------------------------------------------------------
# Pauling bond energy formula
# ---------------------------------------------------------------------------
# Reference: Pauling 1932; Atkins, "Physical Chemistry" §15.6.
#   D(A-B) = √(D(A-A) · D(B-B))  +  96 · (χ_A − χ_B)²       [kJ/mol]
# The first term is the geometric-mean covalent contribution; the second
# is the "ionic resonance" energy from partial charge transfer when the
# electronegativities differ. The constant 96 (kJ/mol per Δχ²) follows
# from Pauling's empirical definition of the electronegativity scale.

# Pauling's electronegativity-to-bond-energy constant (kJ/mol per Δχ²)
PAULING_IONIC_RESONANCE_KJ_PER_MOL = 96.0


def pauling_bond_energy_kjmol(a: Element, b: Element) -> float:
    """Heteronuclear bond dissociation energy in kJ/mol via Pauling's formula.

    Returns 0.0 if either atom has no covalent self-bond (D(X-X) = 0),
    i.e. noble gases.
    """
    d_aa = a.bond_dissociation_self
    d_bb = b.bond_dissociation_self
    if d_aa <= 0.0 or d_bb <= 0.0:
        return 0.0
    geo_mean   = (d_aa * d_bb) ** 0.5
    chi_diff   = a.electronegativity - b.electronegativity
    ionic_term = PAULING_IONIC_RESONANCE_KJ_PER_MOL * chi_diff * chi_diff
    return geo_mean + ionic_term


# ---------------------------------------------------------------------------
# Nuclear fusion product routing
# ---------------------------------------------------------------------------
# The previous hand-curated FUSION_REACTIONS dict has been removed.  The
# actual product of a fusion event is now computed by
# ``sim.nuclear.find_fusion_product`` from real conservation laws:
#
#   * Strict (Z, A) conservation first.
#   * β⁺ branches (Z drops by 1 or 2 via positron emission) considered
#     if the direct product is missing or forbidden by energy
#     conservation (rel_KE + Q ≥ 0).
#   * Among feasible products, the most exothermic Q wins.
#
# The chain that emerges from this for the elements currently in the
# table is essentially the textbook stellar-nucleosynthesis sequence
# (Burbidge, Burbidge, Fowler & Hoyle 1957) plus a few extra channels
# (e.g. C+C → Mg, O+O → S) — which are real reactions observed in
# carbon- and oxygen-burning stages.

def fusion_product(sym1: str, sym2: str) -> str | None:
    """Return the conventional product symbol for sym1 + sym2, or None.

    This is the *routing* — the most-exothermic (Z, A)-conserving product
    that exists in the periodic table, allowing β⁺ branches. Q-value
    sign is **not** gated here; a pair whose direct product is endothermic
    (e.g. He+He → Be-8) still returns its routing symbol, because in real
    stars enough kinetic energy can compensate. The kinetics-aware
    version used by the simulator is
    ``sim.nuclear.find_fusion_product(..., rel_ke=...)``.
    """
    from sim.nuclear import find_fusion_product
    a, b = ELEMENTS[sym1], ELEMENTS[sym2]
    elem, _q = find_fusion_product(
        a.Z, a.A, b.Z, b.A, a.mass, b.mass,
        rel_ke=float('inf'), q_scale=1.0,    # disable energy gate
    )
    return elem.symbol if elem is not None else None
