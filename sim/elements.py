"""
Periodic table data for the simulation.

All atomic data here is real measured / standard reference data. Where
values differ across sources, NIST is preferred so a physicist reading
this file can verify each row against the standard databases.

Sources
-------
- Atomic masses (amu):                IUPAC 2021 standard atomic weights
- Covalent radii (pm):                Cordero et al. 2008, Dalton Trans.
- Electronegativity (Pauling scale):  Pauling 1932 / Allen revision
- First ionization energies (eV):     NIST Atomic Spectra Database
                                      https://physics.nist.gov/asd
- Homonuclear bond dissociation
  energies D(X-X) (kJ/mol):           CRC Handbook of Chemistry and Physics,
                                      95th ed. (2014), Section 9; Luo, JR
                                      "Comprehensive Handbook of Chemical
                                      Bond Energies" (CRC Press, 2007)

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
    mass: float                     # atomic mass (amu, IUPAC 2021) — A inferred as round(mass)
    covalent_radius: float          # pm (Cordero 2008) — used as SU directly
    electronegativity: float        # Pauling scale (0 = noble gas / not applicable)
    valence_electrons: int          # outer shell electrons
    max_bonds: int                  # max covalent bonds this atom can form
    color: tuple[float, float, float]            # CPK colour (RGB 0-1) for rendering
    bond_dissociation_self: float   # D(X-X) homonuclear bond energy in kJ/mol (CRC Handbook).
                                    # 0 for noble gases (no covalent self-bond).
    ionization_energy: float        # first ionization energy in eV (NIST ASD)


# ---------------------------------------------------------------------------
# Element table — every numerical column is real measured data.
# Ionization energies are NIST Atomic Spectra Database first-IE values.
# ---------------------------------------------------------------------------
ELEMENTS: dict[str, Element] = {
    # symbol  name            Z    mass    rc    EN    val bnd  RGB                 D(X-X)/(kJ·mol⁻¹)  IE/eV
    'H':  Element('H',  'Hydrogen',    1,   1.008,  31, 2.20,  1,  1, (0.90, 0.90, 0.90), 436.0, 13.598),
    'D':  Element('D',  'Deuterium',   1,   2.014,  31, 2.20,  1,  1, (0.75, 0.75, 1.00), 443.0, 13.602),
    'He': Element('He', 'Helium',      2,   4.003,  28, 0.00,  2,  0, (0.85, 1.00, 1.00),   0.0, 24.587),
    'Li': Element('Li', 'Lithium',     3,   6.941, 128, 0.98,  1,  1, (0.80, 0.50, 1.00), 110.0,  5.392),
    'Be': Element('Be', 'Beryllium',   4,   9.012,  96, 1.57,  2,  2, (0.76, 1.00, 0.00), 208.0,  9.323),
    'B':  Element('B',  'Boron',       5,  10.811,  84, 2.04,  3,  3, (1.00, 0.71, 0.71), 290.0,  8.298),
    'C':  Element('C',  'Carbon',      6,  12.011,  77, 2.55,  4,  4, (0.50, 0.50, 0.50), 348.0, 11.260),
    'N':  Element('N',  'Nitrogen',    7,  14.007,  71, 3.04,  5,  3, (0.19, 0.31, 0.97), 167.0, 14.534),
    'O':  Element('O',  'Oxygen',      8,  15.999,  66, 3.44,  6,  2, (1.00, 0.13, 0.13), 146.0, 13.618),
    'F':  Element('F',  'Fluorine',    9,  18.998,  57, 3.98,  7,  1, (0.56, 0.82, 0.56), 158.0, 17.423),
    'Ne': Element('Ne', 'Neon',       10,  20.180,  58, 0.00,  8,  0, (0.70, 0.89, 0.96),   0.0, 21.565),
    'Na': Element('Na', 'Sodium',     11,  22.990, 166, 0.93,  1,  1, (0.67, 0.36, 0.95),  75.0,  5.139),
    'Mg': Element('Mg', 'Magnesium',  12,  24.305, 141, 1.31,  2,  2, (0.54, 1.00, 0.00), 130.0,  7.646),
    'Al': Element('Al', 'Aluminum',   13,  26.982, 121, 1.61,  3,  3, (0.75, 0.65, 0.65), 186.0,  5.986),
    'Si': Element('Si', 'Silicon',    14,  28.086, 111, 1.90,  4,  4, (0.94, 0.78, 0.63), 222.0,  8.152),
    'P':  Element('P',  'Phosphorus', 15,  30.974, 107, 2.19,  5,  3, (1.00, 0.50, 0.00), 209.0, 10.487),
    'S':  Element('S',  'Sulfur',     16,  32.060, 105, 2.58,  6,  2, (1.00, 1.00, 0.19), 226.0, 10.360),
    'Cl': Element('Cl', 'Chlorine',   17,  35.453, 102, 3.16,  7,  1, (0.12, 0.94, 0.12), 242.0, 12.968),
    'Ar': Element('Ar', 'Argon',      18,  39.948, 106, 0.00,  8,  0, (0.50, 0.82, 0.89),   0.0, 15.760),
    'K':  Element('K',  'Potassium',  19,  39.098, 203, 0.82,  1,  1, (0.56, 0.25, 0.83),  49.0,  4.341),
    'Ca': Element('Ca', 'Calcium',    20,  40.078, 176, 1.00,  2,  2, (0.24, 1.00, 0.00), 105.0,  6.113),
    'Ti': Element('Ti', 'Titanium',   22,  47.867, 136, 1.54,  4,  6, (0.75, 0.76, 0.78), 158.0,  6.828),
    'Fe': Element('Fe', 'Iron',       26,  55.845, 132, 1.83,  2,  6, (0.88, 0.40, 0.20), 118.0,  7.902),
    'Ni': Element('Ni', 'Nickel',     28,  58.693, 124, 1.91,  2,  6, (0.31, 0.82, 0.31), 207.0,  7.640),
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
    [round(e.mass) for e in ELEMENTS_LIST], dtype=np.float64,
)


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
# Nuclear fusion reaction table (phenomenological — tracked for replacement
# in issue #11 with binding-energy Q-value calculation).
#
# Scientific basis (simplified stellar nucleosynthesis chain):
#   pp-chain:       4H → He-4   (here: H+H → D, D+D → He)
#   Triple-alpha:   3He → C     (He+He → Be-8, Be-8+He → C)
#   Alpha capture:  C+α → O+α → Ne+α → Mg+α → Si+α → S → Ar → Ca → Ti
#   Silicon burning: Si + Si → Fe-56 (endpoint of exothermic fusion)
# ---------------------------------------------------------------------------
FUSION_REACTIONS: dict[frozenset, str] = {
    frozenset({'H',  'H'}):  'D',
    frozenset({'D',  'D'}):  'He',
    frozenset({'H',  'D'}):  'He',
    frozenset({'He', 'He'}): 'Be',
    frozenset({'He', 'Be'}): 'C',
    frozenset({'C',  'He'}): 'O',
    frozenset({'O',  'He'}): 'Ne',
    frozenset({'Ne', 'He'}): 'Mg',
    frozenset({'Mg', 'He'}): 'Si',
    frozenset({'Si', 'He'}): 'S',
    frozenset({'S',  'He'}): 'Ar',
    frozenset({'Ar', 'He'}): 'Ca',
    frozenset({'Ca', 'He'}): 'Ti',
    frozenset({'Si', 'Si'}): 'Fe',
}


def fusion_product(sym1: str, sym2: str) -> str | None:
    """Return product symbol if these two elements can fuse, else None."""
    key = frozenset({sym1, sym2})
    return FUSION_REACTIONS.get(key)
