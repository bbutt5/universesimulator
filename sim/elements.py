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

The covalent radius is used directly as the simulation length unit (pm).

Phenomenological fields still present (tracked for removal in issue #11):
  - de_relative              → to be replaced by Pauling bond-energy formula
  - can_fuse / fusion_threshold_factor → to be replaced by Coulomb-barrier
                                          tunnelling calculation
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    Z: int                          # atomic number
    mass: float                     # atomic mass (amu, IUPAC 2021)
    covalent_radius: float          # pm (Cordero 2008) — used as SU directly
    electronegativity: float        # Pauling scale (0 = noble gas / not applicable)
    valence_electrons: int          # outer shell electrons
    max_bonds: int                  # max covalent bonds this atom can form
    color: tuple[float, float, float]            # CPK colour (RGB 0-1) for rendering
    de_relative: float              # phenomenological dissociation-energy scale (issue #11)
    ionization_energy: float        # first ionization energy in eV (NIST ASD)
    can_fuse: bool = False          # participates in fusion (phenomenological — issue #11)
    fusion_threshold_factor: float = 1.0


# ---------------------------------------------------------------------------
# Element table — every numerical column is real measured data.
# Ionization energies are NIST Atomic Spectra Database first-IE values.
# ---------------------------------------------------------------------------
ELEMENTS: dict[str, Element] = {
    # symbol  name            Z    mass    rc    EN    val bnd  RGB                 de_rel  IE/eV   fuse  ff
    'H':  Element('H',  'Hydrogen',    1,   1.008,  31, 2.20,  1,  1, (0.90, 0.90, 0.90), 1.0,  13.598,  True,  1.0),
    'D':  Element('D',  'Deuterium',   1,   2.014,  31, 2.20,  1,  1, (0.75, 0.75, 1.00), 1.0,  13.602,  True,  0.8),
    'He': Element('He', 'Helium',      2,   4.003,  28, 0.00,  2,  0, (0.85, 1.00, 1.00), 0.0,  24.587,  True,  2.0),
    'Li': Element('Li', 'Lithium',     3,   6.941, 128, 0.98,  1,  1, (0.80, 0.50, 1.00), 0.6,   5.392, False, 1.0),
    'Be': Element('Be', 'Beryllium',   4,   9.012,  96, 1.57,  2,  2, (0.76, 1.00, 0.00), 0.7,   9.323,  True, 3.0),
    'B':  Element('B',  'Boron',       5,  10.811,  84, 2.04,  3,  3, (1.00, 0.71, 0.71), 0.8,   8.298, False, 1.0),
    'C':  Element('C',  'Carbon',      6,  12.011,  77, 2.55,  4,  4, (0.50, 0.50, 0.50), 1.5,  11.260, False, 1.0),
    'N':  Element('N',  'Nitrogen',    7,  14.007,  71, 3.04,  5,  3, (0.19, 0.31, 0.97), 1.3,  14.534, False, 1.0),
    'O':  Element('O',  'Oxygen',      8,  15.999,  66, 3.44,  6,  2, (1.00, 0.13, 0.13), 1.2,  13.618, False, 1.0),
    'F':  Element('F',  'Fluorine',    9,  18.998,  57, 3.98,  7,  1, (0.56, 0.82, 0.56), 1.4,  17.423, False, 1.0),
    'Ne': Element('Ne', 'Neon',       10,  20.180,  58, 0.00,  8,  0, (0.70, 0.89, 0.96), 0.0,  21.565, False, 1.0),
    'Na': Element('Na', 'Sodium',     11,  22.990, 166, 0.93,  1,  1, (0.67, 0.36, 0.95), 0.5,   5.139, False, 1.0),
    'Mg': Element('Mg', 'Magnesium',  12,  24.305, 141, 1.31,  2,  2, (0.54, 1.00, 0.00), 0.7,   7.646, False, 1.0),
    'Al': Element('Al', 'Aluminum',   13,  26.982, 121, 1.61,  3,  3, (0.75, 0.65, 0.65), 0.8,   5.986, False, 1.0),
    'Si': Element('Si', 'Silicon',    14,  28.086, 111, 1.90,  4,  4, (0.94, 0.78, 0.63), 1.2,   8.152, False, 1.0),
    'P':  Element('P',  'Phosphorus', 15,  30.974, 107, 2.19,  5,  3, (1.00, 0.50, 0.00), 1.0,  10.487, False, 1.0),
    'S':  Element('S',  'Sulfur',     16,  32.060, 105, 2.58,  6,  2, (1.00, 1.00, 0.19), 1.1,  10.360, False, 1.0),
    'Cl': Element('Cl', 'Chlorine',   17,  35.453, 102, 3.16,  7,  1, (0.12, 0.94, 0.12), 1.3,  12.968, False, 1.0),
    'Ar': Element('Ar', 'Argon',      18,  39.948, 106, 0.00,  8,  0, (0.50, 0.82, 0.89), 0.0,  15.760, False, 1.0),
    'K':  Element('K',  'Potassium',  19,  39.098, 203, 0.82,  1,  1, (0.56, 0.25, 0.83), 0.5,   4.341, False, 1.0),
    'Ca': Element('Ca', 'Calcium',    20,  40.078, 176, 1.00,  2,  2, (0.24, 1.00, 0.00), 0.6,   6.113, False, 1.0),
    'Ti': Element('Ti', 'Titanium',   22,  47.867, 136, 1.54,  4,  6, (0.75, 0.76, 0.78), 0.9,   6.828, False, 1.0),
    'Fe': Element('Fe', 'Iron',       26,  55.845, 132, 1.83,  2,  6, (0.88, 0.40, 0.20), 1.1,   7.902, False, 1.0),
    'Ni': Element('Ni', 'Nickel',     28,  58.693, 124, 1.91,  2,  6, (0.31, 0.82, 0.31), 1.0,   7.640, False, 1.0),
}

# ---------------------------------------------------------------------------
# Ordered list + index for fast lookup by integer ID (stored in numpy array)
# ---------------------------------------------------------------------------
ELEMENTS_LIST: list[Element] = list(ELEMENTS.values())
SYMBOL_TO_ID: dict[str, int] = {e.symbol: i for i, e in enumerate(ELEMENTS_LIST)}

# Vectorised lookups indexed by elem_id — used in tight inner loops so the
# per-particle ionization check can stay numpy-vectorised.
IONIZATION_ENERGIES_EV: np.ndarray = np.array(
    [e.ionization_energy for e in ELEMENTS_LIST], dtype=np.float64,
)


def get(symbol: str) -> Element | None:
    return ELEMENTS.get(symbol)


def id_of(symbol: str) -> int:
    return SYMBOL_TO_ID[symbol]


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
