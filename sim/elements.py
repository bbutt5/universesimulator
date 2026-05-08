"""
Periodic table data for the simulation.
Covalent radii are in pm — treated as SU (simulation units) directly.
CPK colouring convention throughout.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    Z: int                          # atomic number
    mass: float                     # atomic mass (amu)
    covalent_radius: float          # pm → used as SU
    electronegativity: float        # Pauling scale (0 = noble gas / unknown)
    valence_electrons: int          # outer shell electrons
    max_bonds: int                  # max covalent bonds this atom forms
    color: tuple[float, float, float]  # RGB 0-1 for rendering (CPK)
    de_relative: float              # relative dissociation energy (0 = inert)
    can_fuse: bool = False          # participates in nuclear fusion
    fusion_threshold_factor: float = 1.0  # multiplied by global fusion_ke_threshold


# ---------------------------------------------------------------------------
# Full element table.  Includes all injected elements + all possible fusion
# products so the chemistry module never hits a KeyError.
# ---------------------------------------------------------------------------
ELEMENTS: dict[str, Element] = {
    # symbol  name            Z    mass    rc    EN    val  bonds  RGB                  de_rel  fuse  fuse_factor
    'H':  Element('H',  'Hydrogen',    1,   1.008,  31, 2.20,  1,  1, (0.90, 0.90, 0.90), 1.0,  True,  1.0),
    'D':  Element('D',  'Deuterium',   1,   2.014,  31, 2.20,  1,  1, (0.75, 0.75, 1.00), 1.0,  True,  0.8),
    'He': Element('He', 'Helium',      2,   4.003,  28, 0.00,  2,  0, (0.85, 1.00, 1.00), 0.0,  True,  2.0),
    'Li': Element('Li', 'Lithium',     3,   6.941, 128, 0.98,  1,  1, (0.80, 0.50, 1.00), 0.6,  False, 1.0),
    'Be': Element('Be', 'Beryllium',   4,   9.012,  96, 1.57,  2,  2, (0.76, 1.00, 0.00), 0.7,  True,  3.0),
    'B':  Element('B',  'Boron',       5,  10.811,  84, 2.04,  3,  3, (1.00, 0.71, 0.71), 0.8,  False, 1.0),
    'C':  Element('C',  'Carbon',      6,  12.011,  77, 2.55,  4,  4, (0.50, 0.50, 0.50), 1.5,  False, 1.0),
    'N':  Element('N',  'Nitrogen',    7,  14.007,  71, 3.04,  5,  3, (0.19, 0.31, 0.97), 1.3,  False, 1.0),
    'O':  Element('O',  'Oxygen',      8,  15.999,  66, 3.44,  6,  2, (1.00, 0.13, 0.13), 1.2,  False, 1.0),
    'F':  Element('F',  'Fluorine',    9,  18.998,  57, 3.98,  7,  1, (0.56, 0.82, 0.56), 1.4,  False, 1.0),
    'Ne': Element('Ne', 'Neon',       10,  20.180,  58, 0.00,  8,  0, (0.70, 0.89, 0.96), 0.0,  False, 1.0),
    'Na': Element('Na', 'Sodium',     11,  22.990, 166, 0.93,  1,  1, (0.67, 0.36, 0.95), 0.5,  False, 1.0),
    'Mg': Element('Mg', 'Magnesium',  12,  24.305, 141, 1.31,  2,  2, (0.54, 1.00, 0.00), 0.7,  False, 1.0),
    'Al': Element('Al', 'Aluminum',   13,  26.982, 121, 1.61,  3,  3, (0.75, 0.65, 0.65), 0.8,  False, 1.0),
    'Si': Element('Si', 'Silicon',    14,  28.086, 111, 1.90,  4,  4, (0.94, 0.78, 0.63), 1.2,  False, 1.0),
    'P':  Element('P',  'Phosphorus', 15,  30.974, 107, 2.19,  5,  3, (1.00, 0.50, 0.00), 1.0,  False, 1.0),
    'S':  Element('S',  'Sulfur',     16,  32.060, 105, 2.58,  6,  2, (1.00, 1.00, 0.19), 1.1,  False, 1.0),
    'Cl': Element('Cl', 'Chlorine',   17,  35.453, 102, 3.16,  7,  1, (0.12, 0.94, 0.12), 1.3,  False, 1.0),
    'Ar': Element('Ar', 'Argon',      18,  39.948, 106, 0.00,  8,  0, (0.50, 0.82, 0.89), 0.0,  False, 1.0),
    'K':  Element('K',  'Potassium',  19,  39.098, 203, 0.82,  1,  1, (0.56, 0.25, 0.83), 0.5,  False, 1.0),
    'Ca': Element('Ca', 'Calcium',    20,  40.078, 176, 1.00,  2,  2, (0.24, 1.00, 0.00), 0.6,  False, 1.0),
    'Ti': Element('Ti', 'Titanium',   22,  47.867, 136, 1.54,  4,  6, (0.75, 0.76, 0.78), 0.9,  False, 1.0),
    'Fe': Element('Fe', 'Iron',       26,  55.845, 132, 1.83,  2,  6, (0.88, 0.40, 0.20), 1.1,  False, 1.0),
    'Ni': Element('Ni', 'Nickel',     28,  58.693, 124, 1.91,  2,  6, (0.31, 0.82, 0.31), 1.0,  False, 1.0),
}

# ---------------------------------------------------------------------------
# Ordered list + index for fast lookup by integer ID (stored in numpy array)
# ---------------------------------------------------------------------------
ELEMENTS_LIST: list[Element] = list(ELEMENTS.values())
SYMBOL_TO_ID: dict[str, int] = {e.symbol: i for i, e in enumerate(ELEMENTS_LIST)}


def get(symbol: str) -> Element | None:
    return ELEMENTS.get(symbol)


def id_of(symbol: str) -> int:
    return SYMBOL_TO_ID[symbol]


# ---------------------------------------------------------------------------
# Nuclear fusion reaction table (simplified stellar nucleosynthesis).
# Key: frozenset of two symbols (order-independent).
# Value: product symbol.
#
# Scientific basis:
#   pp-chain:       4H → He-4  (simplified here as H+H → D, D+D → He)
#   Triple-alpha:   3He → C    (He+He → Be*, Be*+He → C)
#   Alpha capture:  C+He→O, O+He→Ne, Ne+He→Mg, Mg+He→Si, Si+He→S …
#   Silicon burning:Si+Si → Fe (end of exothermic fusion chain)
# ---------------------------------------------------------------------------
FUSION_REACTIONS: dict[frozenset, str] = {
    frozenset({'H',  'H'}):  'D',   # proto-deuterium step
    frozenset({'D',  'D'}):  'He',  # D+D → He-4
    frozenset({'H',  'D'}):  'He',  # H+D → He-3 (simplified)
    frozenset({'He', 'He'}): 'Be',  # unstable Be-8 intermediate
    frozenset({'He', 'Be'}): 'C',   # triple-alpha completes
    frozenset({'C',  'He'}): 'O',
    frozenset({'O',  'He'}): 'Ne',
    frozenset({'Ne', 'He'}): 'Mg',
    frozenset({'Mg', 'He'}): 'Si',
    frozenset({'Si', 'He'}): 'S',
    frozenset({'S',  'He'}): 'Ar',
    frozenset({'Ar', 'He'}): 'Ca',
    frozenset({'Ca', 'He'}): 'Ti',
    frozenset({'Si', 'Si'}): 'Fe',  # silicon burning → iron peak (endpoint)
}


def fusion_product(sym1: str, sym2: str) -> str | None:
    """Return product symbol if these two elements can fuse, else None."""
    key = frozenset({sym1, sym2})
    return FUSION_REACTIONS.get(key)
