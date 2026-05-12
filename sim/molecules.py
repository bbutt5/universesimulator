"""
Molecule recognition — connected-component analysis on the bond graph.

A "molecule" is a maximal set of particles connected through covalent bonds.
We compute these on demand by BFS over the world's bond list, then tag each
component with a Hill-ordered formula and (where known) a common name.

This is *labelling*, not new physics: the molecules already exist as bonded
clusters thanks to the chemistry module. This file makes them legible —
"H2O", "CH4", "NH3" — so the viewer can show meaningful names and the HUD
can tick up molecule counts the same way it ticks up bond counts.

Why Hill ordering
-----------------
The Hill system (C first, H second, then alphabetical) is the chemistry
convention used in CAS and IUPAC databases. It's what a chemistry handbook
shows. Using it here means "CH4" and "C2H6O" look the way readers expect.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

from sim.elements import ELEMENTS_LIST


# ---------------------------------------------------------------------------
# Known molecules — common-name lookups, keyed by Hill-ordered formula.
# Add freely; the rest of the code only needs the formula to identify a
# component, the name is a UI nicety.
# ---------------------------------------------------------------------------
KNOWN_MOLECULES: dict[str, str] = {
    # Diatomic homonuclear
    'H2':    'dihydrogen',
    'D2':    'dideuterium',
    'O2':    'dioxygen',
    'N2':    'dinitrogen',
    'F2':    'difluorine',
    'Cl2':   'dichlorine',
    # Diatomic heteronuclear
    'HD':    'hydrogen deuteride',
    'HF':    'hydrogen fluoride',
    'HCl':   'hydrogen chloride',
    'CO':    'carbon monoxide',
    'NO':    'nitric oxide',
    'OH':    'hydroxyl radical',
    'CN':    'cyanide radical',
    # Triatomic
    'H2O':   'water',
    'CO2':   'carbon dioxide',
    'NO2':   'nitrogen dioxide',
    'SO2':   'sulfur dioxide',
    'H2S':   'hydrogen sulfide',
    'HCN':   'hydrogen cyanide',
    'N2O':   'nitrous oxide',
    'O3':    'ozone',
    # Tetra- / pentatomic
    'NH3':   'ammonia',
    'CH4':   'methane',
    'SO3':   'sulfur trioxide',
    # Common organics (just composition — topology not enforced)
    'CH2O':  'formaldehyde',
    'CH4O':  'methanol',
    'C2H6':  'ethane',
    'C2H4':  'ethylene',
    'C2H2':  'acetylene',
    'C2H6O': 'ethanol',
    'C6H6':  'benzene',
    'CO2H2': 'formic acid',
}


@dataclass(frozen=True)
class Molecule:
    """A connected-bond-graph component, tagged with its composition."""
    formula: str                # Hill-ordered (e.g. 'H2O', 'CH4', 'C2H6O')
    name: str | None            # common name if recognised, else None
    indices: tuple[int, ...]    # particle indices belonging to this molecule

    @property
    def size(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Hill ordering — C first, H second, then alphabetical by symbol.
# Hill is the standard chemistry-database convention (CAS, IUPAC), but
# a few traditional names break it: ammonia is written NH₃ (not H₃N),
# phosphine as PH₃, etc.  The overrides table below preserves convention
# for these traditional hydrides; everything else follows Hill.
# ---------------------------------------------------------------------------

def _hill_sort_key(sym: str) -> tuple[int, str]:
    if sym == 'C':
        return (0, sym)
    if sym == 'H' or sym == 'D':           # treat D the same place as H
        return (1, sym)
    return (2, sym)


# Traditional formulas that deviate from strict Hill order.
# Keys are the strict Hill string; values are the conventional rendering.
_CONVENTIONAL_OVERRIDES: dict[str, str] = {
    'H3N':  'NH3',     # ammonia
    'H3P':  'PH3',     # phosphine
    'H3B':  'BH3',     # borane
    'H4Si': 'SiH4',    # silane
    'H3Al': 'AlH3',    # alane
    'H3As': 'AsH3',    # arsine
    'H3Sb': 'SbH3',    # stibine
}


def _formula_from_counts(sym_counts: dict[str, int]) -> str:
    parts: list[str] = []
    for sym in sorted(sym_counts.keys(), key=_hill_sort_key):
        n = sym_counts[sym]
        parts.append(sym if n == 1 else f'{sym}{n}')
    hill = ''.join(parts)
    return _CONVENTIONAL_OVERRIDES.get(hill, hill)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def identify_molecules(world) -> list[Molecule]:
    """Return one Molecule per bonded connected component (≥ 2 atoms).

    Single atoms with no bonds are not returned — they are 'free atoms',
    not molecules. Treatment of monatomic species (noble gases, ionised
    atoms) follows the same rule for the same reason.
    """
    n = world.n
    if n == 0 or not world.bonds:
        return []

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for b in world.bonds:
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)

    visited = bytearray(n)
    molecules: list[Molecule] = []

    for start in range(n):
        if visited[start] or not adj[start]:
            continue
        # Iterative BFS — avoids recursion limits on large bonded chains
        component: list[int] = []
        stack = [start]
        visited[start] = 1
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adj[i]:
                if not visited[j]:
                    visited[j] = 1
                    stack.append(j)

        if len(component) < 2:
            continue

        sym_counts: Counter[str] = Counter(
            ELEMENTS_LIST[world.elem_ids[i]].symbol for i in component
        )
        formula = _formula_from_counts(dict(sym_counts))
        molecules.append(Molecule(
            formula=formula,
            name=KNOWN_MOLECULES.get(formula),
            indices=tuple(sorted(component)),
        ))

    return molecules


def molecule_counts(world) -> dict[str, int]:
    """{formula: count} across all molecules currently in the world."""
    counts: Counter[str] = Counter()
    for m in identify_molecules(world):
        counts[m.formula] += 1
    return dict(counts)


def molecule_of(world, particle_idx: int) -> Molecule | None:
    """Return the Molecule containing this particle, or None if the
    particle is a free atom."""
    for m in identify_molecules(world):
        if particle_idx in m.indices:
            return m
    return None
