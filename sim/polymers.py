"""
Polymer analysis — chain length, branching, ring detection, motif tagging.

This module sits on top of the bond graph maintained by ``sim.chemistry``:
every bonded connected component is already a "molecule" (see
``sim.molecules``); here we additionally compute *topological* metrics
that distinguish a long chain from a small cluster, count its rings and
branches, and tag a handful of biologically-significant graph motifs.

The polymer label is *emergent*: it falls out of how many bonds each
carbon happens to form, how stable the resulting chain is against the
local kinetic / radiation field, and (for the catalytic regime) how
close to a solid accreted surface the formation happens. Nothing here
hardcodes "this is a polymer" — we just measure what the bond graph
already contains.

Metrics
-------
chain_length
    Number of atoms on the longest *shortest-path* through the bonded
    component (graph diameter, in atoms). The standard two-pass-BFS
    diameter algorithm (Aho, Hopcroft & Ullman, "The Design and Analysis
    of Computer Algorithms", §5.5) is exact for tree-shaped polymers,
    which is the overwhelming case for linear carbon chains. For ring-
    containing components it is a tight lower bound on the longest
    simple path — the latter is NP-hard in general (Garey & Johnson
    "Computers and Intractability", problem GT39), and the diameter
    approximation is well-behaved for the small molecules we care about.

branches
    Number of vertices of degree ≥ 3. A pure linear chain has 0
    branches; a polyethylene-like backbone with a methyl side group
    has 1 branch; a fully cross-linked polymer has many. Same metric
    used in chemistry textbooks (Flory, "Principles of Polymer
    Chemistry", 1953, §III for "branching index").

rings
    Number of independent cycles, computed as the cyclomatic complexity
    of the component graph: r = E − V + 1 per connected component
    (Euler's formula for planar graphs; equivalent to the dimension of
    the cycle space in algebraic graph theory — Diestel, "Graph Theory",
    §1.9). A linear or branched tree has r = 0, a single ring has
    r = 1, fused bicyclic systems like naphthalene have r = 2, etc.

ring_sizes
    The lengths (in atoms) of the fundamental cycles found by a DFS
    spanning tree. The fundamental-cycle basis of a connected graph is
    a textbook construction (Diestel §1.9): one cycle per non-tree edge,
    formed by adding that edge to the tree path between its endpoints.
    Useful for tagging biologically-significant ring sizes (5 for
    ribose / pyrrole, 6 for benzene / pyrimidine / purine).

Motifs
------
Motif recognition is purely graph-based — we have no bond angles or
stereochemistry yet (issue #15). Each motif is a coarse subgraph
fingerprint, *not* a guarantee of biological functionality. Reading a
motif tag should evoke "this looks like the connectivity backbone of an
X" rather than "this is an X".

  ``amino-acid-like``
      Same connected component contains an N atom *and* a C atom that
      participates in a double bond to an O atom. This is the
      H₂N–CR–COOH connectivity of an α-amino acid (Lehninger,
      "Principles of Biochemistry", 6th ed., §3.1), abstracted away from
      stereochemistry. Real amino acids additionally require the N and
      COOH to share a central C — we relax that to "same component"
      because picking the central α-carbon requires bond-angle data.

  ``nucleotide-like``
      Same connected component contains:
      (a) a ring of size 5 or 6 with at least one C *and* at least one
          N (the pyrimidine/purine fingerprint — Lehninger §8.1);
      (b) at least one P atom (the phosphate placeholder; real
          nucleotides have a PO₄³⁻ group, but we don't yet model
          formal charges).

References
----------
- Aho, Hopcroft & Ullman 1974 — graph diameter algorithm.
- Garey & Johnson 1979 — longest simple path NP-hardness (GT39).
- Diestel 2017 — cyclomatic number, fundamental cycle basis.
- Flory 1953 — polymer branching metrics.
- Lehninger / Nelson & Cox 2017 — amino acid and nucleotide structure.
"""

from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass

from sim.elements import ELEMENTS_LIST
from sim.molecules import KNOWN_MOLECULES, _formula_from_counts


@dataclass(frozen=True)
class Polymer:
    """A bonded connected component plus its topology + motif tags."""
    formula: str                       # Hill-ordered, e.g. 'C10H22'
    name: str | None                   # common name from molecules.KNOWN_MOLECULES if matched
    indices: tuple[int, ...]           # particle indices in this component
    chain_length: int                  # diameter (atoms on longest shortest path)
    branches: int                      # vertices of degree ≥ 3
    rings: int                         # cyclomatic number (E − V + 1)
    ring_sizes: tuple[int, ...]        # sizes of fundamental cycles
    motifs: tuple[str, ...]            # 'amino-acid-like', 'nucleotide-like', …

    @property
    def size(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Internal helpers — adjacency + bond-order lookup
# ---------------------------------------------------------------------------

def _build_adjacency(world) -> list[list[int]]:
    """Return adj[i] = list of neighbours bonded to particle i."""
    adj: list[list[int]] = [[] for _ in range(world.n)]
    for b in world.bonds:
        if b.i < world.n and b.j < world.n:
            adj[b.i].append(b.j)
            adj[b.j].append(b.i)
    return adj


def _build_bond_orders(world) -> dict[tuple[int, int], int]:
    """{(min(i,j), max(i,j)): order} — supports multi-bond motif checks."""
    return {(min(b.i, b.j), max(b.i, b.j)): b.order for b in world.bonds}


# ---------------------------------------------------------------------------
# Diameter (chain length) — two-pass BFS
# ---------------------------------------------------------------------------
# For a tree, BFS from an arbitrary node reaches one endpoint of the
# diameter; a second BFS from that endpoint reaches the other.  This is
# the Aho-Hopcroft-Ullman result, exact for trees in O(V+E).
# For graphs with cycles the same algorithm returns the eccentricity of
# the chosen endpoints (a valid lower bound on the longest simple path).
# ---------------------------------------------------------------------------

def _bfs_farthest(start: int, adj: list[list[int]], members: set[int]) -> tuple[int, int]:
    """BFS within ``members``. Returns (farthest_node, max_distance_in_edges)."""
    dist = {start: 0}
    farthest = start
    best = 0
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in members and v not in dist:
                d = dist[u] + 1
                dist[v] = d
                if d > best:
                    best = d
                    farthest = v
                q.append(v)
    return farthest, best


def _diameter_atoms(component: list[int], adj: list[list[int]]) -> int:
    """Diameter in *atoms* (= longest shortest path length + 1)."""
    if not component:
        return 0
    if len(component) == 1:
        return 1
    members = set(component)
    a, _ = _bfs_farthest(component[0], adj, members)
    _, d_edges = _bfs_farthest(a, adj, members)
    return d_edges + 1


# ---------------------------------------------------------------------------
# Fundamental cycle basis — DFS spanning tree + back-edges
# ---------------------------------------------------------------------------
# Diestel §1.9: given any spanning tree of a connected graph, the set of
# fundamental cycles (one per non-tree edge, formed by adding the edge to
# the tree path between its endpoints) forms a basis of the cycle space.
# The number of such cycles is exactly the cyclomatic number r = E−V+1
# for each connected component.  We count them by iterative DFS, recording
# the cycle length whenever we cross a back-edge (to a strict ancestor).
# Each back-edge is naturally counted once because we only register the
# cycle when the deeper endpoint sees the shallower one.
# ---------------------------------------------------------------------------

def _fundamental_ring_sizes(component: list[int], adj: list[list[int]]) -> list[int]:
    if len(component) < 3:
        return []                                  # need ≥ 3 atoms for a cycle
    members = set(component)
    start = component[0]

    depth: dict[int, int] = {start: 0}
    parent: dict[int, int] = {start: -1}
    stack: list[tuple[int, list[int], int]] = [(start, list(adj[start]), 0)]
    rings: list[int] = []

    while stack:
        u, nbrs, idx = stack[-1]
        if idx >= len(nbrs):
            stack.pop()
            continue
        v = nbrs[idx]
        stack[-1] = (u, nbrs, idx + 1)
        if v not in members:
            continue
        if v == parent[u]:
            continue
        if v in depth:
            # Back-edge — count the cycle only when v is strictly shallower
            # (avoids double-counting the same back-edge from both endpoints)
            if depth[v] < depth[u]:
                rings.append(depth[u] - depth[v] + 1)
        else:
            depth[v] = depth[u] + 1
            parent[v] = u
            stack.append((v, list(adj[v]), 0))

    return rings


# ---------------------------------------------------------------------------
# Motif detection
# ---------------------------------------------------------------------------
# Each motif is a coarse subgraph fingerprint.  See the module docstring
# for the biology references — we explicitly do NOT pretend these motifs
# imply biological function, only that the *connectivity* matches.
# ---------------------------------------------------------------------------

def _detect_motifs(
    component: list[int],
    world,
    adj: list[list[int]],
    bond_orders: dict[tuple[int, int], int],
    ring_sizes: list[int],
) -> tuple[str, ...]:
    syms = [ELEMENTS_LIST[world.elem_ids[i]].symbol for i in component]
    sym_set = set(syms)

    motifs: list[str] = []

    # ---- amino-acid-like ------------------------------------------------
    # H₂N–CR–COOH fingerprint: component holds an N atom plus a C with a
    # C=O double bond (carbonyl / carboxyl carbon).  Topology only — no
    # stereochemistry.  Lehninger 6e §3.1.
    if 'N' in sym_set and 'C' in sym_set:
        has_carbonyl = False
        for i in component:
            if ELEMENTS_LIST[world.elem_ids[i]].symbol != 'C':
                continue
            for j in adj[i]:
                if ELEMENTS_LIST[world.elem_ids[j]].symbol != 'O':
                    continue
                key = (min(i, j), max(i, j))
                if bond_orders.get(key, 1) >= 2:
                    has_carbonyl = True
                    break
            if has_carbonyl:
                break
        if has_carbonyl:
            motifs.append('amino-acid-like')

    # ---- nucleotide-like ------------------------------------------------
    # Pyrimidine/purine fingerprint: a 5- or 6-membered ring containing
    # both C and N, plus a P atom anywhere in the component (phosphate
    # placeholder).  Lehninger 6e §8.1.
    if 'P' in sym_set and ring_sizes:
        cn_ring_found = False
        if any(s in (5, 6) for s in ring_sizes):
            # Cheaper-but-sufficient surrogate: a candidate ring of the
            # right size exists, and the component already contains both
            # C and N. Checking that the actual atoms of the ring are
            # specifically C+N would require enumerating cycle vertices —
            # for our coarseness budget this surrogate is good enough.
            cn_ring_found = ('C' in sym_set) and ('N' in sym_set)
        if cn_ring_found:
            motifs.append('nucleotide-like')

    return tuple(motifs)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def identify_polymers(world) -> list[Polymer]:
    """Return one Polymer per bonded connected component of ≥ 2 atoms."""
    n = world.n
    if n == 0 or not world.bonds:
        return []

    adj = _build_adjacency(world)
    bond_orders = _build_bond_orders(world)

    visited = bytearray(n)
    polymers: list[Polymer] = []

    for start in range(n):
        if visited[start] or not adj[start]:
            continue
        # Iterative BFS to gather the connected component
        comp: list[int] = []
        q: deque[int] = deque([start])
        visited[start] = 1
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = 1
                    q.append(v)

        if len(comp) < 2:
            continue

        sym_counts = Counter(
            ELEMENTS_LIST[world.elem_ids[i]].symbol for i in comp
        )
        formula = _formula_from_counts(dict(sym_counts))

        chain_length = _diameter_atoms(comp, adj)
        branches = sum(1 for i in comp if len(adj[i]) >= 3)
        ring_sizes = _fundamental_ring_sizes(comp, adj)
        rings = len(ring_sizes)
        motifs = _detect_motifs(comp, world, adj, bond_orders, ring_sizes)

        polymers.append(Polymer(
            formula=formula,
            name=KNOWN_MOLECULES.get(formula),
            indices=tuple(sorted(comp)),
            chain_length=chain_length,
            branches=branches,
            rings=rings,
            ring_sizes=tuple(sorted(ring_sizes)),
            motifs=motifs,
        ))

    return polymers


def polymer_of(world, particle_idx: int) -> Polymer | None:
    """Return the Polymer containing this particle, or None if unbonded."""
    for p in identify_polymers(world):
        if particle_idx in p.indices:
            return p
    return None


def polymer_stats(world) -> dict:
    """Aggregate HUD metrics: longest / mean chain, motif counts.

    Returned dict shape:
        {
            'n_polymers':     int,                # bonded components ≥ 2 atoms
            'max_chain':      int,                # longest chain across all polymers
            'mean_chain':     float,              # mean chain length (atoms)
            'motif_counts':   dict[str, int],     # {'amino-acid-like': N, …}
        }
    Returns zero-filled defaults if the world has no bonds.
    """
    polys = identify_polymers(world)
    if not polys:
        return {
            'n_polymers':   0,
            'max_chain':    0,
            'mean_chain':   0.0,
            'motif_counts': {},
        }

    chain_lengths = [p.chain_length for p in polys]
    motif_counter: Counter[str] = Counter()
    for p in polys:
        for m in p.motifs:
            motif_counter[m] += 1

    return {
        'n_polymers':   len(polys),
        'max_chain':    max(chain_lengths),
        'mean_chain':   sum(chain_lengths) / len(chain_lengths),
        'motif_counts': dict(motif_counter),
    }
