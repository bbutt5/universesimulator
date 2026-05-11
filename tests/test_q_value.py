"""Tests for fusion Q-values derived from real measured isotope masses.

A physicist can verify each value against the handbook:
    Q [MeV] = (m_reactants - m_product) [amu] × 931.494

All Q-values come from the AME 2020 isotope masses stored on the
relevant Element entries.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS, ELEMENTS_LIST, fusion_product
from sim.nuclear import fusion_q_amu, find_fusion_product, AMU_TO_MEV
from sim import chemistry


# Canonical reaction set we expect the search to recover. These are the
# stellar-nucleosynthesis channels the simulator should reproduce; the
# search is implementation, the channels here are the physics.
CANONICAL_REACTIONS = [
    (('H',  'H'),  'D'),     # pp-chain step 1 (β⁺ν branch)
    (('H',  'D'),  'He3'),   # pp-chain step 2
    (('D',  'D'),  'He'),    # D-D fusion
    (('Be8','He'), 'C'),     # triple-α completion
    (('C',  'He'), 'O'),     # α capture
    (('O',  'He'), 'Ne'),
    (('Ne', 'He'), 'Mg'),
    (('Mg', 'He'), 'Si'),
    (('Si', 'He'), 'S'),
    (('S',  'He'), 'Ar'),
    (('Ar', 'He'), 'Ca'),
    (('Ca', 'He'), 'Ti'),
    (('Si', 'Si'), 'Fe'),    # Si burning (2 β⁺ branches)
]


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


def _q_mev(reactant_syms, product_sym):
    masses = [ELEMENTS[s].mass for s in reactant_syms]
    p = ELEMENTS[product_sym].mass
    return fusion_q_amu(sum(masses), 0, p - masses[1] if len(masses) > 1 else 0) * AMU_TO_MEV \
           if len(masses) > 1 else None


def _q_pair_mev(a, b, c):
    return fusion_q_amu(ELEMENTS[a].mass, ELEMENTS[b].mass, ELEMENTS[c].mass) * AMU_TO_MEV


class TestPPChainQ:
    """The proton-proton chain dominates hydrogen burning in low-mass stars."""

    def test_pp_step1(self):
        """H + H → D releases ~1.44 MeV (handbook: 1.442 MeV including β⁺ν)."""
        q = _q_pair_mev('H', 'H', 'D')
        assert q == pytest.approx(1.44, abs=0.05)

    def test_pp_step2(self):
        """H + D → ³He releases ~5.49 MeV (handbook: 5.493 MeV)."""
        q = _q_pair_mev('H', 'D', 'He3')
        assert q == pytest.approx(5.49, abs=0.05)

    def test_dd_fusion(self):
        """D + D → ⁴He releases ~23.85 MeV (handbook for d+d→α channel)."""
        q = _q_pair_mev('D', 'D', 'He')
        assert q == pytest.approx(23.85, abs=0.1)


class TestTripleAlphaQ:
    """The 3α process is how stars make carbon."""

    def test_he_he_to_be8_is_slightly_endothermic(self):
        """⁴He + ⁴He → ⁸Be has Q ≈ −0.092 MeV — Be-8 is unbound by ~92 keV
        relative to two alphas. In real stars this proceeds via resonance."""
        q = _q_pair_mev('He', 'He', 'Be8')
        assert q == pytest.approx(-0.092, abs=0.02)
        assert q < 0

    def test_be8_alpha_to_carbon(self):
        """⁸Be + ⁴He → ¹²C completes the triple-α with Q ≈ +7.37 MeV
        (handbook: 7.367 MeV; Hoyle 1954)."""
        q = _q_pair_mev('Be8', 'He', 'C')
        assert q == pytest.approx(7.37, abs=0.05)
        assert q > 0


class TestAlphaCaptureChain:
    """Each alpha capture adds an α to a heavier nucleus, releasing Q MeV.
    Reference Q-values from NNDC: https://www.nndc.bnl.gov."""

    @pytest.mark.parametrize("a, b, c, expected_mev, tol", [
        ('C',  'He', 'O',  7.16,  0.05),
        ('O',  'He', 'Ne', 4.73,  0.05),
        ('Ne', 'He', 'Mg', 9.32,  0.05),
        ('Mg', 'He', 'Si', 9.98,  0.05),
        ('Si', 'He', 'S',  6.95,  0.05),
        ('S',  'He', 'Ar', 6.64,  0.10),
        ('Ar', 'He', 'Ca', 7.04,  0.10),
        ('Ca', 'He', 'Ti', 5.13,  0.10),   # Ca-40 + α → Ti-44 (Ti-44 unstable, 60 yr)
    ])
    def test_alpha_capture(self, a, b, c, expected_mev, tol):
        q = _q_pair_mev(a, b, c)
        assert q == pytest.approx(expected_mev, abs=tol)
        assert q > 0


class TestSiliconBurning:
    """²⁸Si + ²⁸Si → ⁵⁶Fe is the endpoint of stellar exothermic fusion."""

    def test_si_si_to_fe(self):
        """Q ≈ +17.6 MeV — much larger than alpha captures because two
        heavy nuclei combine into the deeply-bound ⁵⁶Fe."""
        q = _q_pair_mev('Si', 'Si', 'Fe')
        assert q == pytest.approx(17.6, abs=0.5)
        assert q > 0


class TestQValueGate:
    """The chemistry _fuse code enforces energy conservation:
        rel_KE + Q ≥ 0
    Exothermic reactions are always allowed (assuming Coulomb cleared);
    endothermic reactions need enough KE to compensate."""

    def test_he_he_to_be8_blocked_at_zero_ke(self, cfg):
        """At zero relative KE, He+He → Be8 (Q < 0) cannot fuse — physical."""
        cfg.chemistry.coulomb_barrier_scale = 0.01  # trivial barrier
        w = World(cfg)
        r = ELEMENTS['He'].covalent_radius * 2
        _place(w, 'He', [0.0,     0.0, 0.0])     # at rest
        _place(w, 'He', [r * 1.5, 0.0, 0.0])     # at rest
        chemistry._fuse(w)
        assert w.n == 2
        assert w.total_fusions == 0

    def test_he_he_to_be8_allowed_with_enough_ke(self, cfg):
        """With rel_KE > |Q| (≈ 0.092 MeV in mass units), endothermic
        He+He → Be8 IS allowed by energy conservation."""
        cfg.chemistry.coulomb_barrier_scale = 0.01
        cfg.chemistry.q_value_scale = 1.0
        w = World(cfg)
        # KE needed: 0.092 amu in sim units. Each He moves at ±v, rel_v = 2v.
        # rel_KE = ½·μ·(2v)² = m_He·v² ≈ 4·v² (m_He ≈ 4)
        # Want rel_KE > 0.1 amu → v > sqrt(0.1/4) ≈ 0.16
        r = ELEMENTS['He'].covalent_radius * 2
        _place(w, 'He', [0.0,     0.0, 0.0], vel=[ 5.0, 0.0, 0.0])
        _place(w, 'He', [r * 1.5, 0.0, 0.0], vel=[-5.0, 0.0, 0.0])
        chemistry._fuse(w)
        assert w.n == 1
        assert w.total_fusions == 1


class TestHandbookCorroboration:
    """Spot-check: every canonical stellar reaction has a sensible Q-value.
    Endothermic reactions should be small (Be-8 is −0.1 MeV); exothermic
    should not exceed ~25 MeV (D+D is the upper)."""

    @pytest.mark.parametrize("reactants, product", CANONICAL_REACTIONS)
    def test_canonical_reaction_q_is_plausible(self, reactants, product):
        a, b = ELEMENTS[reactants[0]], ELEMENTS[reactants[1]]
        p    = ELEMENTS[product]
        q_mev = (a.mass + b.mass - p.mass) * AMU_TO_MEV
        # Endothermic ≤ 1 MeV; exothermic ≤ 30 MeV
        assert -1.0 < q_mev < 30.0, \
            f'{reactants[0]} + {reactants[1]} → {product} gives Q = {q_mev:.2f} MeV'


class TestEmergentSearch:
    """Validate that find_fusion_product reproduces the stellar chain
    without any hand-curated reaction table."""

    @pytest.mark.parametrize("reactants, expected_product", CANONICAL_REACTIONS)
    def test_canonical_reaction_recovered_by_search(self, reactants, expected_product):
        """The Z+A-conservation search must find each canonical product
        as the highest-Q feasible channel."""
        a, b = ELEMENTS[reactants[0]], ELEMENTS[reactants[1]]
        elem, _q = find_fusion_product(
            a.Z, a.A, b.Z, b.A, a.mass, b.mass,
            rel_ke=1e9, q_scale=1.0,    # plenty of KE to clear any endothermic gate
        )
        assert elem is not None, f'No product found for {reactants[0]} + {reactants[1]}'
        assert elem.symbol == expected_product

    def test_unreachable_pair_returns_none(self):
        """H + Fe — Z=27, A=57; no element with that Z/A in the table
        (Co is missing). Even with a β⁺ branch, Z=26 A=57 ≠ Fe-56."""
        h, fe = ELEMENTS['H'], ELEMENTS['Fe']
        elem, _q = find_fusion_product(
            h.Z, h.A, fe.Z, fe.A, h.mass, fe.mass,
            rel_ke=1e9, q_scale=1.0,
        )
        assert elem is None

    def test_endothermic_blocked_at_zero_ke(self):
        """He + He → Be8 has Q = −0.092 MeV. At rel_KE = 0 it should be
        rejected by the energy-conservation gate."""
        he = ELEMENTS['He']
        elem, q = find_fusion_product(
            he.Z, he.A, he.Z, he.A, he.mass, he.mass,
            rel_ke=0.0, q_scale=1.0,
        )
        assert elem is None

    def test_endothermic_allowed_with_enough_ke(self):
        """Same pair, but with rel_KE >> |Q| → Be-8 forms."""
        he = ELEMENTS['He']
        elem, q = find_fusion_product(
            he.Z, he.A, he.Z, he.A, he.mass, he.mass,
            rel_ke=1.0, q_scale=1.0,     # 1 sim_KE >> 0.0001 amu deficit
        )
        assert elem is not None
        assert elem.symbol == 'Be8'

    def test_carbon_burning_emerges(self):
        """C + C → Mg-24 is real stellar carbon-burning. Should appear from
        the search even though it was never in the hand-curated table."""
        c = ELEMENTS['C']
        elem, q_amu = find_fusion_product(
            c.Z, c.A, c.Z, c.A, c.mass, c.mass,
            rel_ke=1e9, q_scale=1.0,
        )
        assert elem is not None
        assert elem.symbol == 'Mg'                    # Z=12, A=24
        # Real C-12 + C-12 → Mg-24 + γ has Q ≈ 13.93 MeV
        assert q_amu * AMU_TO_MEV == pytest.approx(13.93, abs=0.1)

    def test_oxygen_burning_emerges(self):
        """O + O → S-32 is real oxygen-burning. Same point as above."""
        o = ELEMENTS['O']
        elem, q_amu = find_fusion_product(
            o.Z, o.A, o.Z, o.A, o.mass, o.mass,
            rel_ke=1e9, q_scale=1.0,
        )
        assert elem is not None
        assert elem.symbol == 'S'                     # Z=16, A=32
        # Real O-16 + O-16 → S-32 + γ has Q ≈ 16.54 MeV
        assert q_amu * AMU_TO_MEV == pytest.approx(16.54, abs=0.1)
