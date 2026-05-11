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
from sim.elements import ELEMENTS, ELEMENTS_LIST, fusion_product, FUSION_REACTIONS
from sim.nuclear import fusion_q_amu, AMU_TO_MEV
from sim import chemistry


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
    """Spot-check: every reaction in FUSION_REACTIONS has a sensible Q-value
    (typically positive, except the known Be-8 resonance step). No reaction
    should be more than ~30 MeV — that would indicate a mass-table error."""

    def test_no_reaction_yields_implausible_q(self):
        for key, prod in FUSION_REACTIONS.items():
            syms = list(key)
            if len(syms) == 1:
                syms = [syms[0], syms[0]]
            a, b = ELEMENTS[syms[0]], ELEMENTS[syms[1]]
            p = ELEMENTS[prod]
            q_mev = (a.mass + b.mass - p.mass) * AMU_TO_MEV
            # Q-values should fit comfortably in [−1 MeV, +30 MeV] range.
            # Endothermic ≤ 1 MeV (Be-8 resonance is the only negative case)
            # and exothermic releases ≤ ~25 MeV (D+D is the upper).
            assert -1.0 < q_mev < 30.0, \
                f'{syms[0]} + {syms[1]} → {prod} gives implausible Q = {q_mev:.2f} MeV'
