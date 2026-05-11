"""Tests for sim/nuclear.py — Coulomb-barrier fusion gating.

These validate the *physics* of the Coulomb barrier, not just the code:
the same energy that fuses two hydrogens must fail to fuse two heliums,
because V_C ∝ Z₁·Z₂ and He has 4× the proton-product of H+H. A physicist
can check each assertion against the formula.
"""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS, ELEMENTS_LIST
from sim import chemistry
from sim.nuclear import coulomb_barrier_sim, _R0_FM


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestCoulombBarrierFormula:
    """V_C = scale · Z₁·Z₂ / (r₀·(A₁^⅓+A₂^⅓)). Check the maths directly."""

    def test_hydrogen_hydrogen(self):
        # V_C(H,H) with scale=1: Z₁Z₂=1, r_nuc = 1.2·(1+1) = 2.4 fm → V_C = 1/2.4
        v_c = coulomb_barrier_sim(1, 1, 1, 1, scale=1.0)
        assert v_c == pytest.approx(1.0 / 2.4, rel=1e-9)

    def test_helium_helium(self):
        # V_C(He,He): Z₁Z₂=4, r_nuc = 1.2·(4^⅓+4^⅓) = 1.2·2·4^⅓
        expected = 4.0 / (_R0_FM * 2.0 * 4.0 ** (1.0 / 3.0))
        assert coulomb_barrier_sim(2, 4, 2, 4, 1.0) == pytest.approx(expected, rel=1e-9)

    def test_helium_pair_is_harder_than_hydrogen_pair(self):
        """Real physics: V_C(He,He) / V_C(H,H) ≈ 3.2× (from Z² scaling
        with weak r_nuc growth)."""
        v_hh   = coulomb_barrier_sim(1, 1, 1, 1, 1.0)
        v_hehe = coulomb_barrier_sim(2, 4, 2, 4, 1.0)
        ratio  = v_hehe / v_hh
        # Z₁Z₂ ratio is 4×; r_nuc ratio is 4^(1/3) ≈ 1.587; net ≈ 2.52
        assert ratio == pytest.approx(4.0 / (4.0 ** (1.0/3.0)), rel=1e-6)
        assert ratio > 2.5

    def test_iron_iron_far_higher_than_hydrogen(self):
        """V_C(Fe,Fe) ≈ 175× V_C(H,H) — Coulomb barrier explains why
        stellar fusion ends at iron."""
        v_hh   = coulomb_barrier_sim(1, 1, 1, 1, 1.0)
        v_fefe = coulomb_barrier_sim(26, 56, 26, 56, 1.0)
        assert v_fefe / v_hh > 150.0


class TestCoulombGate:
    """Fusion is now gated by V_C in sim KE units — no can_fuse boolean."""

    def test_at_matched_KE_hydrogen_fuses_but_helium_does_not(self, cfg):
        """Choose a barrier scale where V_C(H,H) < KE < V_C(He,He).
        Only the H pair clears its barrier and fuses."""
        # With scale=1000, V_C(H,H) ≈ 417 and V_C(He,He) ≈ 1050.
        cfg.chemistry.coulomb_barrier_scale = 1000.0

        # rel_KE needs to land between those two barriers.  Use mu·|Δv|²/2
        # where mu = mass/2 for identical species.
        # Pick rel_KE ≈ 700 (between 417 and 1050).
        # For H pair: 0.5 · mu_H · v_rel² = 700 → v_rel = sqrt(2·700/(0.504)) ≈ 52.7
        # For He pair: same KE → v_rel = sqrt(2·700/(2.0015)) ≈ 26.4
        m_h, m_he = ELEMENTS['H'].mass, ELEMENTS['He'].mass
        mu_h  = m_h * m_h / (m_h + m_h)
        mu_he = m_he * m_he / (m_he + m_he)
        ke_target = 700.0
        v_h  = float(np.sqrt(2 * ke_target / mu_h))   / 2.0
        v_he = float(np.sqrt(2 * ke_target / mu_he)) / 2.0

        w = World(cfg)
        r_h  = ELEMENTS['H'].covalent_radius * 2
        r_he = ELEMENTS['He'].covalent_radius * 2
        _place(w, 'H',  [0.0,     0.0, 0.0], vel=[ v_h,  0.0, 0.0])
        _place(w, 'H',  [r_h*1.5, 0.0, 0.0], vel=[-v_h,  0.0, 0.0])
        _place(w, 'He', [0.0,     500.0, 0.0], vel=[ v_he, 0.0, 0.0])
        _place(w, 'He', [r_he*1.5, 500.0, 0.0], vel=[-v_he, 0.0, 0.0])

        n_before = w.n
        chemistry._fuse(w)

        # H pair fused (-2 +1 = -1); He pair unchanged (n drops by exactly 1)
        assert w.n == n_before - 1
        assert w.total_fusions == 1
        # Surviving species: both He still present + one D from the H fusion
        symbols = sorted(ELEMENTS_LIST[w.elem_ids[i]].symbol for i in range(w.n))
        assert symbols == ['D', 'He', 'He']

    def test_below_barrier_no_fusion(self, cfg):
        """Sub-barrier KE → fusion does not happen (Gamow tunnelling not
        modelled — flagged on issue #11)."""
        cfg.chemistry.coulomb_barrier_scale = 1000.0   # V_C(H,H) ≈ 417
        m_h = ELEMENTS['H'].mass
        # rel_KE = 50 (well below 417). v_rel = sqrt(2·50 / (m_h/2)) ≈ 14
        v = float(np.sqrt(2 * 50.0 / (m_h / 2))) / 2.0

        w = World(cfg)
        r = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0, 0.0, 0.0],   vel=[ v, 0.0, 0.0])
        _place(w, 'H', [r*1.5, 0.0, 0.0], vel=[-v, 0.0, 0.0])
        chemistry._fuse(w)

        assert w.n == 2
        assert w.total_fusions == 0

    def test_disabled_when_scale_zero(self, cfg):
        cfg.chemistry.coulomb_barrier_scale = 0.0
        w = World(cfg)
        r = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0,   0.0, 0.0], vel=[ 5000.0, 0.0, 0.0])
        _place(w, 'H', [r*1.5, 0.0, 0.0], vel=[-5000.0, 0.0, 0.0])
        chemistry._fuse(w)
        assert w.n == 2
        assert w.total_fusions == 0
