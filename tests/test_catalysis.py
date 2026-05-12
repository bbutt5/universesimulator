"""Tests for heterogeneous catalysis in sim/chemistry.

Real surface catalysis (Sabatier 1911, Langmuir 1918) accelerates reactions
near solid bodies because the surface offers adsorption sites that lower
the activation barrier. In our model this is implemented as a velocity-
threshold multiplier near accreted bodies.
"""

from __future__ import annotations
import numpy as np
import pytest

from sim.world import World
from sim.elements import ELEMENTS
from sim import chemistry


def _place(world, sym, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(
        ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float),
    )


def _viewer_cfg(cfg):
    """Tests don't load the renderer config — supply the planet_mass_threshold
    here so the catalysis surface-detector works."""
    from types import SimpleNamespace
    cfg.renderer = SimpleNamespace(planet_mass_threshold=10.0)
    return cfg


class TestCatalysisFiresNearBody:
    def test_bond_forms_at_higher_velocity_near_surface(self, cfg):
        """A pair too fast to bond in empty space should bond near a body
        (the catalytic surface). We saturate the catalyst's own valence so
        it can't capture the test atoms directly — the test isolates the
        velocity-threshold relaxation."""
        _viewer_cfg(cfg)
        cfg.chemistry.catalysis_factor = 2.0
        cfg.chemistry.catalysis_radius = 100.0
        cfg.chemistry.bond_velocity_threshold = 50.0

        w = World(cfg)
        # A "catalyst body": Fe with massive accumulated mass AND saturated
        # bonds so the catalyst itself doesn't compete for the H atoms.
        cat = _place(w, 'Fe', [0.0, 0.0, 0.0])
        w.masses[cat] = ELEMENTS['Fe'].mass * 50.0      # mass_ratio = 50 ≥ planet_threshold
        w.bond_counts[cat] = ELEMENTS['Fe'].max_bonds   # all 6 valences used

        # Two H atoms at rel_v = 80 (above raw threshold 50, below catalysed 100)
        # placed so their midpoint is within catalysis_radius (100) of the catalyst.
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [40.0,            0.0, 0.0], vel=[ 40.0, 0.0, 0.0])
        _place(w, 'H', [40.0 + r_eq*1.1, 0.0, 0.0], vel=[-40.0, 0.0, 0.0])

        chemistry._form_bonds(w)
        h_h_bonds = [b for b in w.bonds if b.i in (1, 2) and b.j in (1, 2)]
        assert len(h_h_bonds) == 1   # catalysed bond formed

    def test_no_bond_at_same_velocity_without_catalyst(self, cfg):
        """Same H+H pair at rel_v=80, no catalyst → no bond."""
        _viewer_cfg(cfg)
        cfg.chemistry.catalysis_factor = 1.0
        cfg.chemistry.catalysis_radius = 0.0
        cfg.chemistry.bond_velocity_threshold = 50.0

        w = World(cfg)
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [0.0,         0.0, 0.0], vel=[ 40.0, 0.0, 0.0])
        _place(w, 'H', [r_eq * 1.1,  0.0, 0.0], vel=[-40.0, 0.0, 0.0])

        chemistry._form_bonds(w)
        assert len(w.bonds) == 0       # rel_v=80 > threshold=50, no catalyst


class TestCatalysisFarFromBody:
    def test_pair_outside_catalysis_radius_not_catalysed(self, cfg):
        """Even with a catalyst body in the world, pairs beyond catalysis_radius
        don't get the velocity-threshold relaxation."""
        _viewer_cfg(cfg)
        cfg.chemistry.catalysis_factor = 2.0
        cfg.chemistry.catalysis_radius = 100.0
        cfg.chemistry.bond_velocity_threshold = 50.0

        w = World(cfg)
        # Body far away
        cat = _place(w, 'Fe', [0.0, 0.0, 0.0])
        w.masses[cat] = ELEMENTS['Fe'].mass * 50.0
        # Pair at 1000 SU away from body
        r_eq = ELEMENTS['H'].covalent_radius * 2
        _place(w, 'H', [1000.0,         0.0, 0.0], vel=[ 40.0, 0.0, 0.0])
        _place(w, 'H', [1000.0 + r_eq*1.1, 0.0, 0.0], vel=[-40.0, 0.0, 0.0])

        chemistry._form_bonds(w)
        # rel_v=80 > raw 50, far from body → no catalysis → no bond
        h_bonds = [b for b in w.bonds if b.i in (1, 2) and b.j in (1, 2)]
        assert len(h_bonds) == 0
