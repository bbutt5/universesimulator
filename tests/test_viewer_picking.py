"""Tests for sim/viewer.py particle-picking logic.

Exercises the real vispy projection transform — confirms that clicking on
a particle's projected screen position selects it, and clicks far away
do not.
"""

from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from sim.world import World
from sim.elements import ELEMENTS

vispy = pytest.importorskip("vispy")
from sim.viewer import Viewer  # noqa: E402


class _FakeMouseEvent:
    def __init__(self, x: float, y: float, button: int = 1):
        self.pos    = (x, y)
        self.button = button


@pytest.fixture
def viewer_cfg(cfg):
    cfg.renderer = SimpleNamespace(
        width=800, height=600,
        background=[0.0, 0.0, 0.0],
        fov=60.0,
        camera_distance=2000.0,
        particle_size_base=4.0,
        particle_size_scale=0.05,
        show_bonds=True,
        bond_alpha=0.6,
        planet_mass_threshold=20.0,
        star_mass_threshold=200.0,
        body_size_scale=3.0,
    )
    return cfg


@pytest.fixture
def viewer(viewer_cfg):
    cfg = viewer_cfg
    w = World(cfg)
    w.add_particle(ELEMENTS['H'],  np.array([0.0,    0.0, 0.0]),  np.zeros(3))
    w.add_particle(ELEMENTS['Fe'], np.array([200.0,  0.0, 0.0]),  np.zeros(3))
    w.add_particle(ELEMENTS['C'],  np.array([0.0,  200.0, 0.0]),  np.zeros(3))

    v = Viewer(w, cfg)
    v._update_visuals()
    from vispy import app
    app.process_events()
    yield v
    v.canvas.close()


class TestProjection:
    def test_origin_particle_projects_to_canvas_centre(self, viewer):
        screen_xy, in_front = viewer._project_to_canvas(viewer.world.positions[:1])
        cx, cy = viewer.canvas.size[0] / 2.0, viewer.canvas.size[1] / 2.0
        assert in_front[0]
        np.testing.assert_allclose(screen_xy[0], [cx, cy], atol=2.0)

    def test_in_front_mask_all_true_for_visible_particles(self, viewer):
        _, in_front = viewer._project_to_canvas(viewer.world.positions[:viewer.world.n])
        assert in_front.all()


class TestPicking:
    def test_click_on_particle_selects_it(self, viewer):
        screen_xy, _ = viewer._project_to_canvas(viewer.world.positions[:viewer.world.n])
        assert viewer._selected is None
        x, y = screen_xy[1]
        viewer._on_mouse_press(_FakeMouseEvent(float(x), float(y)))
        assert viewer._selected == 1

    def test_click_far_away_deselects(self, viewer):
        viewer._selected = 0
        viewer._on_mouse_press(_FakeMouseEvent(-9999.0, -9999.0))
        assert viewer._selected is None

    def test_click_same_particle_twice_toggles_off(self, viewer):
        screen_xy, _ = viewer._project_to_canvas(viewer.world.positions[:viewer.world.n])
        x, y = screen_xy[0]
        viewer._on_mouse_press(_FakeMouseEvent(float(x), float(y)))
        assert viewer._selected == 0
        viewer._on_mouse_press(_FakeMouseEvent(float(x), float(y)))
        assert viewer._selected is None

    def test_click_different_particle_switches(self, viewer):
        screen_xy, _ = viewer._project_to_canvas(viewer.world.positions[:viewer.world.n])
        viewer._on_mouse_press(_FakeMouseEvent(float(screen_xy[0, 0]), float(screen_xy[0, 1])))
        assert viewer._selected == 0
        viewer._on_mouse_press(_FakeMouseEvent(float(screen_xy[2, 0]), float(screen_xy[2, 1])))
        assert viewer._selected == 2

    def test_right_click_ignored(self, viewer):
        screen_xy, _ = viewer._project_to_canvas(viewer.world.positions[:viewer.world.n])
        x, y = screen_xy[1]
        viewer._on_mouse_press(_FakeMouseEvent(float(x), float(y), button=2))
        assert viewer._selected is None


class TestParticleInfo:
    def test_info_string_contains_element_and_kind(self, viewer):
        info = viewer._particle_info(0)
        assert 'Hydrogen' in info or 'H' in info
        assert 'atom' in info or 'PLANET' in info or 'STAR' in info
        assert 'mass' in info
        assert 'pos' in info
        assert 'speed' in info

    def test_info_for_invalid_index_does_not_crash(self, viewer):
        # Out-of-range index would raise IndexError — verify the protection
        # that lives in _update_visuals (selected reset when >= n)
        viewer._selected = 999
        viewer._update_visuals()
        assert viewer._selected is None
