"""
3D real-time renderer using vispy.

Particles are drawn as colour-coded spherical markers (CPK colours).
Covalent bonds are drawn as line segments.
A minimal HUD in the window title shows live stats.

Controls (built into vispy TurntableCamera):
  Left-drag   → orbit
  Right-drag  → zoom
  Middle-drag → pan
  Scroll      → zoom
  Space       → pause / resume
  +/-         → speed up / slow down sim
  R           → reset zoom
  Q / Escape  → quit
"""

from __future__ import annotations
import time
from collections import Counter
import numpy as np

from vispy import app, scene
from vispy.scene import visuals

from sim.elements import ELEMENTS_LIST
from sim.world import World

_FPS_TARGET  = 60       # target frame rate
_FPS_WINDOW  = 0.5      # smoothing window for FPS measurement (seconds)
_SPEED_UP    = 2.0      # speed multiplier per keypress
_SPEED_MAX   = 64.0     # maximum sim speed
_SPEED_MIN   = 0.0625   # minimum sim speed (1/16×)


class Viewer:
    def __init__(self, world: World, cfg):
        self.world  = world
        self.cfg    = cfg
        self.paused = False
        self.speed  = 1.0   # time-scale multiplier (keyboard controlled)

        # ---- canvas -------------------------------------------------------
        bg = cfg.renderer.background
        self.canvas = scene.SceneCanvas(
            title='Universe Simulator',
            size=(cfg.renderer.width, cfg.renderer.height),
            bgcolor=tuple(bg) + (1.0,),
            keys='interactive',
            show=True,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera           = 'turntable'
        self.view.camera.fov       = cfg.renderer.fov
        self.view.camera.distance  = cfg.renderer.camera_distance
        self.view.camera.azimuth   = 30.0
        self.view.camera.elevation = 20.0

        # ---- visuals -------------------------------------------------------
        self.markers = visuals.Markers(parent=self.view.scene)
        self.markers.antialias = 0

        self.lines = visuals.Line(
            parent=self.view.scene,
            method='gl',
            connect='segments',
        )

        # ---- timer --------------------------------------------------------
        self._timer = app.Timer(
            interval=1.0 / _FPS_TARGET,
            connect=self._on_timer,
            start=True,
        )

        # ---- key bindings ------------------------------------------------
        self.canvas.events.key_press.connect(self._on_key)

        # ---- perf tracking -----------------------------------------------
        self._last_wall  = time.perf_counter()
        self._fps_acc    = 0.0
        self._fps_frames = 0
        self._fps        = 0.0

    # ------------------------------------------------------------------
    # Timer callback — drives simulation + render each frame
    # ------------------------------------------------------------------
    def _on_timer(self, event) -> None:
        now  = time.perf_counter()
        wall = now - self._last_wall
        self._last_wall = now

        # FPS
        self._fps_acc    += wall
        self._fps_frames += 1
        if self._fps_acc >= _FPS_WINDOW:
            self._fps        = self._fps_frames / self._fps_acc
            self._fps_acc    = 0.0
            self._fps_frames = 0

        # Sim steps
        if not self.paused:
            dt      = self.cfg.simulation.time_step * self.speed
            n_steps = self.cfg.simulation.steps_per_frame
            for _ in range(n_steps):
                self.world.step(dt)

        self._update_visuals()
        self._update_title()

    # ------------------------------------------------------------------
    # Update vispy visuals from world state
    # ------------------------------------------------------------------
    def _update_visuals(self) -> None:
        w   = self.world
        n   = w.n
        cfg = self.cfg.renderer

        if n == 0:
            self.markers.set_data(pos=np.zeros((1, 3)), face_color=(0, 0, 0, 0))
            self.lines.set_data(pos=np.zeros((2, 3)))
            return

        pos   = w.positions[:n].copy()
        e_ids = w.elem_ids[:n]

        # Build colour array (RGBA)
        colors = np.ones((n, 4), dtype=np.float32)
        for idx in range(n):
            colors[idx, :3] = ELEMENTS_LIST[e_ids[idx]].color

        # Particle sizes proportional to covalent radius
        sizes = np.empty(n, dtype=np.float32)
        for idx in range(n):
            rc = ELEMENTS_LIST[e_ids[idx]].covalent_radius
            sizes[idx] = cfg.particle_size_base + cfg.particle_size_scale * rc

        self.markers.set_data(
            pos=pos,
            face_color=colors,
            size=sizes,
            edge_width=0,
        )

        # Bond lines
        if cfg.show_bonds and len(w.bonds) > 0:
            bond_pts = []
            for bond in w.bonds:
                bond_pts.append(w.positions[bond.i])
                bond_pts.append(w.positions[bond.j])
            pts = np.array(bond_pts, dtype=np.float32)
            self.lines.set_data(
                pos=pts,
                color=(0.9, 0.9, 0.9, cfg.bond_alpha),
                connect='segments',
            )
        else:
            self.lines.set_data(pos=np.zeros((2, 3)), color=(0, 0, 0, 0))

        self.canvas.update()

    def _update_title(self) -> None:
        w = self.world
        counts = Counter(ELEMENTS_LIST[w.elem_ids[i]].symbol for i in range(w.n))
        top    = ' '.join(f'{s}:{c}' for s, c in counts.most_common(4))
        status = 'PAUSED ' if self.paused else ''
        self.canvas.title = (
            f'Universe | {status}'
            f'n={w.n}  bonds={len(w.bonds)}  fusions={w.total_fusions}  '
            f't={w.time:.1f}s  fps={self._fps:.0f}  '
            f'speed={self.speed:.1f}x  [{top}]'
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def _on_key(self, event) -> None:
        k = event.key.name if event.key else ''
        if k == 'Space':
            self.paused = not self.paused
        elif k in ('Equal', '+'):
            self.speed = min(self.speed * _SPEED_UP, _SPEED_MAX)
        elif k in ('Minus', '-'):
            self.speed = max(self.speed / _SPEED_UP, _SPEED_MIN)
        elif k == 'R':
            self.view.camera.distance = self.cfg.renderer.camera_distance
        elif k in ('Q', 'Escape'):
            self.canvas.app.quit()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        app.run()
