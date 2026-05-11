"""
3D real-time renderer using vispy.

Particles are drawn as colour-coded spherical markers (CPK colours).
Accreted bodies are rendered by mass: planets as rocky grey, stars as warm
yellow-white — determined entirely by the accumulated mass ratio vs the
base element mass (no hardcoded "this is a star" flag).

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
  Left-click  → inspect particle (click again to deselect)
"""

from __future__ import annotations
import time
from collections import Counter
import numpy as np

from vispy import app, scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from sim.diagnostics import kinetic_energy, gravitational_pe, linear_momentum
from sim.elements import ELEMENTS_LIST
from sim.states import state_counts
from sim.world import World

_FPS_TARGET    = 60
_FPS_WINDOW    = 0.5     # smoothing window for FPS measurement (seconds)
_SPEED_UP      = 2.0     # speed multiplier per keypress
_SPEED_MAX     = 64.0
_SPEED_MIN     = 0.0625  # 1/16×
_CAM_ORBIT_STEP = 5.0    # degrees per arrow-key press
_CAM_ZOOM_STEP  = 1.15   # zoom factor per Page Up/Down press
_CAM_AZIMUTH_0  = 30.0   # default camera azimuth  (degrees)
_CAM_ELEVATION_0 = 20.0  # default camera elevation (degrees)
_PICK_PIXEL_RADIUS = 15.0  # click tolerance for particle picking
_ENERGY_REFRESH_FRAMES = 30   # how often to recompute O(N²) gravitational PE

# CPK colours for rendering classification
_COLOUR_PLANET = np.array([0.55, 0.50, 0.42], dtype=np.float32)   # rocky grey-brown
_COLOUR_STAR   = np.array([1.00, 0.92, 0.65], dtype=np.float32)   # warm yellow-white
_COLOUR_HOT    = np.array([1.00, 0.40, 0.10], dtype=np.float32)   # heat tint blended in
_COLOUR_PLASMA = np.array([0.70, 0.85, 1.00], dtype=np.float32)   # ionised: bluish-white


class Viewer:
    def __init__(self, world: World, cfg):
        self.world  = world
        self.cfg    = cfg
        self.paused = False
        self.speed  = 1.0   # time-scale multiplier

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
        self.view.camera.azimuth   = _CAM_AZIMUTH_0
        self.view.camera.elevation = _CAM_ELEVATION_0

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

        # ---- selection / inspect ------------------------------------------
        self._selected: int | None = None

        self._select_marker = visuals.Markers(parent=self.view.scene)
        self._select_marker.antialias = 0

        self._info_text = visuals.Text(
            '', color=(1.0, 1.0, 1.0, 0.9), font_size=9,
            parent=self.canvas.scene, anchor_x='left', anchor_y='top',
        )
        self._info_text.transform = STTransform(translate=(10, 60))

        # ---- key bindings -------------------------------------------------
        self.canvas.events.key_press.connect(self._on_key)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)

        # ---- perf tracking ------------------------------------------------
        self._last_wall  = time.perf_counter()
        self._fps_acc    = 0.0
        self._fps_frames = 0
        self._fps        = 0.0

        # ---- conservation diagnostics -------------------------------------
        # Recompute the O(N²) PE every _ENERGY_REFRESH_FRAMES; cache between.
        self._ke           = 0.0
        self._pe           = 0.0
        self._momentum_mag = 0.0
        self._energy_ref   = None     # set once we have ≥2 particles
        self._energy_frame = 0

    # ------------------------------------------------------------------
    # Timer callback — drives simulation + render each frame
    # ------------------------------------------------------------------

    def _on_timer(self, event) -> None:
        now  = time.perf_counter()
        wall = now - self._last_wall
        self._last_wall = now

        self._fps_acc    += wall
        self._fps_frames += 1
        if self._fps_acc >= _FPS_WINDOW:
            self._fps        = self._fps_frames / self._fps_acc
            self._fps_acc    = 0.0
            self._fps_frames = 0

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
            self._selected = None
            self.markers.set_data(pos=np.zeros((1, 3)), face_color=(0, 0, 0, 0))
            self._select_marker.set_data(pos=np.zeros((1, 3)), face_color=(0, 0, 0, 0), size=1, edge_width=0)
            self._info_text.text = ''
            self.lines.set_data(pos=np.zeros((2, 3)))
            return

        if self._selected is not None and self._selected >= n:
            self._selected = None

        pos   = w.positions[:n].copy()
        e_ids = w.elem_ids[:n]

        # --- Base CPK colours and covalent-radius sizes --------------------
        colors = np.ones((n, 4), dtype=np.float32)
        sizes  = np.empty(n, dtype=np.float32)
        for idx in range(n):
            elem           = ELEMENTS_LIST[e_ids[idx]]
            colors[idx, :3] = elem.color
            sizes[idx]      = cfg.particle_size_base + cfg.particle_size_scale * elem.covalent_radius

        # --- Temperature tint (cold → CPK, hot → orange-red) --------------
        hot_T = float(getattr(cfg, 'hot_temperature_threshold', 0.0))
        if hot_T > 0.0:
            ke   = 0.5 * w.masses[:n] * np.sum(w.velocities[:n] ** 2, axis=1)
            heat = np.clip(ke / hot_T, 0.0, 1.0).astype(np.float32)[:, np.newaxis]
            colors[:, :3] = colors[:, :3] * (1.0 - heat) + _COLOUR_HOT * heat

        # --- Plasma override (ionised particles) ---------------------------
        ion_mask = w.ionized[:n]
        if ion_mask.any():
            colors[ion_mask, :3] = _COLOUR_PLASMA

        # --- Override for accreted bodies (planets / stars) ----------------
        rend_cfg = cfg
        planet_threshold = float(getattr(rend_cfg, 'planet_mass_threshold', 1e9))
        star_threshold   = float(getattr(rend_cfg, 'star_mass_threshold',   1e9))
        body_size_scale  = float(getattr(rend_cfg, 'body_size_scale',       3.0))

        elem_masses   = np.array([ELEMENTS_LIST[e_ids[i]].mass for i in range(n)], dtype=np.float64)
        actual_masses = w.masses[:n]
        mass_ratios   = actual_masses / elem_masses

        planet_mask = (mass_ratios >= planet_threshold) & (mass_ratios < star_threshold)
        star_mask   = mass_ratios >= star_threshold

        if planet_mask.any():
            body_sizes = (body_size_scale * np.cbrt(mass_ratios)).astype(np.float32)
            colors[planet_mask, :3] = _COLOUR_PLANET
            sizes[planet_mask]      = body_sizes[planet_mask]

        if star_mask.any():
            body_sizes = (body_size_scale * np.cbrt(mass_ratios) * 1.5).astype(np.float32)
            colors[star_mask, :3] = _COLOUR_STAR
            sizes[star_mask]      = body_sizes[star_mask]

        self.markers.set_data(
            pos=pos,
            face_color=colors,
            size=sizes,
            edge_width=0,
        )

        # --- Selection ring + info text ------------------------------------
        sel = self._selected
        if sel is not None:
            self._select_marker.set_data(
                pos=pos[sel:sel + 1],
                face_color=(0.0, 0.0, 0.0, 0.0),
                size=float(sizes[sel]) + 10.0,
                edge_width=2,
                edge_color=(1.0, 1.0, 1.0, 1.0),
            )
            self._info_text.text = self._particle_info(sel)
        else:
            self._select_marker.set_data(
                pos=np.zeros((1, 3), dtype=np.float32),
                face_color=(0.0, 0.0, 0.0, 0.0),
                size=1,
                edge_width=0,
            )
            self._info_text.text = ''

        # --- Bond lines ----------------------------------------------------
        if cfg.show_bonds and len(w.bonds) > 0:
            bond_pts = []
            for bond in w.bonds:
                bond_pts.append(w.positions[bond.i])
                bond_pts.append(w.positions[bond.j])
            self.lines.set_data(
                pos=np.array(bond_pts, dtype=np.float32),
                color=(0.9, 0.9, 0.9, cfg.bond_alpha),
                connect='segments',
            )
        else:
            self.lines.set_data(pos=np.zeros((2, 3)), color=(0, 0, 0, 0))

        self.canvas.update()

    def _update_title(self) -> None:
        w      = self.world
        counts = Counter(ELEMENTS_LIST[w.elem_ids[i]].symbol for i in range(w.n))
        top    = ' '.join(f'{s}:{c}' for s, c in counts.most_common(4))
        sc     = state_counts(w)
        status = 'PAUSED ' if self.paused else ''

        # Conservation diagnostics (throttled — gravitational PE is O(N²))
        self._energy_frame += 1
        if self._energy_frame % _ENERGY_REFRESH_FRAMES == 0 or self._energy_ref is None:
            self._ke = kinetic_energy(w)
            self._pe = gravitational_pe(w)
            self._momentum_mag = float(np.linalg.norm(linear_momentum(w)))
            if self._energy_ref is None and w.n >= 2:
                self._energy_ref = self._ke + self._pe   # set baseline once

        e_total = self._ke + self._pe
        drift   = ''
        if self._energy_ref is not None and self._energy_ref != 0:
            drift_pct = 100.0 * (e_total - self._energy_ref) / abs(self._energy_ref)
            drift     = f' drift={drift_pct:+.2f}%'

        self.canvas.title = (
            f'Universe | {status}'
            f'n={w.n}  bonds={len(w.bonds)}  '
            f'fusions={w.total_fusions}  accreted={w.total_accretions}  '
            f'S:{sc["solid"]} L:{sc["liquid"]} G:{sc["gas"]} P:{sc["plasma"]}  '
            f'E={e_total:.2e}{drift}  clamps={w.total_velocity_clamps}  '
            f't={w.time:.1f}s  fps={self._fps:.0f}  '
            f'speed={self.speed:.1f}x  [{top}]'
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        k   = event.key.name if event.key else ''
        cam = self.view.camera
        if k == 'Space':
            self.paused = not self.paused
        elif k in ('Equal', '+'):
            self.speed = min(self.speed * _SPEED_UP, _SPEED_MAX)
        elif k in ('Minus', '-'):
            self.speed = max(self.speed / _SPEED_UP, _SPEED_MIN)
        elif k == 'Left':
            cam.azimuth -= _CAM_ORBIT_STEP
        elif k == 'Right':
            cam.azimuth += _CAM_ORBIT_STEP
        elif k == 'Up':
            cam.elevation = min(cam.elevation + _CAM_ORBIT_STEP, 90.0)
        elif k == 'Down':
            cam.elevation = max(cam.elevation - _CAM_ORBIT_STEP, -90.0)
        elif k == 'PageUp':
            cam.distance = max(cam.distance / _CAM_ZOOM_STEP, 1.0)
        elif k == 'PageDown':
            cam.distance *= _CAM_ZOOM_STEP
        elif k == 'R':
            cam.azimuth   = _CAM_AZIMUTH_0
            cam.elevation = _CAM_ELEVATION_0
            cam.distance  = self.cfg.renderer.camera_distance
            cam.center    = (0.0, 0.0, 0.0)
        elif k in ('Q', 'Escape'):
            self.canvas.app.quit()

    # ------------------------------------------------------------------
    # Mouse — particle picking
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event) -> None:
        if event.button != 1:
            return
        w = self.world
        if w.n == 0:
            return

        click_xy = np.array(event.pos[:2], dtype=np.float32)
        screen_xy, in_front = self._project_to_canvas(w.positions[:w.n])
        if not in_front.any():
            return

        dists = np.linalg.norm(screen_xy - click_xy, axis=1)
        dists[~in_front] = np.inf
        nearest = int(np.argmin(dists))

        if dists[nearest] < _PICK_PIXEL_RADIUS:
            self._selected = nearest if self._selected != nearest else None
        else:
            self._selected = None

    def _project_to_canvas(self, pos3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D world positions to 2D canvas pixel coords.

        Returns (screen_xy, in_front) where in_front masks particles
        not behind the camera. Handles the perspective divide that
        vispy's transform.map() does not do automatically.
        """
        pts = np.asarray(pos3d, dtype=np.float32)
        tr  = self.markers.transforms.get_transform('visual', 'canvas')
        out = tr.map(pts)

        if out.shape[1] == 4:
            w_h      = out[:, 3]
            in_front = w_h > 1e-6
            safe_w   = np.where(np.abs(w_h) > 1e-9, w_h, 1.0)
            screen_xy = out[:, :2] / safe_w[:, np.newaxis]
        else:
            screen_xy = out[:, :2]
            in_front  = np.ones(pts.shape[0], dtype=bool)

        return screen_xy, in_front

    def _particle_info(self, i: int) -> str:
        w    = self.world
        elem = ELEMENTS_LIST[w.elem_ids[i]]
        mass_ratio = w.masses[i] / elem.mass
        speed      = float(np.linalg.norm(w.velocities[i]))
        pos        = w.positions[i]

        bonded = [
            ELEMENTS_LIST[w.elem_ids[b.j if b.i == i else b.i]].symbol
            for b in w.bonds if b.i == i or b.j == i
        ]

        planet_threshold = float(getattr(self.cfg.renderer, 'planet_mass_threshold', 1e9))
        star_threshold   = float(getattr(self.cfg.renderer, 'star_mass_threshold',   1e9))
        if mass_ratio >= star_threshold:
            kind = 'STAR'
        elif mass_ratio >= planet_threshold:
            kind = 'PLANET'
        else:
            kind = 'atom'

        lines = [
            f'[{kind}] {elem.name} ({elem.symbol})  Z={elem.Z}',
            f'mass: {w.masses[i]:.3g}  ratio: {mass_ratio:.2f}x',
            f'pos:   ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})',
            f'speed: {speed:.1f}',
            f'bonds: {w.bond_counts[i]}' + (f'  → {" ".join(bonded)}' if bonded else ''),
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        app.run()
