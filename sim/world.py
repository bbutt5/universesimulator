"""
World — holds all simulation state and drives the step loop.

Data layout
-----------
All per-particle data lives in pre-allocated numpy arrays indexed 0..n-1.
Particle removal (fusion) uses swap-with-last so the arrays stay contiguous
and no shifting is needed.  Bond indices are patched after every swap.
"""

from __future__ import annotations
import numpy as np

from sim import physics, chemistry
from sim.elements import ELEMENTS_LIST, SYMBOL_TO_ID, Element
from sim.injector import Injector
from sim.particle import Bond

_INIT_CAP = 8_192   # initial allocation; doubles when full


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        cap = _INIT_CAP

        # Per-particle state  (allocated, only [:n] is valid)
        self.positions  = np.zeros((cap, 3), dtype=np.float64)
        self.velocities = np.zeros((cap, 3), dtype=np.float64)
        self.forces_cur = np.zeros((cap, 3), dtype=np.float64)
        self.forces_prev= np.zeros((cap, 3), dtype=np.float64)
        self.masses     = np.zeros(cap,       dtype=np.float64)
        self.elem_ids   = np.zeros(cap,       dtype=np.int32)   # index into ELEMENTS_LIST
        self.bond_counts= np.zeros(cap,       dtype=np.int32)

        self._cap = cap
        self.n: int = 0          # active particle count
        self.time: float = 0.0
        self.bonds: list[Bond] = []

        # Counters for HUD
        self.total_injected: int = 0
        self.total_fusions:  int = 0
        self.total_bonds_formed: int = 0

        # Deferred injection accumulator (fractional particles)
        self._inject_acc: float = 0.0

        self.injector = Injector(cfg)

    # ------------------------------------------------------------------
    # Array growth
    # ------------------------------------------------------------------
    def _grow(self) -> None:
        new_cap = self._cap * 2

        def _ext2(arr: np.ndarray) -> np.ndarray:
            z = np.zeros((new_cap, 3), dtype=arr.dtype)
            z[:self._cap] = arr
            return z

        def _ext1(arr: np.ndarray) -> np.ndarray:
            z = np.zeros(new_cap, dtype=arr.dtype)
            z[:self._cap] = arr
            return z

        self.positions   = _ext2(self.positions)
        self.velocities  = _ext2(self.velocities)
        self.forces_cur  = _ext2(self.forces_cur)
        self.forces_prev = _ext2(self.forces_prev)
        self.masses      = _ext1(self.masses)
        self.elem_ids    = _ext1(self.elem_ids)
        self.bond_counts = _ext1(self.bond_counts)
        self._cap = new_cap

    # ------------------------------------------------------------------
    # Particle management
    # ------------------------------------------------------------------
    def add_particle(self, element: Element, pos: np.ndarray, vel: np.ndarray) -> int:
        if self.n >= self._cap:
            self._grow()
        i = self.n
        self.positions[i]   = pos
        self.velocities[i]  = vel
        self.forces_cur[i]  = 0.0
        self.forces_prev[i] = 0.0
        self.masses[i]      = element.mass
        self.elem_ids[i]    = SYMBOL_TO_ID[element.symbol]
        self.bond_counts[i] = 0
        self.n += 1
        self.total_injected += 1
        return i

    def remove_particle(self, i: int) -> None:
        """Remove particle i using swap-with-last strategy. O(bonds) cost."""
        # Drop all bonds touching i
        surviving = []
        for b in self.bonds:
            if b.i == i or b.j == i:
                other = b.j if b.i == i else b.i
                self.bond_counts[other] = max(0, self.bond_counts[other] - 1)
            else:
                surviving.append(b)
        self.bonds = surviving

        last = self.n - 1
        if i != last:
            # Copy last → i
            self.positions[i]   = self.positions[last]
            self.velocities[i]  = self.velocities[last]
            self.forces_cur[i]  = self.forces_cur[last]
            self.forces_prev[i] = self.forces_prev[last]
            self.masses[i]      = self.masses[last]
            self.elem_ids[i]    = self.elem_ids[last]
            self.bond_counts[i] = self.bond_counts[last]
            # Patch bond indices that pointed at 'last'
            for b in self.bonds:
                if b.i == last: b.i = i
                if b.j == last: b.j = i

        self.n -= 1

    # ------------------------------------------------------------------
    # Element helpers
    # ------------------------------------------------------------------
    def element_of(self, i: int) -> Element:
        return ELEMENTS_LIST[self.elem_ids[i]]

    def symbol_of(self, i: int) -> str:
        return ELEMENTS_LIST[self.elem_ids[i]].symbol

    # ------------------------------------------------------------------
    # Active-slice views (read-only numpy slices for physics code)
    # ------------------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.positions[:self.n]

    @property
    def vel(self) -> np.ndarray:
        return self.velocities[:self.n]

    @property
    def frc(self) -> np.ndarray:
        return self.forces_cur[:self.n]

    @property
    def mass(self) -> np.ndarray:
        return self.masses[:self.n]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        # --- Inject new particles ---
        self.injector.inject(self, dt)

        if self.n == 0:
            self.time += dt
            return

        n = self.n
        pos    = self.positions[:n]
        vel    = self.velocities[:n]
        f_cur  = self.forces_cur[:n]
        f_prev = self.forces_prev[:n]
        m      = self.masses[:n]

        # --- Velocity-Verlet integration (first half) ---
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        pos += vel * dt + 0.5 * (f_cur / m[:, np.newaxis]) * dt * dt

        # --- Compute new forces ---
        f_prev[:] = f_cur
        f_cur[:]  = 0.0

        physics.add_gravity(self, f_cur)
        physics.add_bond_forces(self, f_cur)

        # --- Velocity-Verlet (second half) ---
        # v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt
        vel += 0.5 * ((f_prev + f_cur) / m[:, np.newaxis]) * dt

        # --- Clamp velocity ---
        v_max  = self.cfg.physics.max_velocity
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        too_fast = speeds > v_max
        if np.any(too_fast):
            vel[too_fast[:, 0]] *= (v_max / speeds[too_fast[:, 0]])

        # --- Chemistry ---
        chemistry.update(self)

        self.time += dt
