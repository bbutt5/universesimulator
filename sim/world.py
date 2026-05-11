"""
World — holds all simulation state and drives the step loop.

Data layout
-----------
All per-particle data lives in pre-allocated numpy arrays indexed 0..n-1.
Particle removal (fusion / accretion) uses swap-with-last so the arrays
stay contiguous and no shifting is needed.  Bond indices are patched after
every swap.

Step order
----------
  1. Inject new particles
  2. Velocity-Verlet first half  (position update)
  3. Recompute forces: gravity + bond + thermal pressure
  4. Velocity-Verlet second half (velocity update)
  5. Clamp velocity
  6. Chemistry  (bond break/form, nuclear fusion + radiation kick)
  7. Accretion  (gravitationally bound inert pairs merge)
  8. Advance time

Chemistry fires before accretion so light fusable elements (H, He) fuse
rather than accrete into rocky bodies.
"""

from __future__ import annotations
import numpy as np

from sim import physics, chemistry, accretion
from sim.elements import (
    ELEMENTS_LIST, SYMBOL_TO_ID, Element, IONIZATION_ENERGIES_EV,
)
from sim.injector import Injector
from sim.particle import Bond
from sim.thermal import add_thermal_pressure
from sim.vdw import add_vdw_forces

_INIT_CAP = 8_192   # initial allocation; doubles when full


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        cap = _INIT_CAP

        # Per-particle state  (allocated, only [:n] is valid)
        self.positions   = np.zeros((cap, 3), dtype=np.float64)
        self.velocities  = np.zeros((cap, 3), dtype=np.float64)
        self.forces_cur  = np.zeros((cap, 3), dtype=np.float64)
        self.forces_prev = np.zeros((cap, 3), dtype=np.float64)
        self.masses      = np.zeros(cap,       dtype=np.float64)
        self.elem_ids    = np.zeros(cap,       dtype=np.int32)   # index into ELEMENTS_LIST
        self.bond_counts = np.zeros(cap,       dtype=np.int32)
        self.ionized     = np.zeros(cap,       dtype=bool)       # plasma flag

        self._cap = cap
        self.n: int = 0
        self.time: float = 0.0
        self.bonds: list[Bond] = []

        # Counters for HUD
        self.total_injected:        int = 0
        self.total_fusions:         int = 0
        self.total_bonds_formed:    int = 0
        self.total_accretions:      int = 0
        self.total_velocity_clamps: int = 0    # diagnostics: each clamp breaks momentum conservation

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
        self.ionized     = _ext1(self.ionized)
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
        self.ionized[i]     = False
        self.n += 1
        self.total_injected += 1
        return i

    def remove_particle(self, i: int) -> None:
        """Remove particle i using swap-with-last strategy. O(bonds) cost."""
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
            self.positions[i]   = self.positions[last]
            self.velocities[i]  = self.velocities[last]
            self.forces_cur[i]  = self.forces_cur[last]
            self.forces_prev[i] = self.forces_prev[last]
            self.masses[i]      = self.masses[last]
            self.elem_ids[i]    = self.elem_ids[last]
            self.bond_counts[i] = self.bond_counts[last]
            self.ionized[i]     = self.ionized[last]
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
        # 1. Inject
        self.injector.inject(self, dt)

        if self.n == 0:
            self.time += dt
            return

        n      = self.n
        pos    = self.positions[:n]
        vel    = self.velocities[:n]
        f_cur  = self.forces_cur[:n]
        f_prev = self.forces_prev[:n]
        m      = self.masses[:n]

        # 2. Velocity-Verlet first half: x(t+dt) = x(t) + v·dt + ½a·dt²
        pos += vel * dt + 0.5 * (f_cur / m[:, np.newaxis]) * dt * dt

        # 3. Recompute forces
        f_prev[:] = f_cur
        f_cur[:]  = 0.0
        physics.add_gravity(self, f_cur)
        physics.add_bond_forces(self, f_cur)
        add_thermal_pressure(self, f_cur)
        add_vdw_forces(self, f_cur)

        # 4. Velocity-Verlet second half: v(t+dt) = v(t) + ½(a_old+a_new)·dt
        vel += 0.5 * ((f_prev + f_cur) / m[:, np.newaxis]) * dt

        # 5. Soft velocity saturation
        # Replaces the previous hard cap. New rule (continuous, C¹ smooth):
        #
        #     |v_new| = v_max · tanh(|v| / v_max)
        #
        # which is identity for |v|=0, ~0.76·v_max at |v|=v_max, asymptotes
        # to v_max for |v|→∞.  Direction is preserved, so any "clamp" still
        # affects only magnitude (momentum is no longer conserved exactly
        # for fast particles — same theoretical issue as the hard cap, but
        # the discontinuity in dynamics is gone). The counter below tracks
        # particles whose unfiltered speed exceeded v_max — i.e. those that
        # would have been hard-clamped under the old rule — so HUD diagnostic
        # continuity is preserved.
        v_max  = float(self.cfg.physics.max_velocity)
        speeds = np.linalg.norm(vel, axis=1)
        would_clamp = speeds > v_max
        if np.any(would_clamp):
            self.total_velocity_clamps += int(np.count_nonzero(would_clamp))
        # ratio = |v_new| / |v| ; guard the |v|→0 limit explicitly
        ratio = np.where(speeds > 1e-12,
                         v_max * np.tanh(speeds / v_max) / np.where(speeds > 1e-12, speeds, 1.0),
                         1.0)
        vel *= ratio[:, np.newaxis]

        # 6. Ionisation flag update (must precede chemistry so ionised
        #    atoms drop their bonds this step)
        self._update_ionization()

        # 7. Chemistry: bonds + fusion + radiation kicks
        chemistry.update(self)

        # 8. Accretion: heavy inert particles merge into bodies
        accretion.update(self)

        self.time += dt

    # ------------------------------------------------------------------
    # Ionisation — per-element thresholds from real first-IE data
    # ------------------------------------------------------------------
    # A particle is flagged ionised when its kinetic energy exceeds its
    # element's measured first ionization energy (NIST), scaled by a single
    # unit-conversion factor (eV → sim KE) set in settings.yaml. Recombination
    # uses hysteresis: an ionised atom stays ionised until its KE drops below
    # recombination_fraction × its ionization threshold.
    #
    # Why this is real physics:
    #   - The relative difficulty of ionising different elements is determined
    #     entirely by their measured first-IE values (NIST). Hydrogen ionises
    #     at 13.598 eV, helium at 24.587 eV — He is genuinely ~1.8× harder.
    #   - Only one calibration constant remains (the eV→sim-KE conversion),
    #     and it exists only because the simulation runs in a custom time
    #     scaling. With consistent SI units it would vanish.
    # ------------------------------------------------------------------

    def _update_ionization(self) -> None:
        cfg = getattr(self.cfg, 'thermal', None)
        if cfg is None:
            return
        scale    = float(getattr(cfg, 'ionization_energy_scale', 0.0))
        recomb_f = float(getattr(cfg, 'recombination_fraction',  0.0))
        if scale <= 0.0:
            return

        n = self.n
        if n == 0:
            return

        ke = 0.5 * self.masses[:n] * np.sum(self.velocities[:n] ** 2, axis=1)

        # Per-particle ionization thresholds from real measured IE data
        ion_e_ev      = IONIZATION_ENERGIES_EV[self.elem_ids[:n]]
        ion_thresh    = ion_e_ev * scale
        recomb_thresh = ion_thresh * recomb_f

        currently = self.ionized[:n]
        becoming  = ke > ion_thresh
        staying   = currently & (ke > recomb_thresh)
        self.ionized[:n] = becoming | staying
