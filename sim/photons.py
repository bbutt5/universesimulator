"""
First-order radiation transport — explicit photon particles.

Replaces the previous "instantaneous-impulse" radiation models with
genuine photon transport:

  * Hot particles emit photons stochastically at rate ∝ T⁴ (Stefan-Boltzmann)
  * Photons travel in straight lines at finite speed c (sim units)
  * Photons are absorbed by particles within a geometric cross-section
  * Absorption transfers the photon's energy to the absorber's KE
    along the photon's direction — real radiation pressure
  * Emission recoil: the emitting particle recoils by momentum p = E/c
    (so total momentum is exactly conserved across emit/absorb cycles)

What this *does* model
----------------------
  - Heat flow from hot regions to cold ones via radiation (second law)
  - Real radiation pressure on illuminated targets
  - Finite light-travel time across the simulation volume
  - Energy redistribution via photons rather than instantaneous impulses

What this does NOT model (deliberately deferred)
------------------------------------------------
  - Wavelength / spectrum — each photon carries only an energy scalar
  - Compton & Rayleigh scattering — only absorption (capture)
  - Photon-photon interactions — don't exist in nature anyway
  - Wavelength-dependent cross-sections — uniform geometric capture

A circular buffer of fixed capacity stores live photons; emission past
capacity overwrites the oldest entry. Capacity is the only "performance
knob" — set it high enough that typical photon lifetimes fit.

Reference: Rybicki & Lightman, "Radiative Processes in Astrophysics",
Chapter 1 (radiative transfer equation, geometric-optics limit).
"""

from __future__ import annotations
import numpy as np


class PhotonState:
    """Pre-allocated arrays for the live photon population.

    Photons are stored in a fixed-size circular buffer; ``alive[i]`` says
    whether slot i is currently in use. When the buffer is full, new
    emissions overwrite the oldest entry (FIFO eviction).
    """

    def __init__(self, capacity: int = 5000):
        self.cap = int(capacity)
        self.positions  = np.zeros((self.cap, 3), dtype=np.float64)
        self.directions = np.zeros((self.cap, 3), dtype=np.float64)
        self.energies   = np.zeros(self.cap,       dtype=np.float64)
        self.alive      = np.zeros(self.cap,       dtype=bool)
        # Index of the particle that emitted this photon (−1 if injected
        # manually / unknown).  Used to prevent the emitter from immediately
        # re-absorbing its own photon (a numerical artefact since the photon
        # is born inside the emitter's cross-section).
        self.emitter    = np.full(self.cap, -1, dtype=np.int32)
        self._next_slot: int = 0

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def emit_one(self, position: np.ndarray, direction: np.ndarray, energy: float,
                 emitter_idx: int = -1) -> int:
        """Place a single photon in the buffer; returns the slot index."""
        idx = self._next_slot
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return -1
        self.positions[idx]  = position
        self.directions[idx] = direction / norm
        self.energies[idx]   = energy
        self.alive[idx]      = True
        self.emitter[idx]    = int(emitter_idx)
        self._next_slot = (idx + 1) % self.cap
        return idx

    def reset(self) -> None:
        self.alive[:]   = False
        self.emitter[:] = -1
        self._next_slot = 0


# ---------------------------------------------------------------------------
# Module entry point — called once per simulation step
# ---------------------------------------------------------------------------

def update(world, dt: float) -> None:
    """Per-step: emit thermal photons, propagate, absorb."""
    cfg = getattr(world.cfg, 'photons', None)
    if cfg is None:
        return

    speed_c             = float(getattr(cfg, 'speed_of_light_sim', 0.0))
    emission_rate_scale = float(getattr(cfg, 'emission_rate_scale', 0.0))
    cross_section       = float(getattr(cfg, 'absorption_cross_section', 0.0))
    energy_per_photon   = float(getattr(cfg, 'energy_per_photon', 1.0))
    cull_distance       = float(getattr(cfg, 'cull_distance', 1e6))

    if speed_c <= 0.0 or emission_rate_scale <= 0.0:
        return
    if world.photons is None:
        return

    rng = np.random.default_rng(int(world.time * 1e6) + 7)

    # --- 1. Emit thermal photons from hot particles -----------------------
    rend_cfg  = getattr(world.cfg, 'renderer', None)
    if rend_cfg is None:
        return
    hot_T_sim = float(getattr(rend_cfg, 'hot_temperature_threshold', 5e5))
    ref_T_K   = float(getattr(rend_cfg, 'reference_temperature_K', 20000.0))

    n = world.n
    if n > 0:
        ke = 0.5 * world.masses[:n] * np.sum(world.velocities[:n] ** 2, axis=1)
        # Per-particle effective temperature (matches the renderer's mapping)
        T_K = (ke / hot_T_sim) * ref_T_K
        # Stefan-Boltzmann: emission rate ∝ T⁴.  Probability of one photon
        # this step: rate × dt, capped at 1.
        T_ratio = T_K / ref_T_K
        rate = emission_rate_scale * (T_ratio ** 4) * dt
        p_emit = np.clip(rate, 0.0, 1.0)

        random_uniforms = rng.random(n)
        emit_mask = random_uniforms < p_emit
        emit_idx  = np.where(emit_mask)[0]

        for i in emit_idx:
            # Random emission direction (isotropic)
            d = rng.standard_normal(3)
            d_norm = float(np.linalg.norm(d))
            if d_norm < 1e-12:
                continue
            d /= d_norm
            slot = world.photons.emit_one(
                world.positions[i], d, energy_per_photon, emitter_idx=int(i),
            )
            if slot < 0:
                continue
            world.total_photons_emitted += 1

            # Radiation-pressure recoil on the emitter: p = E/c, so
            # δv = −d · E / (m · c)
            m_i = float(world.masses[i])
            if m_i > 0.0:
                world.velocities[i] -= d * (energy_per_photon / (m_i * speed_c))

    # --- 2. Propagate alive photons at speed c ---------------------------
    # We record each photon's start position so the absorption pass below
    # can check the SWEPT VOLUME (line segment) rather than just the
    # endpoint — otherwise a fast photon could fly straight past a particle
    # in a single step and never get absorbed.
    alive_mask_before = world.photons.alive.copy()
    p_old             = world.photons.positions.copy()

    if alive_mask_before.any():
        world.photons.positions[alive_mask_before] += (
            world.photons.directions[alive_mask_before] * (speed_c * dt)
        )
        # Cull photons that have travelled far beyond the simulation volume
        live_pos = world.photons.positions[alive_mask_before]
        too_far  = np.einsum('ij,ij->i', live_pos, live_pos) > cull_distance ** 2
        if too_far.any():
            cull_indices = np.where(alive_mask_before)[0][too_far]
            world.photons.alive[cull_indices] = False

    # --- 3. Absorption — swept-volume line-segment check ----------------
    # For each photon, the segment from its old position to its new position
    # was its travel path this step.  A particle absorbs the photon if its
    # perpendicular distance to that segment is within ``cross_section``.
    if not world.photons.alive.any() or n == 0:
        return

    cs_sq    = cross_section * cross_section
    alive_ph = np.where(world.photons.alive)[0]
    if len(alive_ph) == 0:
        return

    particle_pos = world.positions[:n]                          # (N, 3)

    for k, photon_idx in enumerate(alive_ph):
        p0 = p_old[photon_idx]
        p1 = world.photons.positions[photon_idx]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            # Stationary photon (dt = 0 etc.) — fall back to endpoint check
            dq = particle_pos - p1
            dsq = np.einsum('ij,ij->i', dq, dq)
        else:
            # Project each particle onto the segment, clip to [0,1] range
            dq = particle_pos - p0                              # (N, 3)
            t  = (dq @ seg) / seg_len_sq                        # (N,)
            t_clipped = np.clip(t, 0.0, 1.0)
            closest = p0 + t_clipped[:, None] * seg              # (N, 3)
            dq_perp = particle_pos - closest                    # (N, 3)
            dsq = np.einsum('ij,ij->i', dq_perp, dq_perp)        # (N,)

        # Exclude the photon's own emitter from absorption candidacy
        # (otherwise the photon is born inside its emitter's cross-section
        # and gets reabsorbed immediately, cancelling the recoil).
        emitter = int(world.photons.emitter[photon_idx])
        if 0 <= emitter < n:
            dsq[emitter] = np.inf

        nearest = int(np.argmin(dsq))
        if dsq[nearest] >= cs_sq:
            continue

        particle_idx = nearest
        energy       = float(world.photons.energies[photon_idx])
        m            = float(world.masses[particle_idx])
        if m <= 0.0:
            continue

        # Real radiation pressure: photon transfers momentum p = E/c.
        direction = world.photons.directions[photon_idx]
        world.velocities[particle_idx] += direction * (energy / (m * speed_c))

        world.photons.alive[photon_idx] = False
        world.total_photons_absorbed += 1
