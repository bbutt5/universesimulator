"""
Particle injector — spawns atoms into the void on a configurable schedule.

Initial conditions applied at birth (no ongoing forces):

  1. Expanding injection zone
     The sphere grows linearly with sim-time, starting near zero.
     Particles born earlier live in a denser, smaller universe.

  2. Hubble initial velocity  (cosmology.hubble_initial)
     Each particle is born with a small outward velocity proportional to its
     distance from the origin:
         v_hubble = hubble_initial * pos
     This is the Big Bang initial condition — the universe was born expanding.
     After injection, only gravity governs the particle.

  3. Primordial density perturbations  (cosmology.n_density_seeds)
     A fixed set of overdense seed regions (generated once, stored as
     co-moving fractions of the injection sphere) biases where particles
     are born.  This mimics the quantum-fluctuation seeds from inflation.
     Gravity amplifies the seeds → separate collapsing clouds → tidal
     torques between neighbouring overdensities → angular momentum emerges
     naturally (tidal torque theory).  Nothing is hardcoded after birth.
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS


class Injector:
    """Manages particle injection schedule and birth-condition sampling."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._rng = np.random.default_rng()
        self._dist_cache: tuple | None = None
        self._seeds: np.ndarray | None = None

    # ---------------------------------------------------------------------------
    # Element distribution cache
    # ---------------------------------------------------------------------------

    def _get_distribution(self) -> tuple:
        if self._dist_cache is None:
            raw = dict(self._cfg.injection.elements.__dict__)
            symbols = [s for s in raw if s in ELEMENTS]
            weights = np.array([raw[s] for s in symbols], dtype=np.float64)
            weights /= weights.sum()
            self._dist_cache = (symbols, weights)
        return self._dist_cache

    # ---------------------------------------------------------------------------
    # Density seed positions (co-moving fractions, generated once)
    # ---------------------------------------------------------------------------

    def _get_seeds(self) -> np.ndarray:
        """Return seed positions as fractions of the injection sphere (0–1 magnitude)."""
        if self._seeds is not None:
            return self._seeds

        cosmo = getattr(self._cfg, 'cosmology', None)
        n = int(getattr(cosmo, 'n_density_seeds', 0)) if cosmo else 0

        if n == 0:
            self._seeds = np.empty((0, 3))
            return self._seeds

        seeds = []
        while len(seeds) < n:
            v = self._rng.standard_normal(3)
            v /= np.linalg.norm(v)
            # Place seeds between 10%–80% of the sphere radius so they are
            # well inside the volume (not on the boundary or at the centre)
            r = self._rng.uniform(0.1, 0.8) ** (1.0 / 3.0)
            seeds.append(v * r)

        self._seeds = np.array(seeds)   # (n, 3) — unit-sphere co-moving coords
        return self._seeds

    # ---------------------------------------------------------------------------
    # Position sampling
    # ---------------------------------------------------------------------------

    def _uniform_sphere_volume(self, radius: float) -> np.ndarray:
        """Uniformly distributed point inside a sphere."""
        v = self._rng.standard_normal(3)
        v /= np.linalg.norm(v)
        r = radius * self._rng.uniform(0.0, 1.0) ** (1.0 / 3.0)
        return v * r

    def _sample_position(self, radius: float) -> np.ndarray:
        """
        Sample a birth position.

        With density perturbations enabled, a particle is born either:
          - In the smooth background  (weight 1)
          - Near one of the overdense seeds  (weight perturbation_amplitude each)

        The seeds are stored as co-moving fractions and scaled by the current
        injection radius, so they expand with the universe naturally.
        """
        cosmo = getattr(self._cfg, 'cosmology', None)
        seeds = self._get_seeds()

        if cosmo is None or len(seeds) == 0:
            return self._uniform_sphere_volume(radius)

        amp     = float(cosmo.perturbation_amplitude)
        scale   = float(cosmo.perturbation_scale) * radius
        n_seeds = len(seeds)

        # Mixture weight: background=1, each seed=amp
        total_w = 1.0 + amp * n_seeds
        draw    = self._rng.uniform(0.0, total_w)

        if draw < 1.0:
            # Background — uniform inside sphere
            return self._uniform_sphere_volume(radius)

        # Overdense seed region
        seed_idx = int((draw - 1.0) / amp) % n_seeds
        seed_pos = seeds[seed_idx] * radius          # scale to current injection sphere

        # Offset from seed centre drawn from an isotropic Gaussian
        offset = self._rng.standard_normal(3) * scale
        pos    = seed_pos + offset

        # Clamp to injection sphere (keeps particles inside the born universe)
        norm = np.linalg.norm(pos)
        if norm > radius:
            pos *= radius / norm

        return pos

    # ---------------------------------------------------------------------------
    # Velocity sampling
    # ---------------------------------------------------------------------------

    def _sample_velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Thermal random velocity + Hubble initial condition.

        The Hubble term v = H * pos is an initial condition (set once at birth),
        not a force.  It represents the outward motion matter inherited from the
        Big Bang.  After this, only gravity and chemistry govern the particle.
        """
        v_max = self._cfg.simulation.injection_velocity_max

        # Thermal / random component
        direction = self._rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        speed = self._rng.uniform(0.0, v_max)
        vel   = direction * speed

        # Hubble initial condition
        cosmo = getattr(self._cfg, 'cosmology', None)
        if cosmo is not None:
            H = float(getattr(cosmo, 'hubble_initial', 0.0))
            if H > 0.0:
                vel += H * pos   # outward velocity proportional to distance

        return vel

    # ---------------------------------------------------------------------------
    # Main injection method
    # ---------------------------------------------------------------------------

    def inject(self, world, dt: float) -> None:
        """Inject particles proportional to dt × injection_rate."""
        cfg   = self._cfg
        max_n = cfg.simulation.max_particles
        if world.n >= max_n:
            return

        world._inject_acc += cfg.simulation.injection_rate * dt
        n_inject = int(world._inject_acc)
        world._inject_acc -= n_inject
        n_inject = min(n_inject, max_n - world.n)
        if n_inject <= 0:
            return

        symbols, weights = self._get_distribution()

        # Current injection radius (universe expanding)
        sim    = cfg.simulation
        radius = sim.injection_radius_initial + sim.injection_radius_expansion * world.time

        chosen = self._rng.choice(symbols, size=n_inject, p=weights)
        for sym in chosen:
            pos = self._sample_position(radius)
            vel = self._sample_velocity(pos)
            world.add_particle(ELEMENTS[sym], pos, vel)
