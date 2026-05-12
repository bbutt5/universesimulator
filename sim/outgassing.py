"""
Outgassing — temperature-driven release of light volatiles from hot bodies.

Real bodies emit gas continuously from their surfaces:
  - Hot stars  : stellar winds eject ~10⁻¹⁴ M☉/year (Sun) to 10⁻⁵ M☉/year (O-class)
  - Terrestrial: atmospheric Jeans escape; Earth loses ~3 kg/s of hydrogen
  - Comets     : H₂O / CO₂ outgassing as they approach perihelion

We model this as a per-step probabilistic ejection of one light atom from
each accreted body whose surface temperature exceeds a threshold. The
ejected atom is placed at the body's surface (cube-root-of-mass radius)
with the body's bulk velocity plus an outward radial kick at the body's
escape velocity:

    v_esc  =  √( 2·G·M / R )           (real Jeans escape condition)

The body's composition record and total mass are decremented by the
emitted atom's mass, so total mass + momentum of the (body + escapee)
system are conserved.

This couples directly into Phases 1-3:
  * a hot rocky planet sheds its primordial H/He envelope over time
    (real exoplanet science: hot Neptunes lose atmospheres)
  * a star continuously throws H/He back into the surrounding gas, which
    can then recondense or fuse elsewhere
  * the cosmic-abundance ratio in the gas phase shifts toward heavier
    elements as bodies sequester them
"""

from __future__ import annotations
import numpy as np

from sim.elements import ELEMENTS_LIST


def update(world) -> None:
    cfg = getattr(world.cfg, 'outgassing', None)
    if cfg is None:
        return

    rate          = float(getattr(cfg, 'rate_per_step',          0.0))
    T_threshold_K = float(getattr(cfg, 'temperature_threshold_K', 1000.0))
    light_mass_lim = float(getattr(cfg, 'light_element_mass_max', 5.0))

    if rate <= 0.0 or world.n == 0:
        return

    rend_cfg = getattr(world.cfg, 'renderer', None)
    if rend_cfg is None:
        return
    planet_threshold = float(getattr(rend_cfg, 'planet_mass_threshold',     20.0))
    star_threshold   = float(getattr(rend_cfg, 'star_mass_threshold',     200.0))
    hot_T_sim        = float(getattr(rend_cfg, 'hot_temperature_threshold', 5e5))
    ref_T_K          = float(getattr(rend_cfg, 'reference_temperature_K',  20000.0))

    G = float(world.cfg.physics.gravity_constant)

    n = world.n
    elem_masses_arr = np.array(
        [ELEMENTS_LIST[world.elem_ids[i]].mass for i in range(n)],
        dtype=np.float64,
    )
    mass_ratios = world.masses[:n] / elem_masses_arr
    body_mask = mass_ratios >= planet_threshold
    if not body_mask.any():
        return

    # Effective surface temperature (max of kinetic proxy and stellar mass-
    # luminosity relation), same logic as the renderer.
    ke = 0.5 * world.masses[:n] * np.sum(world.velocities[:n] ** 2, axis=1)
    T_K_kinetic = (ke / hot_T_sim) * ref_T_K
    T_K_mass    = ref_T_K * np.sqrt(np.maximum(mass_ratios / star_threshold, 0.0))
    T_K         = np.maximum(T_K_kinetic, T_K_mass)

    # Indices of "light" elements in ELEMENTS_LIST
    light_ids = [i for i, e in enumerate(ELEMENTS_LIST) if 0.0 < e.mass < light_mass_lim]
    if not light_ids:
        return

    rng = np.random.default_rng(int(world.time * 1e6) + 42)

    bodies = np.where(body_mask)[0]
    emissions: list[tuple[int, int]] = []   # (body_idx, light_elem_id)

    for idx in bodies:
        T = T_K[idx]
        if T <= T_threshold_K:
            continue

        excess = (T - T_threshold_K) / T_threshold_K
        p_emit = min(1.0, rate * excess)
        if rng.random() >= p_emit:
            continue

        # Pick a light element from this body's composition, weighted by stock.
        # Require enough mass to emit one full atom.
        candidate_ids: list[int] = []
        weights:       list[float] = []
        for elem_id in light_ids:
            available = world.composition[idx, elem_id]
            atom_mass = ELEMENTS_LIST[elem_id].mass
            if available >= atom_mass:
                candidate_ids.append(elem_id)
                weights.append(float(available))
        if not candidate_ids:
            continue
        w_arr = np.array(weights, dtype=np.float64)
        w_arr /= w_arr.sum()
        chosen = candidate_ids[int(rng.choice(len(candidate_ids), p=w_arr))]
        emissions.append((idx, chosen))

    if not emissions:
        return

    for body_idx, elem_id in emissions:
        elem = ELEMENTS_LIST[elem_id]

        # Body's effective radius: cube-root-of-mass scaling, with a floor of
        # one covalent radius so we don't divide by zero for tiny bodies.
        ratio = world.masses[body_idx] / elem_masses_arr[body_idx]
        body_radius = max(elem.covalent_radius, 5.0 * ratio ** (1.0 / 3.0))

        v_esc = float(np.sqrt(2.0 * G * world.masses[body_idx] / body_radius))

        # Random outward direction
        direction = rng.standard_normal(3)
        direction /= max(np.linalg.norm(direction), 1e-9)

        emit_pos = world.positions[body_idx] + direction * body_radius
        emit_vel = world.velocities[body_idx] + direction * v_esc

        # Conservation: subtract emitted mass from body composition + total mass
        world.composition[body_idx, elem_id] -= elem.mass
        world.masses[body_idx]               -= elem.mass

        world.add_particle(elem, emit_pos.copy(), emit_vel.copy())
        world.total_outgassing += 1
