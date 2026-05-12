# Universe Simulator

A 3D real-time particle physics simulation that grows a universe from nothing — starting with empty space, injecting atoms at cosmic abundances, and letting four independent emergent mechanisms produce structure.

No galaxy shapes, planet locations, or stellar lifecycles are hardcoded. They fall out of:

1. **N-body gravity** with primordial density seeds and Hubble expansion (initial conditions only) — drives long-range structure formation.
2. **Covalent chemistry** (Pauling bond energies from real electronegativity data) — atoms bind into molecules in cold dense regions.
3. **Nuclear fusion** (Coulomb-barrier gated, Q-values from AME 2020 isotope masses) — stellar nucleosynthesis chain: pp-chain → triple-alpha → alpha-capture → iron peak.
4. **Gravitational accretion** of bound non-fusing pairs — converts the binding-energy excess into solid planetesimals and rocky bodies. This is a *separate* mechanism from gravity, doing the work of coagulation that real molecular clouds and protoplanetary disks rely on.

The simulator surfaces conservation-law diagnostics in its HUD (total energy + drift %, velocity-clamp counter) so you can verify the symplectic integrator isn't accumulating error and the velocity cap isn't silently breaking momentum.

**Project principle:** prefer emergent / derived-from-physics approaches over hardcoded constants, tables, and flags. Every numerical value in `sim/elements.py` is a real measured quantity (NIST, AME, CRC) with a citation in the source. See [PHILOSOPHY.md](PHILOSOPHY.md).

## Features

- **N-body gravity** — symmetric O(N²/2) with Newton's 3rd law, vectorised with NumPy. Hits a wall around N=2000 in pure NumPy; see Performance.
- **Covalent bonding** — Pauling bond energies (`D(A-B) = √(D_AA·D_BB) + 96·Δχ²`, kJ/mol) with measured electronegativities; Morse potential drives the force.
- **Nuclear fusion** — Coulomb-barrier gate (`V_C = scale·Z₁Z₂ / r_nuc`, real r₀ = 1.2 fm from Krane); Q-values from AME 2020 isotope masses; energy-conservation gate (rel_KE + Q ≥ 0). The chain that emerges: pp-chain → triple-alpha → alpha-capture → silicon-burning → iron peak.
- **Gravitational accretion** — bound non-fusing pairs merge, conserving momentum and inheriting the heavier element's identity. This is the *coagulation* mechanism that produces planetesimals (mostly heavy elements) while light hot pairs defer to fusion.
- **Per-element ionisation** — first-IE values from NIST drive each particle's plasma threshold; hysteresis on recombination matches real plasma physics.
- **Four states of matter** — solid / liquid / gas / plasma classified from local density + relative-KE + bonding + ionisation. Live counts in the HUD.
- **Conservation diagnostics** — total KE + PE in the HUD with drift % since t=0 (velocity-Verlet is symplectic, so non-zero drift means dt is too large or softening is too small). Velocity-clamp counter exposes how often the numerical max-speed band-aid fires.
- **Cosmological initial conditions** — Hubble expansion velocity and primordial density seeds; angular momentum from tidal torques, no hardcoded rotation.
- **Real periodic table** — masses from AME 2020 (dominant isotope), covalent radii from Cordero 2008, electronegativities from Pauling, bond energies from CRC Handbook, ionisation energies from NIST ASD. Every value cited in `sim/elements.py`.
- **3D vispy renderer with real radiative output** — every halo is driven by per-particle Stefan-Boltzmann luminosity (radius ∝ T²) and rendered in real Planck-blackbody RGB (Tanner Helland's CIE fit). Press `P` to switch the core colour from CPK chemistry convention to the same physical blackbody. Effective temperatures come from kinetic energy (gas) and the stellar mass-luminosity relation T ∝ √M (accreted bodies). No starfield, no painted backdrop — the void is empty, as it should be.

## Requirements

- Python 3.11+
- [vispy](https://vispy.org/)
- NumPy
- PyYAML

```bash
pip install vispy numpy pyyaml
```

> **Note:** If you use conda, install dependencies into your environment first or use a virtualenv:
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Windows: .venv\Scripts\activate
> pip install vispy numpy pyyaml
> ```

## Usage

```bash
python main.py
python main.py --settings path/to/settings.yaml
```

### Controls

| Key / Mouse             | Action                  |
|-------------------------|-------------------------|
| Left-drag               | Orbit camera            |
| Right-drag / Scroll     | Zoom                    |
| Middle-drag             | Pan                     |
| Arrow keys              | Orbit (5° per press)    |
| Page Up / Page Down     | Zoom in / out           |
| Space                   | Pause / resume          |
| `+` / `-`               | Speed up / slow down    |
| `R`                     | Reset camera to default |
| `Q` / `Escape`          | Quit                    |

## Configuration

All parameters are in [`settings.yaml`](settings.yaml). Key sections:

```yaml
simulation:
  injection_rate: 100        # atoms per real second
  max_particles: 1000        # cap (raise with numba installed)

cosmology:
  hubble_initial: 0.006      # Big Bang outward velocity per distance unit
  n_density_seeds: 8         # primordial overdense regions

injection:
  elements:
    H:  73.9                 # cosmic abundances by particle fraction
    He: 24.0
    # …
```

## Performance

Pure NumPy gravity runs smoothly at N=1000 on most modern hardware. For larger simulations:

- **N=1k–10k:** install [numba](https://numba.pydata.org/) — the gravity backend auto-detects it and switches to a parallel JIT-compiled inner loop with no code changes required:
  ```bash
  pip install numba
  ```
  Measured speedup on this codebase (Apple Silicon laptop):

  | N      | NumPy   | numba   | speedup |
  |-------:|--------:|--------:|--------:|
  | 100    | 0.73 ms | 0.11 ms | 6.3×    |
  | 500    | 6.43 ms | 0.19 ms | 33.8×   |
  | 1000   | 19.6 ms | 0.57 ms | 34.2×   |
  | 2000   | 70.4 ms | 1.35 ms | 52.0×   |

  Both backends produce numerically identical force fields (verified to 1e-10 in `tests/test_physics_gravity.py`).
- **N=10k+:** the O(N²) gravity becomes the bottleneck even with numba. You'd want a Barnes–Hut tree (O(N log N)). Not implemented yet — open for contribution.

## Architecture

```
main.py          — entry point, config loading
settings.yaml    — all tunable parameters
sim/
  elements.py    — periodic table (real measured data + citations)
  nuclear.py     — Coulomb barrier + Q-value formulae for fusion
  particle.py    — Bond class (Morse potential)
  spatial.py     — O(1) neighbour-lookup grid (shared by chemistry, thermal, vdw, accretion)
  physics.py     — gravity + bond forces (NumPy vectorised)
  thermal.py     — thermal pressure (kinetic-pressure analogue)
  vdw.py         — van der Waals attraction (cold-pair condensation)
  chemistry.py   — bond formation/breaking, nuclear fusion + radiation kicks
  accretion.py   — gravitational coagulation of bound non-fusing pairs
  states.py      — solid / liquid / gas / plasma classifier
  diagnostics.py — energy + momentum conservation diagnostics
  injector.py    — particle birth (position, velocity, element sampling)
  world.py       — simulation state, velocity-Verlet integrator, step loop
  viewer.py      — vispy 3D renderer + keyboard / mouse controls
```

## License

MIT — see [LICENSE](LICENSE).
