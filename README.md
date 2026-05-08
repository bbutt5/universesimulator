# Universe Simulator

A 3D real-time particle physics simulation that grows a universe from nothing — starting with empty space, injecting atoms at cosmic abundances, and letting gravity, chemistry, and nuclear fusion do the rest.

No galaxy shapes are hardcoded. Structure emerges naturally from primordial density perturbations, Hubble expansion, and gravitational collapse.

## Features

- **N-body gravity** — symmetric O(N²/2) with Newton's 3rd law, vectorised with NumPy
- **Covalent bonding** — Morse potential; atoms stay separate, held by a spring-like force
- **Nuclear fusion** — stellar nucleosynthesis chain (pp-chain → triple-alpha → alpha capture → iron peak)
- **Cosmological initial conditions** — Hubble expansion velocity and primordial density seeds produce emerging angular momentum without any hardcoded rotation
- **Real periodic table** — cosmic element abundances (H 74%, He 24%, traces of O, C, Ne, Fe, …)
- **3D vispy renderer** — CPK-coloured particles, bond lines, live HUD, interactive camera

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

Pure NumPy gravity runs smoothly at N=1000 on most modern hardware. For larger simulations, install [numba](https://numba.pydata.org/) and uncomment the `@njit` block in [`sim/physics.py`](sim/physics.py) for a 10–50× speedup.

## Architecture

```
main.py          — entry point, config loading
settings.yaml    — all tunable parameters
sim/
  elements.py    — periodic table data, fusion reaction table
  particle.py    — Bond class (Morse potential)
  world.py       — simulation state, velocity-Verlet integrator
  physics.py     — gravity + bond forces (NumPy vectorised)
  chemistry.py   — bond formation/breaking, nuclear fusion
  injector.py    — particle birth (position, velocity, element sampling)
  viewer.py      — vispy 3D renderer + keyboard controls
```

## License

MIT — see [LICENSE](LICENSE).
