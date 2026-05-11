"""
Conservation-law diagnostics for the simulator.

These functions do not change behaviour — they let you verify that the
symplectic integrator is doing its job. A correctly-implemented
velocity-Verlet pass should produce total energy that *oscillates*
around a constant value but does NOT drift over many steps; total
linear momentum should remain exactly constant in the absence of
external forces.

Interpreting "drift_pct" in production
--------------------------------------
The drift metric only measures *integrator* fidelity when the
non-conservative events are disabled. In a normal run with
fusion / accretion / injection / radiation kicks enabled, total
energy WILL change — those processes literally add or remove energy
from the kinetic+potential book:

  - Fusion releases mass-defect energy (Q × c²) via radiation kicks → +KE
  - Accretion merges silently absorb relative kinetic energy → −KE
  - Particle injection adds fresh KE+PE per particle → ±E
  - Bond formation / breaking is conservative if Morse is implemented
    correctly (energy moves between KE and bond-PE)

So in a default 500-step production run, drift of hundreds-of-percent
is *expected* and reflects physics. To diagnose the integrator alone,
disable fusion/accretion/injection (the test fixtures do this) and
verify drift stays below ~0.1% over many steps. The included tests in
test_diagnostics.py do exactly that and pass at 0.001%.

If you DO observe drift in a pure-gravity run, the time step is too
large or the softening length is too small relative to the closest
particle approaches. If you observe momentum drift, the velocity
clamp is firing (it silently rescales magnitude — see
`World.total_velocity_clamps`) or a force pair has lost its Newton-3
partner.

All values are returned in the simulator's native units (KE has units of
amu·(SU/s)²; gravitational PE uses the simulator's amplified G).
"""

from __future__ import annotations
import numpy as np


def kinetic_energy(world) -> float:
    """Σᵢ ½·mᵢ·|vᵢ|² over active particles."""
    n = world.n
    if n == 0:
        return 0.0
    v_sq = np.einsum('ij,ij->i', world.velocities[:n], world.velocities[:n])
    return 0.5 * float(np.sum(world.masses[:n] * v_sq))


def gravitational_pe(world) -> float:
    """Σᵢ<ⱼ −G·mᵢ·mⱼ / √(r² + ε²) — same softening as the force calculation.

    O(N²). Slow for large N — diagnostic use only.
    """
    n = world.n
    if n < 2:
        return 0.0

    G         = float(world.cfg.physics.gravity_constant)
    softening = float(getattr(world.cfg.physics, 'softening_length', 0.0))
    eps_sq    = softening * softening

    pos = world.positions[:n]
    m   = world.masses[:n]

    diff = pos[:, None, :] - pos[None, :, :]
    r_sq = np.einsum('ijk,ijk->ij', diff, diff) + eps_sq
    np.fill_diagonal(r_sq, np.inf)               # skip self-pairs
    r    = np.sqrt(r_sq)
    mm   = np.outer(m, m)
    return -0.5 * G * float(np.sum(mm / r))      # ½ to avoid double-count


def total_energy(world) -> float:
    """KE + PE (no thermal/bond/VdW potential terms — these are bounded and
    don't affect long-term drift diagnosis)."""
    return kinetic_energy(world) + gravitational_pe(world)


def linear_momentum(world) -> np.ndarray:
    """Total Σ mᵢ·vᵢ. Should be exactly conserved unless the velocity clamp
    fires."""
    n = world.n
    if n == 0:
        return np.zeros(3)
    return np.sum(world.velocities[:n] * world.masses[:n, None], axis=0)


def conservation_report(world, reference_energy: float | None = None) -> dict:
    """Return a snapshot of conserved quantities + drift since reference.

    Pass the energy from a prior step as `reference_energy` to compute
    drift percentage; useful for monitoring integrator stability over a run.
    """
    ke = kinetic_energy(world)
    pe = gravitational_pe(world)
    e  = ke + pe
    p  = linear_momentum(world)
    out = {
        'kinetic':   ke,
        'potential': pe,
        'total':     e,
        '|momentum|': float(np.linalg.norm(p)),
        'clamps':    int(world.total_velocity_clamps),
    }
    if reference_energy is not None and reference_energy != 0:
        out['drift_pct'] = 100.0 * (e - reference_energy) / abs(reference_energy)
    return out
