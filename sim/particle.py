"""
Bond dataclass — represents a covalent bond between two particles.
Uses a Morse potential so atoms stay as separate entities (they are
pulled/pushed by the bond force but never merged).

Morse potential:
    V(r) = De * (1 - exp(-a*(r - re)))²

Force on particle i (radial component, positive = repulsive):
    F(r) = -dV/dr = -2*De*a*(1 - exp(-a*(r-re))) * exp(-a*(r-re))

    > 0  when r < re  (repulsive — atoms too close)
    < 0  when r > re  (attractive — atoms too far apart)
    = 0  at r = re    (equilibrium)
"""

from __future__ import annotations
import numpy as np


class Bond:
    __slots__ = ('i', 'j', 'length_eq', 'D_e', 'a', 'order')

    def __init__(
        self,
        i: int,
        j: int,
        length_eq: float,   # equilibrium length = rc_i + rc_j  (SU)
        D_e: float,         # dissociation energy (simulation energy units)
        k: float,           # spring constant at well bottom (energy/SU²)
        order: int = 1,
    ):
        self.i = i
        self.j = j
        self.length_eq = length_eq
        self.D_e = D_e
        # Morse parameter derived from harmonic approximation: k = 2*De*a²
        self.a = np.sqrt(k / (2.0 * D_e)) if D_e > 1e-12 else 1.0
        self.order = order

    def radial_force(self, r: float) -> float:
        """Radial force magnitude. Negative = attractive (toward other atom)."""
        x     = r - self.length_eq
        exp_t = np.exp(-self.a * x)
        return -2.0 * self.D_e * self.a * (1.0 - exp_t) * exp_t

    def potential_energy(self, r: float) -> float:
        x     = r - self.length_eq
        exp_t = np.exp(-self.a * x)
        return self.D_e * (1.0 - exp_t) ** 2
