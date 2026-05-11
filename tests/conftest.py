"""Shared test fixtures."""

from __future__ import annotations
import pytest
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


@pytest.fixture
def cfg():
    """Minimal simulation config covering all modules."""
    return _to_ns({
        'simulation': {
            'injection_rate': 10,
            'injection_radius_initial': 500.0,
            'injection_radius_expansion': 0.0,
            'injection_velocity_max': 5.0,
            'time_step': 0.01,
            'steps_per_frame': 1,
            'max_particles': 200,
        },
        'physics': {
            'gravity_constant': 100.0,
            'softening_length': 1.0,
            'max_velocity': 1e6,
        },
        'thermal': {
            'pressure_constant': 0.4,
            'pressure_cutoff': 200.0,
            'vdw_strength': 25.0,
            'vdw_cutoff':   300.0,
            # Scale chosen so H ionises at KE = 13.598 × 7355 ≈ 100k sim units,
            # matching the pre-physicalisation test calibration. He's threshold
            # is now 24.587 × 7355 ≈ 181k, automatically — real NIST physics.
            'ionization_energy_scale': 7355.0,
            'recombination_fraction':  0.1,
            'plasma_repulsion_factor': 2.0,
        },
        'chemistry': {
            'bond_formation_factor': 1.4,
            'bond_velocity_threshold': 1000.0,
            'bond_spring_constant': 0.8,
            'bond_de_scale': 80.0,
            'bond_break_factor': 2.8,
            'fusion_ke_threshold': 1.0,
            'fusion_enabled': True,
            'radiation_radius': 500.0,
            'radiation_energy_scale': 100.0,
        },
        'accretion': {
            'accretion_radius': 150.0,
        },
        'cosmology': {
            'hubble_initial': 0.01,
            'n_density_seeds': 3,
            'perturbation_amplitude': 2.0,
            'perturbation_scale': 0.2,
        },
        'injection': {
            'elements': {'H': 90.0, 'He': 10.0},
        },
    })
