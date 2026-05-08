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
    """Minimal simulation config for testing."""
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
        'chemistry': {
            'bond_formation_factor': 1.4,
            'bond_velocity_threshold': 1000.0,
            'bond_spring_constant': 0.8,
            'bond_de_scale': 80.0,
            'bond_break_factor': 2.8,
            'fusion_ke_threshold': 1.0,
            'fusion_enabled': True,
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
