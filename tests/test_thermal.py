"""Tests for sim/thermal.py — thermal pressure forces."""

from __future__ import annotations
import numpy as np
import pytest
from sim.world import World
from sim.elements import ELEMENTS
from sim.thermal import add_thermal_pressure


def _place(world, sym: str, pos, vel=None):
    if vel is None:
        vel = np.zeros(3)
    return world.add_particle(ELEMENTS[sym], np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestThermalPressure:
    def test_repulsive_for_hot_close_pair(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0,   0.0, 0.0], vel=[ 50.0, 0.0, 0.0])
        _place(w, 'H', [50.0,  0.0, 0.0], vel=[-50.0, 0.0, 0.0])  # high relative KE
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        # particle 0 should be pushed away from particle 1 (−x direction)
        assert forces[0, 0] < 0
        # particle 1 should be pushed away from particle 0 (+x direction)
        assert forces[1, 0] > 0

    def test_newton_third_law(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0], vel=[ 30.0, 0.0, 0.0])
        _place(w, 'H', [80.0, 0.0, 0.0], vel=[-30.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-12)

    def test_no_force_when_at_rest_relative_to_each_other(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0], vel=[5.0, 5.0, 5.0])
        _place(w, 'H', [80.0, 0.0, 0.0], vel=[5.0, 5.0, 5.0])  # same velocity
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_no_force_beyond_cutoff(self, cfg):
        w = World(cfg)
        cutoff = cfg.thermal.pressure_cutoff
        _place(w, 'H', [0.0,           0.0, 0.0], vel=[ 50.0, 0.0, 0.0])
        _place(w, 'H', [cutoff + 10.0, 0.0, 0.0], vel=[-50.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_force_stronger_when_closer(self, cfg):
        def _force_at(sep):
            w = World(cfg)
            _place(w, 'H', [0.0, 0.0, 0.0], vel=[ 50.0, 0.0, 0.0])
            _place(w, 'H', [sep, 0.0, 0.0], vel=[-50.0, 0.0, 0.0])
            forces = np.zeros((w.n, 3))
            add_thermal_pressure(w, forces)
            return abs(forces[0, 0])

        assert _force_at(30.0) > _force_at(80.0)

    def test_force_stronger_with_higher_relative_velocity(self, cfg):
        def _force_at_speed(v):
            w = World(cfg)
            _place(w, 'H', [0.0,  0.0, 0.0], vel=[ v, 0.0, 0.0])
            _place(w, 'H', [80.0, 0.0, 0.0], vel=[-v, 0.0, 0.0])
            forces = np.zeros((w.n, 3))
            add_thermal_pressure(w, forces)
            return abs(forces[0, 0])

        assert _force_at_speed(80.0) > _force_at_speed(20.0)

    def test_no_force_single_particle(self, cfg):
        w = World(cfg)
        _place(w, 'H', [0.0, 0.0, 0.0], vel=[100.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)
        np.testing.assert_array_equal(forces, 0)

    def test_skipped_when_thermal_cfg_absent(self, cfg):
        del cfg.thermal   # remove thermal section
        w = World(cfg)
        _place(w, 'H', [0.0,  0.0, 0.0], vel=[ 50.0, 0.0, 0.0])
        _place(w, 'H', [80.0, 0.0, 0.0], vel=[-50.0, 0.0, 0.0])
        forces = np.zeros((w.n, 3))
        add_thermal_pressure(w, forces)   # must not raise
        np.testing.assert_array_equal(forces, 0)
