"""
Universe Simulator — entry point.

Usage:
    python main.py
    python main.py --settings path/to/settings.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from sim.world  import World
from sim.viewer import Viewer


def _to_ns(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_cfg(path: str = 'settings.yaml') -> SimpleNamespace:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description='Universe Simulator')
    parser.add_argument('--settings', default='settings.yaml', help='Path to settings YAML')
    args = parser.parse_args()

    settings_path = Path(args.settings)
    if not settings_path.exists():
        print(f'Error: settings file not found: {settings_path}', file=sys.stderr)
        sys.exit(1)

    cfg = load_cfg(str(settings_path))

    world  = World(cfg)
    viewer = Viewer(world, cfg)

    print('Universe Simulator starting.')
    print(f'  Injection rate : {cfg.simulation.injection_rate} particles/s')
    print(f'  Max particles  : {cfg.simulation.max_particles}')
    print(f'  Gravity G      : {cfg.physics.gravity_constant}')
    print(f'  Fusion         : {"enabled" if cfg.chemistry.fusion_enabled else "disabled"}')
    print()
    print('Controls:')
    print('  Left-drag        → orbit   |  Right-drag / Scroll → zoom  |  Middle → pan')
    print('  Arrow keys       → orbit   |  Page Up/Down        → zoom')
    print('  Space            → pause   |  +/-                 → speed up/down')
    print('  F                → cinematic camera (auto orbit)')
    print('  P                → toggle astronomy colour palette')
    print('  R                → reset camera  |  Q / Esc → quit')
    print('  Left-click       → inspect particle')
    print()

    viewer.run()


if __name__ == '__main__':
    main()
