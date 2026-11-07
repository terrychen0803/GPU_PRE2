"""Frozen configurations shipped with this repository, never found in raw folders."""
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[3] / 'configs'

def configuration(relative):
    path = (CONFIGS/relative).resolve()
    if CONFIGS.resolve() not in path.parents or not path.is_file():
        raise ValueError('Unknown shipped configuration: '+relative)
    return path
