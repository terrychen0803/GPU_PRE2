"""Explicit retained-stage storage. No globbing or third-party loader discovery."""
from __future__ import annotations
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGES = frozenset({'E2', 'I2', 'I3', 'I4'})


def data_root():
    return Path(os.environ.get('GPU_PRE2_DATA_ROOT', REPO.parent / 'research_data')).expanduser().resolve()


def inside_data(path):
    root = data_root()
    target = Path(path).expanduser().resolve()
    if target == root or root not in target.parents:
        raise ValueError(f'Expected a registered path inside GPU_PRE2_DATA_ROOT: {target}')
    return target


def dataset(dataset_id, *, stage):
    if stage not in STAGES:
        raise ValueError('Only E2/I2/I3/I4 are retained dataset stages')
    registry = json.loads((REPO / 'data/manifests/DATASETS.json').read_text())
    found = [r for r in registry['datasets'] if r['id'] == dataset_id]
    if len(found) != 1 or found[0]['stage'] != stage:
        raise ValueError(f'No explicit {stage} registration for {dataset_id}')
    entry = found[0]
    if entry['status'] != 'sha256_verified':
        raise ValueError('Dataset migration is not verified')
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Dataset registry requires contained relative paths')
    root = inside_data(data_root() / relative)
    if not root.exists():
        raise FileNotFoundError(root)
    return root
