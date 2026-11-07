"""Run-local I4 artifacts. No credentials, CUDA, network or collector mutations."""
import csv
import hashlib
import json
from pathlib import Path

from gpu_pre2.data.storage import data_root
RUN = data_root() / 'checkpoints/i4_p0_p2'
CLASSES = ('adaptive_average_pool', 'add', 'batchnorm_eval', 'bilinear_resize',
           'bmm', 'concat', 'conv', 'depthwise', 'gemm', 'maxpool', 'relu')
TARGETS = ('GB10', 'RTX3090')
SEEDS = (17, 42, 73)
SPLITS = ('train', 'validation', 'test_exploratory')
L = 8


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def table(p, data):
    data = list(data)
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', newline='') as f:
        if data:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def oid(cid):
    return hashlib.sha256(('o4-i4-source:' + cid).encode()).hexdigest()[:24]
