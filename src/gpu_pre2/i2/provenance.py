"""Independent environment snapshot without importing historical O1/O2."""
from __future__ import annotations
import platform
import subprocess
import sys
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def source_hashes():
    files = []
    for name in ('src', 'configs', 'e2_calibration'):
        files.extend(p for p in (ROOT/name).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.json', '.cu'})
    files.append(ROOT/'pyproject.toml')
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}

def verify_release():
    path = ROOT/'data/manifests/CODE_RELEASE.json'
    frozen = json.loads(path.read_text())
    if source_hashes() != frozen['files']:
        raise ValueError('GPU_PRE2 source differs from its sealed code release')
    return {'release_id':frozen['release_id'], 'source_verified':True,
            'strict_reference_verified':False}

def _cmd(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=10)
        return r.stdout.strip() if r.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None

def snapshot():
    return {"python":sys.version,"platform":platform.platform(),
            "git_commit":_cmd(["git","rev-parse","HEAD"]),
            "nvidia_smi":_cmd(["nvidia-smi","--query-gpu=name,uuid,driver_version",
                               "--format=csv,noheader"]),
            "nsys_version":_cmd(["nsys","--version"]),
            "status":"unverified_port_require_frozen_source_hash_check"}
