"""Independent environment snapshot without importing historical O1/O2."""
from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path

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
