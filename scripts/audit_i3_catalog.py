#!/usr/bin/env python3
"""CPU-only I3 configuration check; does not access GPUs or raw data."""
import json
from gpu_pre2.i3.registry import catalog_summary
if __name__=="__main__":
    print(json.dumps(catalog_summary(),ensure_ascii=False,indent=2))
