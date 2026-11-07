# E2 GPU hardware calibration (v3)

This package preserves the **original E2 calibration implementation** from `terrychen0803/gpu-prediction` commit `b28144163e1ff08fc22913f5f05e75bdb561eea0` without modifying its numerical protocol. Its three intended devices are RTX4090, RTX3090 and GB10; no RTX5090 experiments are imported.

```bash
python -m e2_calibration --label rtx4090 --expect-gpu 4090 --smoke --out /data/e2/4090_smoke
python -m e2_calibration --label rtx4090 --expect-gpu 4090 --out /data/e2/4090_full
python -m e2_calibration.summarize /data/e2/4090_full
```

RTX3090: `--label rtx3090 --expect-gpu 3090`. GB10: `--label gb10 --expect-gpu GB10`. These commands run on the named GPU, not the development PC. They create new outputs and do not import historical raw results by themselves. Three existing 100-cell profiles were recorded in the old project; their data must be separately validated before registering in `data/manifests`.

Outputs include `hardware_profile.json`, `curves.csv`, `microbench_samples.csv` and per-run provenance. E2 launch/transfer are measured effective costs, not theoretical peak specs.

Preserved source filenames: `__init__.py`, `__main__.py`, `kernels.cu`, `summarize.py`, `trace_smoke.py`; use the migration manifest for upstream exact file hashes. Raw results are **not** in Git.
