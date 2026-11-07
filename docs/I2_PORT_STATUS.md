# I2: seven-A forward adapter and worker port

- The original seven workload specifications, model-specific frozen asset adapter and \`formal_worker\` timing scope were **extracted** from old \`o4_pilot/o4_research/\` at source commit \`b28144163e1ff08fc22913f5f05e75bdb561eea0\`.
- The minimal \`precision_policy\`, R dispatch instrumentation and own-CUDA-context-reset utilities are included once in \`gpu_pre2.i2.runtime\` instead of importing complete \`vendor/legacy\` O1/O2 trees.
- \`gpu_pre2.i2.formal_protocol.PROTOCOL\` retains GI/P/R, 40 measured iterations × 3 fresh processes. Existing timing and protocol are engineering-only.
- \`gpu_pre2.i2.prepare_assets\` requires existing checkpoint/weights files. It does **not** download or invent model assets.
- \`gpu_pre2.i2.formal_worker\` is a port of one worker process. It is **NOT** a full validated replacement for the old \`run_formal\` orchestrator + Nsight parser + acceptance gate. External \`nsys profile\`, audit/QA, immutable run scheduling and three-node replay still need to be migrated and verified before any new GPU collection.
- The exact historical I2 raw traces and frozen assets are not present in Git; their manifest/SHA and per-model compatibility must be checked separately.

## Minimum migration validation

1. \`python -m unittest discover -s tests -v\` on a Python environment with torch.
2. Check that each adapter loads the exact frozen A assets, native model route/precision, outputs and code hashes.
3. In a new experimental session (not historical outputs), run one plan/smoke GI→P→R on each device using a validated orchestrator.
4. Compare semantic scope, R dispatch, GPU event timing and quality against the fixed original collector. Do not mark port compatible before this.
