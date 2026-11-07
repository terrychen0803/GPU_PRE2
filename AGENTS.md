# GPU_PRE2 retained research stages

RTX4090 P-only → RTX3090/GB10 operator CTX and single-forward GI. Six inference models, seven workload groups. Report in Traditional Chinese.

- All runtime imports must resolve to this package or declared dependencies. Never add old O1/O2/O3/O4 paths to sys.path, PYTHONPATH, plugin registries or entry points.
- Load E2/I2/I3/I4 through explicit dataset IDs and the retained dataset registry. No recursive discovery of historical datasets or fallback to removed directories.
- Preserve GI/P CTX/R distinctions, numeric event bytes, unknown/unsupported records, group splits and original quality. Migration never upgrades reference qualification.
- R/shape/class are supervision/evaluator only. Target test traces and times never reach predictor features.
- Keep raw captures, tensors and checkpoints in GPU_PRE2_DATA_ROOT outside Git. Use relative paths and SHA256 manifests.
- Never copy a whole source tree per run; freeze source hashes. No GPU runs or refits are implied by offline migration.
- Run unit tests plus relevant saved-trace and frozen-model regressions for parser/model changes. Live collector ports require new smoke qualification before claiming numerical capture equivalence.
- Do not disturb K3s, Pods, Netdata, RustDesk, drivers or other services.
