# GPU_PRE2 — RTX4090 → RTX3090 / GB10 Inference Runtime Prediction

Research objective: **predict the single-forward CUDA-event GPU elapsed time (GI)** on RTX3090 or GB10, using a **P-only profiling trace from RTX4090**. Six inference architectures: ResNet-50, MobileNetV3-Large, Swin-T, U-Net, YOLO26n, Qwen3-0.6B (separate prefill and one-step decode = seven workload groups).

The system identifies operator boundaries/classes in source P events; N0 estimates target GPU contextual operator spans; future I5 combines source execution structure and these estimates into target GI. An observed label is **one iteration of one complete scoped forward**, not the sum of 40 iterations, a CPU/host G0 interval, or total GPU kernel busy time.

## Project status

**Research reconstruction / development branch**. This is a clean successor to the old gpu-prediction repository; it is not a copy of old branches or the O1/O2 research history. It contains E2 calibration, selected I2 forward adapters and worker, I3 catalogs/graph definitions, I4 experimental Semi-Markov/N0 model and a newly implemented grouped mixed-training trainer. The complete former I2/I3 capture orchestrators, Nsight R-to-P replay/acceptance logic, original I4 checkpoints and large raw captures **are not yet migrated/validated**. No new GPU experiments or actual new I4 fits were run in this repository.

**Existing trace quality remains development-only** (strict_reference_verified=false). New code is NOT numerically guaranteed equivalent to old frozen O3/I4 checkpoints.

## Modules

| Component | Location | Status |
|---|---|---|
| E2 hardware microbenchmarks | e2_calibration/ | Exact executable source extracted from fixed old commit; 3-GPU profiles stored externally |
| I2 7-A workloads | src/gpu_pre2/i2/ | Seven model adapters, asset prep, single-process GI/P/R worker extracted; capture coordinator/parser still pending |
| I3 S1–S3 | configs/operators/; src/gpu_pre2/i3/ | Frozen 174-entry **design pool**; 72 selected configurations actually measured per node, subject to per-node quality |
| I3 S4 | configs/contexts/ | Six scale0 D/I/S graph definitions and engineering summaries; one context group |
| I4 recognition | src/gpu_pre2/models/semimarkov.py | Differentiable L8 Semi-Markov score, partition, structured loss, Viterbi |
| I4 training | src/gpu_pre2/models/train_mixed.py | Isolated-only, context-only, pretrain→context, pretrain→mixed; group-balanced, train-only normalizer |
| I4 N0 | src/gpu_pre2/models/n0.py | 43→64→32→1 time-ratio model; historical weights NOT imported |
| I4 results | results/i4/ | Historical P0–P2 exploratory summary, not reproduced |
| I5 | docs/DATA_CONTRACT.md | GI target contract only; ITER predictor not implemented |

The old O3 Semi-Markov/N0 principles are absorbed into the I4 module, not carried forward as a separate historical experiment tree. Old O1/O2 raw datasets and vendor/legacy copies are intentionally excluded.

## I4 mixed structured learning

1. **Pretrain on isolated S1–S3**: one gold operator per source P sequence; learn local operator class, event structure and within-operator grouping.
2. **Continue with mixed S1–S3 + S4 D/I**: a shared Semi-Markov recognizer learns *not merging neighboring operators*, while retaining isolated supervision. Group-balanced sampling prevents the 40 iterations × 3 process repeats from masquerading as independent training examples.
3. **Evaluate independently** on different connected signature/context graph groups; compare 4 learning strategies. R true boundaries/classes are training/evaluator-only, never predictor features.
4. **N0** is a separate source P-feature → target P CTX regression head; CTX and GI have different timing semantics.

Existing S4 Scale0 constitutes only **one connected context group**, so mixed learning can be a development exercise, **not yet an independently validated context generalization result**. Missing Hardswish/Hardsigmoid/Mul/Softmax and Memset event geometry need appropriate feature/taxonomy/data treatment. The new candidate aggregator is mean-pooling and is **not checkpoint-identical** to historical I4 E1 scoring.

## Quickstart (CPU development)

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python scripts/audit_i3_catalog.py

Train only after a verified local dataset converter has exported the JSONL contract:

    python scripts/train_i4.py --train /data/i4/train.jsonl --validation /data/i4/val.jsonl --mode pretrain_mixed --seed 17 --out /data/i4/fits/mixed_17

No GPU data collection or target-trace access occurs in that CLI. See docs/I4_MIXED_TRAINING.md.

## Data storage

- Git: executable source, experiment definitions, small provenance summaries, reports and tests.
- **Outside Git**: NSys .nsys-rep/SQLite raw, frozen weights/tensors, E2 actual profile runs, old I4 trained checkpoints, new fit output. These require validated SHA256 mapping before being called migrated.
- data/manifests/EXTERNAL_ASSETS_REQUIRED.json tracks absent artifacts. No imaginary completed transfers.

Read docs/MIGRATION_STATUS.md, docs/TASK_COVERAGE.md, docs/DATA_CONTRACT.md, docs/I2_PORT_STATUS.md and docs/I3_PORT_STATUS.md first.
