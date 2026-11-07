# GPU_PRE2 — Cross-GPU Inference Runtime Prediction

> **Research source:** RTX 4090 P-only profiling trace. **Targets:** RTX 3090 and NVIDIA GB10.
>
> **Status:** clean-repository migration in progress. This repository deliberately does **not** import the complete historical `gpu-prediction` repository.

## Research objective

Given a fixed inference forward pass on RTX 4090, infer the operator boundaries/classes using only permitted numeric P-trace features; estimate target-GPU operator contextual spans via a calibrated cross-device predictor; then estimate the **single forward iteration** GPU event elapsed time (`GI`, `iteration_device_event_us`) on RTX 3090 or GB10.

- Six inference model families: ResNet-50, MobileNetV3-Large, Swin-T, U-Net, YOLO26n, Qwen3-0.6B.
- Seven inference groups: Qwen prefill and one-step decode are distinct.
- GI measures one complete scoped forward, **not** a sum over 40 measured iterations, GPU busy union, or host G0.
- Target test traces, target GI labels, reference-only boundaries/types/shapes must **never** enter source-only predictor inputs.

## Scope

Keep only **E2** hardware calibration, essential **O3** Semi-Markov + N0 method, **I2** task-scoped forward measurement, **I3** isolated S1–S3 and contextual S4, and selectively the frozen **I4** exploratory model/pipeline baseline. The legacy O1/O2 and unrelated experiment history are *not* part of the new research pipeline.

### I4 learning design

1. Isolated pretraining: S1–S3 single-operator P traces, trained with structured Semi-Markov loss.
2. Mixed structured training: initialize from isolated model; draw grouped samples from **S1–S3 and S4 D/I**, preserving both isolated classification and multi-operator boundary supervision.
3. Evaluate against isolated-only, context-only, and context-only fine-tuning on separate signature/graph groups.
4. Train N0 CTX cross-GPU time model with source P-only features and train-only target P-CTX labels; separately evaluate isolated and mixed-context variants.
5. Use independent held-out contexts and task-derived real-forward coverage; **never split iterations/repeats of the same configuration across evaluation groups**.

**Data status:** raw Nsight reports, SQLite captures, frozen input/weights and most learned checkpoints live on experiment hosts, not in the historical Git repository. Their migration is not assumed until manifest/SHA validation. Existing trace quality is engineering/development-only, `strict_reference_verified=false`.

See `docs/` for migration status and the new data contract. This initial commit is only the repository bootstrap.
