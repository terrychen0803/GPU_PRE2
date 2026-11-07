# Cross-GPU data contract

Source: RTX4090 P-only GPU numeric events. Targets: RTX3090 and GB10 separately. The six inference model families are ResNet-50, MobileNetV3, Swin-T, U-Net, YOLO26n, Qwen3-0.6B (prefill and decode as distinct forward groups).

GI (iteration_device_event_us) is **one complete forward execution measured by no-profiler CUDA events**, microseconds. It includes waiting/gaps inside the interval, not a simple sum of kernel durations. Forty repeated measurements are forty separate GI labels, not a 40-iteration sum.

P CTX (contextual_trace_span_us) is a profiled operator/segment span (first recorded event start to last end). R gives evaluated ownership, true class, shape and boundary. R is training/evaluator-only. Target P CTX may supervise N0 in TRAIN and target GI may supervise I5 in TRAIN; target TEST traces/labels never enter source-only predictor input. Old G0 host latency / GE isolated latency are not GI.

I3 S1–S3 isolated training teaches operator identity and within-operator grouping. I3 S4 D/I teaches boundaries and context; S provides isolated-position controls. Same Signature/Graph component across devices, modes, view and repeat must stay in the same split. Historical context_scale_0 is only one context group.

All current captures remain engineering/development evidence with strict_reference_verified=false. Preserve original timestamps, copies/memsets, overlaps, API/correlation failures, multistream uncertainty; do not fabricate missing events or geometry.
