# I2 task -> I3 operators -> S4 subgraph -> I4 -> I5

Status: conceptual mapping, not measured exact/union/event coverage.

| Real task | I3 S1–S3 | S4 approximate graph | Known missing families or gaps |
|---|---|---|---|
| ResNet-50 | Conv/BN/ReLU/Add/Pool/Linear | CBR, Residual | Actual bottleneck shape and projection |
| MobileNetV3-Large | DW, pointwise, BN, Add | Depthwise-Pointwise, SE | Hardswish/Hardsigmoid/Mul training coverage |
| U-Net | Conv, ReLU, Resize, Concat, Pool | Resize-Concat-Conv | Real decoder shape and Memset |
| YOLO26n | Conv, Resize, Concat | Resize-Concat-Conv | SiLU, real neck/head, top-k/copy |
| Swin-T | BMM/GEMM/Linear | Generic eager attention | LayerNorm/GELU/window shift, fused backend |
| Qwen3 prefill/decode | BMM/GEMM | Generic eager attention | RMSNorm/RoPE/gating/KV update and real backend |

Important historical audit: S4 D positions exactly matching S1–S3 controlled signatures = 0 of 28. This is unseen Signature PLUS new Context, not a pure Context effect.

Future task coverage schema: workload_id, model_hash, forward_scope, operator_overload, runtime_signature, source_event_range, reference_quality, isolated_match, graph_match, i4_support, missing_reason, source_P_id, target_GI_id.

Do not publish per-task numeric coverage percentages until original I2 R/P/recorded-work material is independently audited. Kernel fusion/copy and overlapping event union require explicit treatment.
