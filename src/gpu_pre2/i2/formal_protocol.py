"""Frozen seven-A 40x3 engineering protocol; do not relabel as strict truth."""
from __future__ import annotations
import statistics

PROTOCOL = {
    "protocol_id": "gpu_pre2:o4_seven_A_40x3_engineering_v2_20261106",
    "status": "new_seven_A_development_pilot_not_original_frozen_v2",
    "modes": ["GI", "P", "R"],
    "independent_process_repeats": 3,
    "measured_iterations": 40,
    "transition_iterations": 5,
    "warmup_min": 50,
    "warmup_max": 500,
    "warmup_window": 20,
    "stable_windows": 3,
    "stability_threshold": 0.05,
    "repeat_spread_threshold": 0.05,
    "schedule_seed": 61106,
    "trace_backend": "cuda-sw",
    "dtype": "float16",
    "eager": True,
    "autocast": False,
    "compile": False,
    "cuda_graph": False,
    "single_stream_required": True,
    "target_kind": "iteration_device_event_us",
    "correctness": "adapter_v1_frozen_inputs_and_native_route_parity",
    "preparation": "adapter.prepare outside every timed iteration; explicit NVTX scope in P/R",
    "timing": "initialized CUDA events around one native forward, end.synchronize",
    "reference": "ATen leaf dispatch only in R; no strict certification",
    "nvtx_warning_policy": "provisional_engineering_only_preserve_diagnostics",
    "strict_verified": 0,
    "finalization": "release adapter/events; synchronize; sleep1s; own-context reset; ProfilerStop",
    "engineering_acceptance": "o4_engineering_acceptance_v1_20261106",
    "training_eligible": False,
    "formal_training_eligible": False,
}
def validate_protocol(value):
    if value != PROTOCOL:
        raise ValueError("Seven-A engineering protocol mismatch")
    return value

def stable(values, policy=PROTOCOL):
    w=policy["warmup_window"];n=policy["stable_windows"]
    if len(values)<max(policy["warmup_min"],w*n):
        return False
    med=[statistics.median(
        values[len(values)-w*(n-i):len(values)-w*(n-i-1)]
        if i<n-1 else values[-w:]) for i in range(n)]
    return all(a>0 and abs(b/a-1)<=policy["stability_threshold"]
               for a,b in zip(med,med[1:]))

def warmup_stable(values):
    return stable(values,PROTOCOL)

def process_spread(values):
    if len(values)!=PROTOCOL["independent_process_repeats"]:
        return None
    mid=statistics.median(values)
    if mid<=0:
        raise ValueError("Nonpositive process median")
    return (max(values)-min(values))/mid

def expected_scopes():
    return {f"o4.iter/{stage}/{i:06d}"
            for stage,n in (("transition",PROTOCOL["transition_iterations"]),
                            ("measure",PROTOCOL["measured_iterations"]))
            for i in range(n)}
