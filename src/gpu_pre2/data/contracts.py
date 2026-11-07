"""I4 development JSONL contract.

This is a NEW adapter contract. It must not be mislabeled as the exact historical
I4 data/predictor_input_v2 schema. Gold fields are restricted to train/evaluator.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

KINDS = {"isolated", "context_d", "context_i"}
SPLITS = {"train", "validation", "test", "confirmation"}
PREDICTOR_FIELDS = ("source_event_features", "source_hw")
FORBIDDEN_PREDICTOR_FIELDS = {
    "gold_segments", "true_class", "true_shape", "r_boundary",
    "r_reference", "target_trace", "target_gi_us", "target_ctx_us",
    "kernel_names", "operator_names", "config_id", "group_id", "sample_id",
}
FEATURE_DIM = 27
HW_DIM = 2

def _finite_vector(values, n, name):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"{name}: expected {n} numeric values")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool)
           or not math.isfinite(float(x)) for x in values):
        raise ValueError(f"{name}: non-finite/non-numeric value")

def validate_row(row: dict, *, classes: int = 11, max_len: int = 8) -> None:
    if row.get("source_gpu") != "RTX4090":
        raise ValueError("I4 main predictor source must be RTX4090")
    if row.get("kind") not in KINDS:
        raise ValueError("kind must be isolated/context_d/context_i")
    if row.get("split") not in SPLITS:
        raise ValueError("unrecognized split")
    if not isinstance(row.get("group_id"), str) or not row["group_id"]:
        raise ValueError("group_id is mandatory; repeats are not groups")
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
        raise ValueError("sample_id mandatory")
    if row.get("quality", {}).get("usable_for_segmentation") is not True:
        raise ValueError("missing explicit segmentation-quality acceptance")
    events = row.get("source_event_features")
    if not isinstance(events, list) or len(events) == 0:
        raise ValueError("missing ordered source events")
    for i, event in enumerate(events):
        _finite_vector(event, FEATURE_DIM, f"event[{i}]")
    _finite_vector(row.get("source_hw"), HW_DIM, "source_hw")
    gold = row.get("gold_segments")
    if not isinstance(gold, list) or not gold:
        raise ValueError("gold_segments required for supervised dataset")
    pos = 0
    for segment in gold:
        if not isinstance(segment, dict):
            raise ValueError("invalid gold segment")
        a, b, label = (segment.get("start"), segment.get("end"),
                       segment.get("class_id"))
        if not all(isinstance(z, int) and not isinstance(z, bool)
                   for z in (a, b, label)):
            raise ValueError("segment must use integer start/end/class_id")
        if a != pos or b <= a or b > len(events):
            raise ValueError("gold must cover every recorded event exactly once")
        if b-a > max_len:
            raise ValueError(f"gold segment longer than L={max_len}")
        if not 0 <= label < classes:
            raise ValueError("unsupported/unlabeled classes cannot be invented")
        pos = b
    if pos != len(events):
        raise ValueError("gold must cover all recorded events")
    if row["kind"] == "isolated" and len(gold) != 1:
        raise ValueError("isolated must have exactly one true operator")

def predictor_view(row: dict) -> dict:
    """The *only* content passed to inference; opaque ID is outside model."""
    view = {k: row[k] for k in PREDICTOR_FIELDS}
    validate_predictor_view(view)
    return view

def validate_predictor_view(view: dict) -> None:
    if set(view) != set(PREDICTOR_FIELDS):
        extra = sorted(set(view) - set(PREDICTOR_FIELDS))
        raise ValueError(f"illegal predictor fields: {extra}")
    for i, event in enumerate(view["source_event_features"]):
        _finite_vector(event, FEATURE_DIM, f"predictor event[{i}]")
    _finite_vector(view["source_hw"], HW_DIM, "predictor source_hw")

def load_jsonl(path, *, classes=11, max_len=8) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_row(row, classes=classes, max_len=max_len)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows

def audit_splits(rows: list[dict]) -> dict:
    by_group = defaultdict(set)
    ids = set()
    for row in rows:
        key = row["sample_id"]
        if key in ids:
            raise ValueError(f"duplicate sample_id: {key}")
        ids.add(key)
        by_group[row["group_id"]].add(row["split"])
    collisions = {g: sorted(v) for g, v in by_group.items() if len(v) > 1}
    if collisions:
        raise ValueError(f"group leakage across splits: {collisions}")
    return {"samples":len(rows), "groups":len(by_group),
            "splits":{s:sum(s in v for v in by_group.values()) for s in SPLITS}}

def load_split_files(train_file, validation_file, *, classes=11, max_len=8):
    train = load_jsonl(train_file, classes=classes, max_len=max_len)
    val = load_jsonl(validation_file, classes=classes, max_len=max_len)
    audit_splits(train+val)
    if any(row["split"] != "train" for row in train):
        raise ValueError("train file contains non-training samples")
    if any(row["split"] != "validation" for row in val):
        raise ValueError("validation file contains non-validation samples")
    return train, val
