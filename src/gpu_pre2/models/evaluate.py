"""Segment-level metrics with all gold operations kept in the denominator."""
from __future__ import annotations
from .semimarkov import viterbi

def exact_boundary_f1(gold: list[dict], predicted: list[dict]) -> dict:
    g={(x["start"],x["end"]) for x in gold}
    p={(x["start"],x["end"]) for x in predicted}
    true=len(g & p)
    prec=true/len(p) if p else 0.
    rec=true/len(g) if g else 0.
    return {"precision":prec,"recall":rec,
            "f1":2*prec*rec/(prec+rec) if prec+rec else 0.}

def typed_f1(gold: list[dict], predicted: list[dict]) -> dict:
    g={(x["start"],x["end"],x["class_id"]) for x in gold}
    p={(x["start"],x["end"],x["class_id"]) for x in predicted}
    true=len(g & p)
    prec=true/len(p) if p else 0.
    rec=true/len(g) if g else 0.
    return {"precision":prec,"recall":rec,
            "f1":2*prec*rec/(prec+rec) if prec+rec else 0.}
