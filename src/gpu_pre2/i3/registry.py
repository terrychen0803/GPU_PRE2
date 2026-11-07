"""Small, dependency-free registry for I3 isolated and context definitions.

Historical catalog status strings were not rewritten after actual GPU collection.
An experiment plan is NOT evidence of a completed capture: see data/manifests.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

def repository_root():
    return Path(__file__).resolve().parents[3]

CATALOG="configs/operators/I3_S1_S3_CATALOG.json"
CONTEXT_A="configs/contexts/S4_A_SCALE0_PLAN.json"
CONTEXT_BC="configs/contexts/S4_BC_SCALE0_PLAN.json"

def _read(path,root=None):
    return json.loads(((root or repository_root())/path).read_text())

def load_operators(root=None):
    content=_read(CATALOG,root)
    configs=content["configurations"]
    if len({c["id"] for c in configs})!=len(configs):
        raise ValueError("duplicate operator config ID")
    for c in configs:
        for key in ("id","family","leakage_group","semantic_sha256","definition"):
            if key not in c:
                raise ValueError(f"missing {key} in I3 operator config")
    return configs

def load_contexts(root=None):
    graphs=(_read(CONTEXT_A,root)["graphs"]+
            _read(CONTEXT_BC,root)["graphs"])
    if len({g["id"] for g in graphs})!=len(graphs):
        raise ValueError("duplicate Context Graph ID")
    for g in graphs:
        if not g.get("leakage_group") or not g.get("nodes"):
            raise ValueError(f"incomplete context graph {g.get('id')}")
        nodes=[n["id"] for n in g["nodes"]]
        if len(set(nodes))!=len(nodes):
            raise ValueError("repeated graph node ID")
    return graphs

def catalog_summary(root=None):
    configs=load_operators(root)
    graphs=load_contexts(root)
    families=defaultdict(int)
    groups=defaultdict(set)
    for c in configs:
        families[c["family"]]+=1
        groups[c["leakage_group"]].add(c.get("split_B"))
    errors={g:sorted(s) for g,s in groups.items() if len(s)>1}
    if errors:
        raise ValueError(f"operator anchor group split collision: {errors}")
    pos=sum(len(g["nodes"]) for g in graphs)
    return {
        "operator_catalog_count":len(configs),
        "families":dict(sorted(families.items())),
        "operator_design_groups":len(groups),
        "context_graph_count":len(graphs),
        "context_position_count":pos,
        "context_work_items_theoretical":2*len(graphs)+pos,
        "context_leakage_groups":sorted({g["leakage_group"] for g in graphs}),
        "warning":"This is a design catalog; actual captured cells require audited manifests."
    }
