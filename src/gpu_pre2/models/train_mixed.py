"""Group-balanced I4 training: isolated pretrain -> context/mixed structured fit.

This trainer is experimental and does not silently reuse historical I4 weights.
No R/reference or target-GPU data reaches model.forward; gold is used only for
the supervised loss and evaluator.
"""
from __future__ import annotations
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
import torch
from .semimarkov import SemiMarkovRecognizer, structured_loss, viterbi
from .evaluate import typed_f1, exact_boundary_f1
from gpu_pre2.data.contracts import audit_splits, predictor_view

class TrainOnlyNormalizer:
    def __init__(self, mean, std):
        self.mean = [float(x) for x in mean]
        self.std = [float(x) for x in std]

    @classmethod
    def fit(cls, rows):
        if not rows or any(r["split"] != "train" for r in rows):
            raise ValueError("normalizer must use training rows only")
        x = torch.tensor([e for row in rows
                          for e in row["source_event_features"]],
                         dtype=torch.float64)
        mean=x.mean(dim=0)
        std=x.std(dim=0,unbiased=False).clamp_min(1e-6)
        return cls(mean.tolist(),std.tolist())

    def inputs(self, row, device="cpu"):
        v = predictor_view(row)
        e = torch.tensor(v["source_event_features"],
                         dtype=torch.float32,device=device)
        hw = torch.tensor(v["source_hw"],dtype=torch.float32,device=device)
        m = torch.tensor(self.mean,dtype=torch.float32,device=device)
        s = torch.tensor(self.std,dtype=torch.float32,device=device)
        return (e-m)/s, hw

    def to_json(self):
        return {"mean":self.mean,"std":self.std,"fit":"train_only_event_weighted"}

def _group_index(rows, kinds):
    grouped=defaultdict(list)
    for r in rows:
        if r["kind"] in kinds:
            grouped[r["group_id"]].append(r)
    if not grouped:
        raise ValueError(f"no records of kinds {sorted(kinds)}")
    return dict(grouped)

def _sample(index, rng):
    group=rng.choice(sorted(index))
    return rng.choice(index[group])

def _loss(model, normalizer, row, device):
    events, hw=normalizer.inputs(row,device)
    scores=model(events,hw)
    return structured_loss(scores,row["gold_segments"])

@torch.no_grad()
def evaluate(model, normalizer, rows, device="cpu"):
    model.eval()
    bt=bp=gt=gp=0
    per_kind=defaultdict(lambda:{"gold":0,"pred":0,"typed":0,"boundary":0})
    for row in rows:
        e,hw=normalizer.inputs(row,device)
        predicted=viterbi(model(e,hw))
        gold=row["gold_segments"]
        g={(x["start"],x["end"],x["class_id"]) for x in gold}
        p={(x["start"],x["end"],x["class_id"]) for x in predicted}
        gb={(x["start"],x["end"]) for x in gold}
        pb={(x["start"],x["end"]) for x in predicted}
        t=len(g & p); b=len(gb & pb)
        gt+=len(g);gp+=len(p);bt+=t;bp+=b
        k=per_kind[row["kind"]]
        k["gold"]+=len(g);k["pred"]+=len(p)
        k["typed"]+=t;k["boundary"]+=b
    def f1(hits,ng,np):
        return (2*hits/(ng+np)) if ng+np else 0.
    return {"typed_f1":f1(bt,gt,gp),"boundary_f1":f1(bp,gt,gp),
            "gold":gt,"predicted":gp,
            "by_kind":{k:{**v,"typed_f1":f1(v["typed"],v["gold"],v["pred"]),
                           "boundary_f1":f1(v["boundary"],v["gold"],v["pred"])}
                       for k,v in per_kind.items()}}

def _fit_stage(model, optimizer, normalizer, train, validation, mode,
               updates, seed, device, iso_weight=0.5, log_every=60):
    if updates <= 0: return [], copy.deepcopy(model.state_dict())
    rng=random.Random(seed)
    iso=_group_index(train,{"isolated"}) if mode in {"isolated","mixed"} else {}
    ctx=_group_index(train,{"context_d","context_i"}) if mode in {"context","mixed"} else {}
    history=[]
    best=None
    best_score=float("-inf")
    for step in range(1,updates+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if mode=="isolated":
            loss=_loss(model,normalizer,_sample(iso,rng),device)
        elif mode=="context":
            loss=_loss(model,normalizer,_sample(ctx,rng),device)
        elif mode=="mixed":
            # Equal-domain gradient contributions, independent of repeat count.
            # Split is checked *before* any context sample is admitted.
            loss=(iso_weight*_loss(model,normalizer,_sample(iso,rng),device)
                  +(1-iso_weight)*_loss(model,normalizer,_sample(ctx,rng),device))
        else:
            raise ValueError(f"invalid mode {mode}")
        if not torch.isfinite(loss):
            raise ValueError("non-finite structured loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%log_every==0 or step==updates:
            metrics=evaluate(model,normalizer,validation,device)
            record={"update":step,"mode":mode,"loss":float(loss.detach()),
                    "validation":metrics}
            history.append(record)
            # Tie => earlier checkpoint is retained.
            if metrics["typed_f1"]>best_score:
                best_score=metrics["typed_f1"]
                best=copy.deepcopy(model.state_dict())
    if best is None:
        best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return history,best

def fit_i4(train,validation, *, mode="pretrain_mixed", seed=17,
           pretrain_updates=720, second_updates=720,
           iso_weight=0.5, device="cpu", log_every=60):
    """Return fitted experimental model, train-only normalizer, diagnostics.

    Evaluation/confirmation rows are NOT accepted here; caller may load and
    score them only after freezing and versioning the model.
    """
    if mode not in {"isolated_only","context_only","pretrain_context",
                    "pretrain_mixed"}:
        raise ValueError("unknown I4 training mode")
    audit_splits(train+validation)
    if not train or not validation:
        raise ValueError("nonempty grouped train and validation required")
    if any(r["split"]!="train" for r in train):
        raise ValueError("test/confirmation data passed into training")
    if any(r["split"]!="validation" for r in validation):
        raise ValueError("test/confirmation data passed into selection")
    if not 0 < iso_weight < 1:
        raise ValueError("iso_weight must be strictly between zero and one")
    # Numeric (and optimizer) RNG fixed. No target or reference data is read.
    random.seed(seed);torch.manual_seed(seed)
    norm=TrainOnlyNormalizer.fit(train)
    model=SemiMarkovRecognizer().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    history=[]
    if mode=="isolated_only":
        h,_=_fit_stage(model,opt,norm,train,validation,"isolated",
                       pretrain_updates,seed,device,log_every=log_every)
        history+=h
    elif mode=="context_only":
        h,_=_fit_stage(model,opt,norm,train,validation,"context",
                       second_updates,seed,device,log_every=log_every)
        history+=h
    else:
        h,_=_fit_stage(model,opt,norm,train,validation,"isolated",
                       pretrain_updates,seed,device,log_every=log_every)
        history+=h
        # Reset optimizer between stages for interpretable stage comparison.
        opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
        stage="context" if mode=="pretrain_context" else "mixed"
        h,_=_fit_stage(model,opt,norm,train,validation,stage,
                       second_updates,seed+1000,device,iso_weight,log_every)
        history+=h
    final=evaluate(model,norm,validation,device)
    metadata={"mode":mode,"seed":seed,"pretrain_updates":pretrain_updates,
              "second_updates":second_updates,"iso_weight":iso_weight,
              "max_segment_length":8,"source_gpu":"RTX4090",
              "feature_dim":27,"num_classes":11,
              "train_groups":len(set(r["group_id"] for r in train)),
              "validation_groups":len(set(r["group_id"] for r in validation)),
              "validation":final,"history":history,
              "checkpoint_compatible_with_historical_i4":False}
    return model,norm,metadata

def save_fit(directory,model,norm,metadata):
    p=Path(directory)
    if p.exists():
        raise FileExistsError(f"refuse to overwrite model run: {p}")
    p.mkdir(parents=True)
    torch.save({"state_dict":model.cpu().state_dict(),
                "architecture":"semimarkov_e1_meanpool_v1",
                "max_segment_length":8,"num_classes":11},
               p/"recognizer.pt")
    (p/"normalizer.json").write_text(
        json.dumps(norm.to_json(),indent=2),encoding="utf-8")
    (p/"report.json").write_text(
        json.dumps(metadata,indent=2),encoding="utf-8")
