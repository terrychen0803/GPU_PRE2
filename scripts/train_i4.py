#!/usr/bin/env python3
"""CPU/GPU development training entry point; NO GPU data collection."""
from __future__ import annotations
import argparse
from gpu_pre2.data.contracts import load_split_files
from gpu_pre2.models.train_mixed import fit_i4,save_fit

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True,help="train-only labeled JSONL")
    p.add_argument("--validation",required=True,help="different-group validation JSONL")
    p.add_argument("--out",required=True,help="new non-existing output directory")
    p.add_argument("--mode",choices=("isolated_only","context_only",
        "pretrain_context","pretrain_mixed"),default="pretrain_mixed")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--pretrain-updates",type=int,default=720)
    p.add_argument("--second-updates",type=int,default=720)
    p.add_argument("--iso-weight",type=float,default=0.5)
    p.add_argument("--device",default="cpu")
    p.add_argument("--log-every",type=int,default=60)
    a=p.parse_args()
    train,val=load_split_files(a.train,a.validation)
    model,norm,report=fit_i4(
        train,val,mode=a.mode,seed=a.seed,
        pretrain_updates=a.pretrain_updates,
        second_updates=a.second_updates,iso_weight=a.iso_weight,
        device=a.device,log_every=a.log_every)
    save_fit(a.out,model,norm,report)
    print({"mode":a.mode,"validation":report["validation"],
           "output":a.out})

if __name__=="__main__":
    main()
