"""Standalone source-only inference. Does not import prepare/train/evaluate.

Only source numeric events, declared hardware and frozen models are opened.
The output must be sealed before any evaluator attaches target test masks.
"""
import argparse
import numpy as np
import torch
from .common import RUN, L, TARGETS, read, save, sha
from .features import request, Unsupported
from .model import Detector, N0, n0_x, detect


def predict_one(raw,detector,dnorm,time,tnorm,hardware,baselines):
    try:
        r=request(raw)
    except Unsupported as e:
        return {'id':raw['id'],'status':'unsupported','reason':str(e),
                'events':len(raw['events']),'predicted':[],'candidate_times':{}}
    p=detect(detector,r,dnorm,L)
    p.update(status='closed_set_exploratory',n=r['n'],
             semantic_coverage_status='unverified_unknown_detector_not_calibrated',
             time_coverage_status='candidate_outputs_only_not_GI',candidate_times={})
    for target in TARGETS:
        z=[c['z'] for c in r['candidates']]
        span=torch.tensor([c['z'][1] for c in r['candidates']],dtype=torch.float32)
        x=n0_x(z,[r['h_source']]*len(z),[hardware[target]]*len(z),tnorm)
        with torch.no_grad():
            pred=time(x,span).exp().numpy().astype(float)
        ridge=np.exp(np.log(span.numpy().astype(float))+np.column_stack([np.ones(len(x)),x.numpy()])@np.asarray(baselines['ridge_coefficients']))
        times={'N0':pred.tolist(),'Identity':span.tolist(),
               'Ratio':(span.numpy()*baselines['ratios'][target]).tolist(),'Ridge':ridge.tolist()}
        if not all(np.isfinite(v).all() and (np.asarray(v)>0).all() for v in times.values()):
            raise ValueError('nonfinite or nonpositive timing prediction')
        p['candidate_times'][target]=times
    return p


def load_models(root,seed):
    dp=root/f'detector_s{seed}';tp=root/f'N0_s{seed}'
    d=Detector();d.load_state_dict(torch.load(dp/'model.pt',map_location='cpu',weights_only=True));d.eval()
    t=N0();t.load_state_dict(torch.load(tp/'model.pt',map_location='cpu',weights_only=True));t.eval()
    return d,read(dp/'normalizers.json'),t,read(tp/'normalizers.json'),read(root/'baselines.json')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input',required=True)
    a.add_argument('--hardware',required=True)
    a.add_argument('--models',required=True)
    a.add_argument('--out',required=True)
    args=a.parse_args()
    from pathlib import Path
    root=Path(args.models)
    frozen=read(root/'FROZEN.json')
    for rel,h in frozen['files'].items():
        assert sha(root/rel)==h,rel
    torch.set_num_threads(1)
    hw=read(args.hardware);rs=read(args.input)
    out=[]
    for seed in frozen['seeds']:
        d,dn,t,tn,b=load_models(root,seed)
        out.extend({'seed':seed,**predict_one(r,d,dn,t,tn,hw,b)} for r in rs)
    save(args.out,out)
    save(str(args.out)+'.seal.json',{'prediction_sha256':sha(args.out),'input_sha256':sha(args.input),
         'hardware_sha256':sha(args.hardware),'frozen_sha256':sha(root/'FROZEN.json'),
         'target_test_read':False,'reference_read':False})
    print('sealed',args.out,len(out),flush=True)


if __name__=='__main__':
    main()
