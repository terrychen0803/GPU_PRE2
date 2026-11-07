"""CPU-only independent detector and N0. Generic L throughout packing/DP/decode."""
import numpy as np
import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self, classes=11):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(29,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.segment=nn.Linear(32,classes)

    def forward(self,x):
        return self.segment(self.encoder(x))


class N0(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(43,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x,span):
        if x.shape[1]!=43 or bool(torch.count_nonzero(x[:,31:])):
            raise ValueError('N0 fixed-zero slots violated')
        return span.log()+self.net(x).squeeze(1)


def normalizer(z,h):
    def fit(v,log):
        x=log(np.asarray(v,dtype=np.float64))
        m=x.mean(0);s=x.std(0)
        # Constant detector hardware must remain exactly zero after transform.
        s=np.where(s<1e-12,1,s)
        return {'mean':m.tolist(),'std':s.tolist()}
    return {'z':fit(z,np.log1p),'h':fit(h,np.log)}


def transform(v,n,hardware=False):
    raw=np.asarray(v,dtype=np.float64)
    x=((np.log(raw) if hardware else np.log1p(raw))-np.asarray(n['mean']))/np.asarray(n['std'])
    t=torch.tensor(x,dtype=torch.float32)
    if not bool(torch.isfinite(t).all()):
        raise ValueError('nonfinite features')
    return t


def detector_x(r,norm):
    z=transform([c['z'] for c in r['candidates']],norm['z'])
    h=transform([r['h_source']]*len(z),norm['h'],True)
    return torch.cat([z,h],1)


def n0_x(z,hs,ht,norm):
    return torch.cat([transform(z,norm['z']),transform(hs,norm['h'],True),
                      transform(ht,norm['h'],True),torch.zeros((len(z),12))],1)


def pack_layout(records,limit):
    n=max(r['n'] for r in records)
    index=torch.zeros((len(records),n,limit),dtype=torch.long)
    mask=torch.zeros_like(index,dtype=torch.bool)
    offsets=[];offset=0
    for row,r in enumerate(records):
        offsets.append(offset)
        expected={(a,b) for b in range(1,r['n']+1) for a in range(max(0,b-limit),b)}
        actual={(c['a'],c['b']) for c in r['candidates']}
        if actual!=expected or len(actual)!=len(r['candidates']):
            raise ValueError('candidate/DP limit mismatch')
        for j,c in enumerate(r['candidates']):
            index[row,c['b']-1,c['b']-c['a']-1]=offset+j
            mask[row,c['b']-1,c['b']-c['a']-1]=True
        for end in range(r['n'],n):
            mask[row,end,0]=True
        offset+=len(r['candidates'])
    return {'idx':index,'mask':mask,'length':torch.tensor([r['n'] for r in records]),
            'offsets':offsets,'limit':limit}


def partition(logits,layout):
    index=layout['idx'];limit=layout['limit']
    weights=torch.logsumexp(logits,1)[index].masked_fill(~layout['mask'],-float('inf'))
    alpha=[logits.new_zeros(index.shape[0])]
    for end in range(1,index.shape[1]+1):
        m=min(limit,end)
        past=torch.stack([alpha[end-k] for k in range(1,m+1)],1)
        alpha.append(torch.logsumexp(past+weights[:,end-1,:m],1))
    return torch.stack(alpha,1).gather(1,layout['length'][:,None]).squeeze(1)


def viterbi(cs,n,logits,limit):
    # The same strict layout validator also checks the decoder's L.
    pack_layout([{'n':n,'candidates':cs}],limit)
    values=logits.detach().cpu().numpy()
    ends=[[] for _ in range(n+1)]
    for i,c in enumerate(cs):
        ends[c['b']].append(i)
    dp=[0.]+[-float('inf')]*n;back=[None]*(n+1)
    for b in range(1,n+1):
        for i in ends[b]:
            c=cs[i];cls=int(np.argmax(values[i]));v=dp[c['a']]+float(values[i,cls])
            if v>dp[b]:
                dp[b]=v;back[b]=(i,cls)
    result=[];b=n
    while b:
        if back[b] is None:
            raise ValueError('unreachable partition')
        i,cls=back[b];c=cs[i]
        result.append({'a':c['a'],'b':c['b'],'c':cls,'candidate_index':i})
        b=c['a']
    return result[::-1]


def structure_loss(model,rs,truth,norm,limit):
    layout=pack_layout(rs,limit)
    logits=model(torch.cat([detector_x(r,norm) for r in rs]))
    scores=[]
    for r,offset in zip(rs,layout['offsets']):
        ix={(c['a'],c['b']):i+offset for i,c in enumerate(r['candidates'])}
        gold=truth[r['id']]
        scores.append(torch.stack([logits[ix[g['a'],g['b']],g['c']] for g in gold]).sum())
    return ((partition(logits,layout)-torch.stack(scores))/layout['length']).mean()


def detect(model,r,norm,limit):
    with torch.no_grad():
        logits=model(detector_x(r,norm))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite detector')
        return {'id':r['id'],'predicted':viterbi(r['candidates'],r['n'],logits,limit),
                'candidate_q':torch.softmax(logits,1).tolist()}
