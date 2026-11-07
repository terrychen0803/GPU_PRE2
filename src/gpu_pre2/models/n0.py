"""O3-inspired N0 log-ratio time head: source P CTX -> target P CTX.

This defines the same 43->64->32->1 architecture, but is not an
assertion that archived O3/I4 weights can be loaded without adapter checks.
"""
from __future__ import annotations
import torch
from torch import nn

class N0TimeRatio(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(43,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, source_features, source_hw, target_hw,
                source_ctx_us):
        if source_features.shape[-1] != 27:
            raise ValueError("27 source-only features required")
        if source_hw.shape[-1] != 2 or target_hw.shape[-1] != 2:
            raise ValueError("source/target hardware descriptors must be 2-D")
        if source_ctx_us.ndim != 1:
            raise ValueError("source_ctx_us must be [batch]")
        if not torch.all(source_ctx_us > 0):
            raise ValueError("source_ctx_us must be positive")
        batch = source_features.shape[0]
        if (source_hw.shape[0] != batch or target_hw.shape[0] != batch
                or len(source_ctx_us) != batch):
            raise ValueError("inconsistent batch")
        # Do not feed true class, true shape, q or target trace features.
        features = torch.cat((
            source_features,source_hw,target_hw,
            source_features.new_zeros((batch,3)),
            source_features.new_zeros((batch,9))
        ),dim=-1)
        return source_ctx_us * torch.exp(self.network(features).squeeze(-1))

def log_huber_loss(pred_us, truth_us, delta=0.2):
    if torch.any(pred_us<=0) or torch.any(truth_us<=0):
        raise ValueError("times must be positive")
    err = pred_us.log()-truth_us.log()
    abs_err=err.abs()
    return torch.where(abs_err<=delta,0.5*err.square(),
                       delta*(abs_err-0.5*delta)).mean()
