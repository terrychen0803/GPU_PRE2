"""Differentiable Semi-Markov recognizer for isolated and multi-op P traces.

Candidate scoring uses source-only numeric event features. Historical O3/I4
weights are NOT assumed compatible: the old exact E1 candidate aggregator must
be ported and checked before attempting checkpoint replay.
"""
from __future__ import annotations
import torch
from torch import nn

class SemiMarkovRecognizer(nn.Module):
    def __init__(self, feature_dim=27, hardware_dim=2, num_classes=11,
                 max_segment_length=8):
        super().__init__()
        if max_segment_length < 1:
            raise ValueError("max_segment_length must be positive")
        self.max_segment_length = max_segment_length
        self.num_classes = num_classes
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim+hardware_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, events: torch.Tensor, source_hw: torch.Tensor):
        """Return scores[start, length-1, class]; invalid lengths=-inf."""
        if events.ndim != 2 or events.shape[0] < 1:
            raise ValueError("events must be non-empty [N,27]")
        if source_hw.shape != (2,):
            raise ValueError("source_hw must have shape [2]")
        n, lmax = len(events), self.max_segment_length
        feats = []
        slots = []
        for a in range(n):
            for length in range(1, min(lmax, n-a)+1):
                feature = events[a:a+length].mean(dim=0)
                feats.append(torch.cat((feature, source_hw), dim=0))
                slots.append((a, length-1))
        vals = self.scorer(torch.stack(feats, dim=0))
        # Use differentiable index_copy to preserve autograd through scores.
        index = torch.tensor([a*lmax+l for a,l in slots],
                             dtype=torch.long, device=vals.device)
        flat = vals.new_full((n*lmax, self.num_classes), float("-inf"))
        flat = flat.index_copy(0, index, vals)
        return flat.reshape(n, lmax, self.num_classes)

def log_partition(scores: torch.Tensor) -> torch.Tensor:
    """Exact log-sum-exp DP over all valid typed segmentations."""
    if scores.ndim != 3:
        raise ValueError("scores must be [N,L,C]")
    n, lmax, _ = scores.shape
    if n < 1:
        raise ValueError("empty sequence")
    z = [scores.new_zeros(())]
    for end in range(1, n+1):
        parts = []
        for length in range(1, min(end, lmax)+1):
            a = end-length
            parts.append(z[a] + torch.logsumexp(scores[a,length-1,:],dim=0))
        z.append(torch.logsumexp(torch.stack(parts), dim=0))
    return z[-1]

def gold_score(scores: torch.Tensor, segments: list[dict]) -> torch.Tensor:
    n, lmax, classes = scores.shape
    pos = 0
    parts = []
    for segment in segments:
        a, end, label = (int(segment["start"]), int(segment["end"]),
                         int(segment["class_id"]))
        if a != pos or end <= a or end > n or end-a > lmax:
            raise ValueError("invalid/incomplete gold boundaries")
        if not 0 <= label < classes:
            raise ValueError("gold class out of range")
        parts.append(scores[a,end-a-1,label])
        pos = end
    if pos != n or not parts:
        raise ValueError("gold segments do not cover all events")
    return torch.stack(parts).sum()

def structured_loss(scores: torch.Tensor, gold: list[dict]) -> torch.Tensor:
    return (log_partition(scores) - gold_score(scores, gold)) / scores.shape[0]

@torch.no_grad()
def viterbi(scores: torch.Tensor) -> list[dict]:
    n, lmax, classes = scores.shape
    if n < 1:
        raise ValueError("empty sequence")
    # Viterbi is inference-only; source numeric features only.
    dp = [0.0] + [float("-inf")] * n
    back = [None] * (n+1)
    for end in range(1, n+1):
        for length in range(1, min(lmax, end)+1):
            a = end-length
            if dp[a] == float("-inf"):
                continue
            for cls in range(classes):
                score = dp[a] + float(scores[a,length-1,cls])
                if score > dp[end]:
                    dp[end] = score
                    back[end] = (a,cls)
    if back[n] is None:
        raise ValueError("no valid segmentation")
    spans=[]
    end=n
    while end>0:
        a,cls=back[end]
        spans.append({"start":a,"end":end,"class_id":cls})
        end=a
    return spans[::-1]
