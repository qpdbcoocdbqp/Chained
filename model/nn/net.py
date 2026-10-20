"""
net.py — RepoContextNet: scores every file of a repo at once.

    path tokens ──EmbeddingBag──┐
                                ├─ MLP ─> h_i  (file encoding)
    NUMERIC_COLUMNS ────────────┘
                                         │ mean-pool h over the file's parent directory,
                                         │ its top-level directory and the whole repo
                                         v
    [h_i, h_i - parent_i, parent_i, top_i, repo] ─ MLP ─> (gold logit, llm logit)

The pooling is what the gold lists ask for: a gold file is one of a few
picks among siblings that look alike (one of 298 docs/cmdline-opts pages), so
the head sees each file next to the average of its directory, and a
directory's weight depends on the rest of the repo (docs/ of a docs-heavy
repo vs. of a server).  Everything is O(files), so a 5k-file repo is one
forward pass.

The second logit can learn the LLM labels of generate_training_data.py as
an auxiliary task (scorer.Config.w_llm, off by default: it did not help);
only the gold logit is used for ranking.
"""
import torch
from torch import nn


def scatter_mean(h: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    """Mean of the rows of h per group id; [groups, dim]."""
    sums = torch.zeros(groups, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, index, h)
    counts = torch.bincount(index, minlength=groups).clamp(min=1).unsqueeze(1).to(h.dtype)
    return sums / counts


class RepoContextNet(nn.Module):
    def __init__(self, n_tokens: int, n_numeric: int, d_tok: int = 32, d_hidden: int = 64,
                 dropout: float = 0.2, context: bool = True):
        super().__init__()
        self.context = context
        self.tok = nn.EmbeddingBag(n_tokens, d_tok, mode="mean", padding_idx=0)
        self.encode = nn.Sequential(
            nn.Linear(d_tok + n_numeric, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
        )
        d_in = d_hidden * (5 if context else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_in), nn.Dropout(dropout),
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 2),
        )

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor, numeric: torch.Tensor,
                parent: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        """tokens / offsets: flat token ids and per-file offsets (EmbeddingBag);
        numeric: [n, n_numeric]; parent / top: [n] directory group ids.
        Returns [n, 2] logits (gold, llm)."""
        h = self.encode(torch.cat([self.tok(tokens, offsets), numeric], 1))
        if self.context:
            p = scatter_mean(h, parent, int(parent.max()) + 1)[parent]
            t = scatter_mean(h, top, int(top.max()) + 1)[top]
            r = h.mean(0, keepdim=True).expand_as(h)
            h = torch.cat([h, h - p, p, t, r], 1)
        return self.head(h)
