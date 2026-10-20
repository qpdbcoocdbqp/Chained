"""
scorer.py — NNScorer: the trained RepoContextNet ensemble with the token
vocabulary and feature scaling it was trained with.  This is what
collector.py loads (a .pt file); training lives in model/nn/train.py.

    scorer = NNScorer.load("nn_scorer.pt")
    scores = scorer.score_repo(repo_root, all_files)   # {rel_path: score}
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from model.nn.net import RepoContextNet
from model.nn.repo_features import FEATURE_VERSION, NUMERIC_COLUMNS, RepoFeatures, RepoTensors

PAD, UNK = 0, 1
# collector.py keeps the scorer's top MAX_SECONDARY_FILES files scored at
# least this.  Out-of-sample, the 40th best file of a tuning repo scores
# 0.17 (go-redis) to 0.80 (vite), so the floor only drops junk in tiny repos.
MIN_SCORE = 0.05


@dataclass
class Config:
    """Architecture and training settings; saved with the model."""
    d_tok: int = 32
    d_hidden: int = 64
    dropout: float = 0.2
    context: bool = True            # pool over parent / top-level directory and repo
    use_tokens: bool = True         # path token embeddings
    drop_columns: list[str] = field(default_factory=list)   # NUMERIC_COLUMNS zeroed (ablation)
    min_repos: int = 2              # a token needs to occur in this many training repos
    epochs: int = 50                # more overfits: LORO gold AP 0.50 at 50, 0.42 at 400
    lr: float = 3e-3
    weight_decay: float = 1e-3
    token_dropout: float = 0.1
    pos_weight: float = 5.0         # BCE weight of a gold file
    core_weight: float = 2.0        # extra factor for a core file, BCE and ranking loss
    w_bce: float = 1.0
    w_rank: float = 1.0             # listwise softmax loss over the repo's files
    w_llm: float = 0.0              # auxiliary LLM-label loss; 0.5 left LORO gold AP equal, core AP lower
    seeds: int = 5                  # ensemble members


class NNScorer:
    def __init__(self, cfg: Config, vocab: dict[str, int], mean: np.ndarray, std: np.ndarray,
                 states: list[dict] | None = None, threshold: float = MIN_SCORE, device: str | None = None):
        self.cfg = cfg
        self.vocab = vocab
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.keep = np.array([c not in cfg.drop_columns for c in NUMERIC_COLUMNS], np.float32)
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.nets = []
        for s in states or []:
            net = self.new_net()
            net.load_state_dict(s)
            net.eval()
            self.nets.append(net)

    # -- construction --------------------------------------------------------------

    @classmethod
    def fit_preprocessing(cls, xs: list[RepoTensors], cfg: Config) -> "NNScorer":
        """Vocabulary (tokens found in >= cfg.min_repos repos) and numeric
        scaling from the training repos; no networks yet."""
        repos_with = Counter()
        for x in xs:
            repos_with.update({t for toks in x.tokens for t in toks})
        vocab = {t: i + 2 for i, t in enumerate(sorted(t for t, c in repos_with.items() if c >= cfg.min_repos))}
        numeric = np.concatenate([x.numeric for x in xs])
        return cls(cfg, vocab, numeric.mean(0), numeric.std(0).clip(min=1e-3))

    def new_net(self) -> RepoContextNet:
        c = self.cfg
        return RepoContextNet(len(self.vocab) + 2, len(NUMERIC_COLUMNS), c.d_tok, c.d_hidden,
                              c.dropout, c.context).to(self.device)

    # -- inputs ----------------------------------------------------------------------

    def encode(self, x: RepoTensors) -> dict[str, torch.Tensor]:
        """Network inputs of one repo, on self.device."""
        ids, offsets = [], []
        for toks in x.tokens:
            offsets.append(len(ids))
            if self.cfg.use_tokens:
                ids += [self.vocab.get(t, UNK) for t in toks] or [UNK]
            else:
                ids.append(UNK)
        numeric = (x.numeric - self.mean) / self.std * self.keep
        dev = self.device
        return {
            "tokens": torch.tensor(ids, dtype=torch.long, device=dev),
            "offsets": torch.tensor(offsets, dtype=torch.long, device=dev),
            "numeric": torch.tensor(numeric, dtype=torch.float32, device=dev),
            "parent": torch.tensor(x.parent, dtype=torch.long, device=dev),
            "top": torch.tensor(x.top, dtype=torch.long, device=dev),
        }

    # -- inference -------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: RepoTensors) -> np.ndarray:
        """Gold probability of every file, averaged over the ensemble."""
        if not self.nets:
            raise ValueError("NNScorer has no trained networks")
        inputs = self.encode(x)
        probs = [torch.sigmoid(net(**inputs)[:, 0]) for net in self.nets]
        return torch.stack(probs).mean(0).cpu().numpy()

    def score_repo(self, root: str | Path, files: list[str], repo_name: str | None = None) -> dict[str, float]:
        """Scores of `files` (every file of the checkout, as collector.Repo.files
        lists them: directory context comes from the whole list), keyed by the
        paths as given."""
        x = RepoFeatures(root, files, repo_name).build()
        return dict(zip(files, self.predict(x).tolist()))

    # -- persistence -----------------------------------------------------------------

    def save(self, path: str | Path, **meta) -> None:
        torch.save({
            "feature_version": FEATURE_VERSION,
            "numeric_columns": NUMERIC_COLUMNS,
            "config": asdict(self.cfg),
            "vocab": self.vocab,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "threshold": self.threshold,
            "states": [{k: v.cpu() for k, v in net.state_dict().items()} for net in self.nets],
            "meta": meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "NNScorer":
        data = torch.load(path, map_location="cpu", weights_only=True)
        # Fail loudly on another feature layout, as FileScorer.load does
        if data["feature_version"] != FEATURE_VERSION or list(data["numeric_columns"]) != NUMERIC_COLUMNS:
            raise ValueError(f"{path}: trained on features {data['feature_version']}, "
                             f"not {FEATURE_VERSION}. Retrain the model.")
        return cls(Config(**data["config"]), data["vocab"], np.array(data["mean"]), np.array(data["std"]),
                   data["states"], data["threshold"], device)
