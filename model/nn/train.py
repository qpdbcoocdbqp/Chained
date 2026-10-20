"""
train.py — Train and evaluate the neural file scorer against the cheatsheet
gold lists in model/tmp/gold/.

The training target is the gold list itself (core + useful = 1).  The LLM
labels of generate_training_data.py can be added as an auxiliary task
(Config.w_llm), which did not help (see model/nn/README.md).  Evaluation
follows tmp/tune_xgb_gold.py: every file collector.py scores (priority files
excluded), each repo scored by a model that never saw its labels.

    # how well each feature column alone ranks the gold files
    python -m model.nn.train signals

    # leave-one-repo-out over the tuning repos, with baselines and ablations
    python -m model.nn.train loro [--xgb] [--pkl llm_scorer_v8.pkl] [--ablate]

    # the same, end to end: the files collector.py actually picks
    python -m model.nn.train e2e [--pkl llm_scorer_v8.pkl]

    # final test on the holdout repos (once per tuning round)
    python -m model.nn.train holdout [--pkl llm_scorer_v8.pkl]

    # production model, trained on every gold repo
    python -m model.nn.train fit --out nn_scorer.pt

Ranking metrics per repo: AP over core + useful (gold), AP over core, P@20
(share of the top 20 that are gold; collector keeps 20 secondary files), R@20
(share of the core files in the top 20).  End-to-end metrics are those of tmp/eval_collector_gold.py: core recall and
hit rate (share of the picked files that are gold), priority files included.
"""
import argparse
import contextlib
import io
import os
import random
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from model.nn.data import DEFAULT_LLM_LABELS, HOLDOUT, RepoData, load_repos
from model.nn.repo_features import NUMERIC_COLUMNS
from model.nn.scorer import MIN_SCORE, UNK, Config, NNScorer

TOP_K = 20  # collector.MAX_SECONDARY_FILES

# Column groups for --ablate
GROUPS = {
    "content": ["size", "lines", "unreadable", "generated", "stub", "fences", "prompts", "headings",
                "md_links", "install_cmds", "cli_flags", "exported", "has_main", "go_examples"],
    "mentions": ["hub_link", "path_mentions", "stem_user_mentions", "stem_mentions", "ref", "example_ref",
                 "sib_stem_user_pct", "sib_stem_pct", "sib_path_pct"],
    "structure": ["is_root", "stem_is_parent", "go_example_test", "dir_files", "dir_same_ext",
                  "dir_same_ext_frac", "dir_subdirs", "top_share", "repo_files", "sib_size_pct"],
}
ABLATIONS = {
    "no context pooling": {"context": False},
    "no path tokens": {"use_tokens": False},
    "no content columns": {"drop_columns": GROUPS["content"]},
    "no mention columns": {"drop_columns": GROUPS["mentions"]},
    "no structure columns": {"drop_columns": GROUPS["structure"]},
    "no ranking loss": {"w_rank": 0.0},
    "+ LLM aux loss": {"w_llm": 0.5},
    "LLM labels only": {"w_bce": 0.0, "w_rank": 0.0, "w_llm": 1.0},
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_net(scorer: NNScorer, batches: list[tuple[dict, dict]], seed: int) -> torch.nn.Module:
    """One network; an epoch is one optimizer step per training repo (the
    directory pooling needs the whole repo in the batch)."""
    cfg = scorer.cfg
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = scorer.new_net()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(batches))
    net.train()
    for _ in range(cfg.epochs):
        rng.shuffle(batches)
        for inputs, y in batches:
            tokens = inputs["tokens"]
            if cfg.token_dropout:
                tokens = torch.where(torch.rand_like(tokens, dtype=torch.float) < cfg.token_dropout,
                                     torch.full_like(tokens, UNK), tokens)
            loss = repo_loss(net(**{**inputs, "tokens": tokens}), y, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


def repo_loss(logits: torch.Tensor, y: dict, cfg: Config) -> torch.Tensor:
    """Weighted BCE on the gold labels of the scored files, plus a listwise
    softmax loss putting the repo's probability mass on its gold files (core
    files weigh core_weight times more in both), plus the optional LLM-label
    BCE on the second logit."""
    m = y["scored"]
    g, gold, core = logits[:, 0][m], y["gold"][m], y["core"][m]
    w = gold * (1 + (cfg.core_weight - 1) * core)
    loss = cfg.w_bce * F.binary_cross_entropy_with_logits(g, gold, weight=1 + (cfg.pos_weight - 1) * w)
    if cfg.w_rank and gold.sum() > 0:
        loss = loss + cfg.w_rank * -(F.log_softmax(g, 0) * w).sum() / w.sum()
    if cfg.w_llm:
        lm = y["llm"] >= 0
        if lm.any():
            loss = loss + cfg.w_llm * F.binary_cross_entropy_with_logits(logits[:, 1][lm], y["llm"][lm])
    return loss


def targets(r: RepoData, device) -> dict:
    return {
        "scored": torch.tensor(r.scored, device=device),
        "gold": torch.tensor(r.gold, device=device),
        "core": torch.tensor(r.core, device=device),
        "llm": torch.tensor(r.llm, device=device),
    }


def train_scorer(repos: list[RepoData], cfg: Config) -> NNScorer:
    """Ensemble of cfg.seeds networks trained on `repos`."""
    scorer = NNScorer.fit_preprocessing([r.x for r in repos], cfg)
    batches = [(scorer.encode(r.x), targets(r, scorer.device)) for r in repos]
    scorer.nets = [train_net(scorer, list(batches), seed) for seed in range(cfg.seeds)]
    return scorer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def metrics(scores: np.ndarray, r: RepoData) -> dict:
    m = r.scored
    s, gold, core = scores[m], r.gold[m], r.core[m]
    order = np.argsort(-s, kind="stable")
    return {
        "gold": average_precision_score(gold, s),
        "core": average_precision_score(core, s) if core.sum() else float("nan"),
        "p@20": gold[order[:TOP_K]].mean(),
        "r@20": core[order[:TOP_K]].sum() / core.sum() if core.sum() else float("nan"),
    }


def show(title: str, rows: dict[str, dict], names: list[str], cols: tuple[str, ...]) -> None:
    fmt = {"gold": "{:.3f}", "core": "{:.3f}", "p@20": "{:.2f}", "r@20": "{:.2f}",
           "recall": "{:.2f}", "hit": "{:.2f}", "picked": "{:3.0f}"}
    width = sum(len(fmt[c].format(0)) + 1 for c in cols) + 1
    print(f"\n{title}")
    print(f"  {'':24s}" + "".join(f"{n[:width - 2]:>{width}s}" for n in names) + f"{'mean':>{width}s}")
    print(f"  {'':24s}" + ("  " + " ".join(f"{c:>{len(fmt[c].format(0))}s}" for c in cols)) * (len(names) + 1))
    for label, res in rows.items():
        cells = ""
        for n in [*names, "mean"]:
            v = res[n] if n != "mean" else {c: np.nanmean([res[k][c] for k in names]) for c in cols}
            cells += "  " + " ".join(fmt[c].format(v[c]) for c in cols)
        print(f"  {label:24s}{cells}")


def path_vectors(r: RepoData) -> np.ndarray:
    """file_features.FEATURE_COLUMNS of every file, the XGBoost scorer's input."""
    from model.file_features import extract_features, features_to_vector

    return np.array([features_to_vector(extract_features(f)) for f in r.x.files], float)


def xgb_scores(train: list[RepoData], test: RepoData) -> np.ndarray:
    """The XGBoost scorer (XGB_PARAMS, FEATURE_COLUMNS) trained on the LLM
    labels of `train`."""
    from xgboost import XGBClassifier

    from model.train_ml_scorer import XGB_PARAMS

    X, y = np.concatenate([path_vectors(r) for r in train]), np.concatenate([r.llm for r in train])
    model = XGBClassifier(**XGB_PARAMS).fit(X[y >= 0], y[y >= 0])
    return model.predict_proba(path_vectors(test))[:, 1]


XGB_GOLD_VARIANTS = {
    "XGB gold, XGB_PARAMS": {},
    "XGB gold, shallow": dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=3,
                              subsample=0.8, colsample_bytree=0.7, reg_lambda=5),
    "XGB gold, shallow+weight": dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=3,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=5, scale_pos_weight=20),
}


def xgb_gold_scores(train: list[RepoData], test: RepoData, **params) -> np.ndarray:
    """XGBoost on the same NUMERIC_COLUMNS and gold labels as RepoContextNet
    (no path tokens, no directory pooling; sibling percentile columns stay)."""
    from xgboost import XGBClassifier

    from model.train_ml_scorer import XGB_PARAMS

    X = np.concatenate([r.x.numeric[r.scored] for r in train])
    y = np.concatenate([r.gold[r.scored] for r in train])
    model = XGBClassifier(**{**XGB_PARAMS, **params}).fit(X, y)
    return model.predict_proba(test.x.numeric)[:, 1]


def rule_scores(r: RepoData) -> np.ndarray:
    return r.x.numeric[:, NUMERIC_COLUMNS.index("rule_score")]


def evaluate(cfgs: dict[str, Config], train_sets: dict[str, list[RepoData]], tests: list[RepoData],
             xgb: bool = False, pkl: str | None = None, xgb_gold: bool = False) -> dict[str, dict]:
    """Ranking metrics of rule_score, the baselines asked for and each
    config, every test repo scored by a model trained on train_sets[repo]."""
    rows: dict[str, dict] = {"rule_score": {r.name: metrics(rule_scores(r), r) for r in tests}}
    if pkl:
        from model.train_ml_scorer import FileScorer

        scorer = FileScorer.load(pkl)
        rows[Path(pkl).name] = {r.name: metrics(np.array(scorer.predict(path_vectors(r))), r) for r in tests}
    if xgb:
        rows["XGB on LLM labels"] = {r.name: metrics(xgb_scores(train_sets[r.name], r), r) for r in tests}
    if xgb_gold:
        for label, params in XGB_GOLD_VARIANTS.items():
            rows[label] = {r.name: metrics(xgb_gold_scores(train_sets[r.name], r, **params), r) for r in tests}
    for label, cfg in cfgs.items():
        t0 = time.time()
        rows[label] = {r.name: metrics(train_scorer(train_sets[r.name], cfg).predict(r.x), r) for r in tests}
        print(f"  {label}: {time.time() - t0:.0f}s")
    return rows


def collector_metrics(model: str, source: Path, r: RepoData, threshold: float | None = None) -> dict:
    """Run collector.FileSelector on the checkout as build_context_bundle
    does; core recall and hit rate of the files it picks."""
    import collector

    with contextlib.redirect_stdout(io.StringIO()):
        selector = collector.FileSelector(model, threshold)
        repo = collector.Repo(source / r.name)
        priority = selector.priority(repo)
        picked = {p.replace(os.sep, "/") for p in [*priority, *selector.secondary(repo, exclude=priority)]}
    gold = {f for f, g in zip(r.x.files, r.gold) if g}
    core = {f for f, c in zip(r.x.files, r.core) if c}
    return {"recall": len(picked & core) / len(core), "hit": len(picked & gold) / len(picked),
            "picked": len(picked)}


def evaluate_e2e(cfg: Config, train_sets: dict[str, list[RepoData]], tests: list[RepoData],
                 source: Path, pkl: str | None) -> dict[str, dict]:
    """End-to-end metrics with no model (rule_score), the pickle asked for
    and the neural scorer, each test repo's scorer trained on
    train_sets[repo]."""
    rows = {"rule_score (no model)": {r.name: collector_metrics("no_model.pkl", source, r) for r in tests}}
    if pkl:
        rows[Path(pkl).name] = {r.name: collector_metrics(str(Path(pkl).resolve()), source, r) for r in tests}
    rows["RepoContextNet"] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for r in tests:
            t0 = time.time()
            path = Path(tmp) / f"{r.name}.pt"
            train_scorer(train_sets[r.name], cfg).save(path)
            rows["RepoContextNet"][r.name] = collector_metrics(str(path), source, r)
            print(f"  {r.name}: {time.time() - t0:.0f}s")
    return rows


def signals(repos: list[RepoData], top: int) -> None:
    """AP of every NUMERIC_COLUMNS column on its own, per repo: the evidence
    for the feature set (a column can rank gold files low, so the better of
    the column and its negation is shown, negations marked with -)."""
    res = {}
    for i, col in enumerate(NUMERIC_COLUMNS):
        res[col] = {}
        for r in repos:
            m = r.scored
            v, y = r.x.numeric[m, i], r.gold[m]
            pos, neg = average_precision_score(y, v), average_precision_score(y, -v)
            res[col][r.name] = (pos, "+") if pos >= neg else (neg, "-")
    print(f"\n{'column':22s}" + "".join(f"{r.name[:9]:>11s}" for r in repos) + f"{'mean':>8s}")
    print(f"{'(base rate)':22s}" + "".join(f"{r.gold[r.scored].mean():11.3f}" for r in repos))
    order = sorted(res, key=lambda c: -np.mean([res[c][r.name][0] for r in repos]))
    for col in order[:top]:
        cells = "".join(f"{res[col][r.name][0]:10.3f}{res[col][r.name][1]}" for r in repos)
        print(f"{col:22s}{cells}{np.mean([res[col][r.name][0] for r in repos]):8.3f}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

RANK_COLS = ("gold", "core", "p@20", "r@20")
E2E_COLS = ("recall", "hit", "picked")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("signals", "loro", "e2e", "holdout", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--source", default="source", help="directory holding the gold repo checkouts")
        p.add_argument("--llm-labels", default=str(DEFAULT_LLM_LABELS),
                       help="generate_training_data.py JSONL, for the XGB baseline and w_llm > 0")
        p.add_argument("--seeds", type=int, default=Config.seeds)
        p.add_argument("--epochs", type=int, default=Config.epochs)
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="override Config fields, e.g. --set dropout=0.3 drop_columns=size,lines")
    for name in ("loro", "e2e", "holdout"):
        sub.choices[name].add_argument("--pkl", help="also score with a saved FileScorer pickle, "
                                       "e.g. llm_scorer_v8.pkl (as is, whatever it was trained on)")
    for name in ("loro", "holdout"):
        sub.choices[name].add_argument("--xgb-gold", action="store_true",
                                       help="also train XGBoost on the same features and gold labels")
        sub.choices[name].add_argument("--xgb", action="store_true",
                                       help="also train XGBoost (XGB_PARAMS) on the LLM labels")
    sub.choices["loro"].add_argument("--ablate", action="store_true", help="also run ABLATIONS")
    sub.choices["signals"].add_argument("--top", type=int, default=len(NUMERIC_COLUMNS))
    sub.choices["fit"].add_argument("--out", default="nn_scorer.pt")
    sub.choices["fit"].add_argument("--tuning-only", action="store_true",
                                    help="leave the holdout repos out of training")
    sub.choices["fit"].add_argument("--threshold", type=float, default=MIN_SCORE,
                                    help="collector keeps no file scored below this")
    args = ap.parse_args()

    cfg = Config(seeds=args.seeds, epochs=args.epochs)
    for kv in args.set:
        key, value = kv.split("=", 1)
        current = getattr(cfg, key)
        if isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes")
        elif isinstance(current, list):
            value = [v for v in value.split(",") if v]
        cfg = replace(cfg, **{key: type(current)(value)})
    source = Path(args.source)
    repos = load_repos(source, Path(args.llm_labels) if args.llm_labels else None)
    tuning = [r for r in repos if r.name not in HOLDOUT]
    holdout = [r for r in repos if r.name in HOLDOUT]
    print(f"device {'cuda' if torch.cuda.is_available() else 'cpu'}; tuning {[r.name for r in tuning]}, "
          f"holdout {[r.name for r in holdout]}")
    print(f"config {asdict(cfg)}")
    loro_sets = {r.name: [t for t in tuning if t is not r] for r in tuning}

    if args.cmd == "signals":
        signals(tuning, args.top)

    elif args.cmd == "loro":
        cfgs = {"RepoContextNet": cfg}
        if args.ablate:
            cfgs |= {k: replace(cfg, **v) for k, v in ABLATIONS.items()}
        rows = evaluate(cfgs, loro_sets, tuning, args.xgb, args.pkl, args.xgb_gold)
        show("leave-one-repo-out over the tuning repos", rows, [r.name for r in tuning], RANK_COLS)

    elif args.cmd == "e2e":
        rows = evaluate_e2e(cfg, loro_sets, tuning, source, args.pkl)
        show("collector.py picks, leave-one-repo-out over the tuning repos", rows,
             [r.name for r in tuning], E2E_COLS)

    elif args.cmd == "holdout":
        train_sets = {r.name: tuning for r in holdout}
        rows = evaluate({"RepoContextNet": cfg}, train_sets, holdout, args.xgb, args.pkl, args.xgb_gold)
        show("holdout, trained on every tuning repo", rows, [r.name for r in holdout], RANK_COLS)
        rows = evaluate_e2e(cfg, train_sets, holdout, source, args.pkl)
        show("holdout, collector.py picks", rows, [r.name for r in holdout], E2E_COLS)

    elif args.cmd == "fit":
        train = tuning if args.tuning_only else repos
        scorer = train_scorer(train, cfg)
        scorer.threshold = args.threshold
        print("\nin-sample (trained on these repos):")
        show("", {"RepoContextNet": {r.name: metrics(scorer.predict(r.x), r) for r in train}},
             [r.name for r in train], RANK_COLS)
        scorer.save(args.out, repos=[r.name for r in train])
        print(f"\n✅ saved {args.out}: {len(scorer.vocab)} tokens, {len(scorer.nets)} networks, "
              f"collector threshold {scorer.threshold:.2f}")


if __name__ == "__main__":
    main()
