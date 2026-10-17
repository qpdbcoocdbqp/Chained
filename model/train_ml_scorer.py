#!/usr/bin/env python3
"""
train_ml_scorer.py — Train a gradient-boosted ML scorer from JSONL training
data, evaluate it, and save the model for use in collector.py.

The trained model accepts the same feature vector produced by
file_features.extract_features() and outputs a probability score in [0, 1]
indicating how likely a file is worth scanning for documentation generation.

Usage:
    # Train and save
    python3 train_ml_scorer.py train \\
        --data training_data.jsonl \\
        --model scorer.pkl \\
        [--test-split 0.2]

    # Evaluate an existing model against a JSONL file
    python3 train_ml_scorer.py eval \\
        --data training_data.jsonl \\
        --model scorer.pkl

    # Score files from stdin (one path per line) using a saved model
    python3 train_ml_scorer.py score \\
        --model scorer.pkl \\
        [--top-k 50] \\
        [--threshold 0.5]
    # e.g.: git ls-files | python3 train_ml_scorer.py score --model scorer.pkl

Dependencies:
    pip install scikit-learn xgboost
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

from model.file_features import FEATURE_COLUMNS, extract_features, features_to_vector


def load_jsonl(path: str, sources: set[str] | None = None) -> tuple[list, list, list]:
    """
    Read a JSONL file produced by generate_training_data.py.

    sources : keep only records whose "source" is in this set (e.g.
              {"llm", "auto"}); None keeps everything.

    Returns X (feature vectors), y (0/1 labels) and meta (dicts with
    "repo", "path", "source", "reason").
    """
    X, y, meta = [], [], []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue
            feats, label = record.get("features"), record.get("label")
            if feats is None or label not in (0, 1):
                continue
            if sources is not None and record.get("source", "rule") not in sources:
                continue
            if not all(col in feats for col in FEATURE_COLUMNS):
                print(f"⚠️ Skipping line {lineno}: features lack columns in "
                      f"FEATURE_COLUMNS; regenerate the training data", file=sys.stderr)
                continue
            X.append(features_to_vector(feats))
            y.append(int(label))
            meta.append({k: record.get(k, "") for k in ("repo", "path", "source", "reason")})
    return X, y, meta


# Random search over 200 configs on the qwen2.5-7b Colab labels, scored by
# PR-AUC averaged over stratified 5-fold CV and leave-one-repo-out (train on
# llama.cpp, test on milvus and vice versa). Against the previous sklearn GBT:
# CV 0.637 vs 0.617, leave-one-repo-out 0.492 vs 0.437, stable across seeds.
# scale_pos_weight stays 1 so probabilities match collector.py's 0.5 cut-off;
# monotone constraints on rule_score did not help.
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=3,
    subsample=1.0, colsample_bytree=1.0, reg_lambda=0.5, gamma=1,
    tree_method="hist", random_state=42,
)


class FileScorer:
    """
    XGBoost classifier over the file_features vector.

    Boosted trees because feature interactions matter (rule_score matters
    differently by depth), features need no scaling, and it stays small and
    fast at this dimensionality.

    Usage
    -----
    scorer = FileScorer.load("scorer.pkl")
    files = scorer.rank(["src/api.py", "tests/test_api.py", "README.md"])
    # files is a list of (score, path) sorted by score descending
    """

    def __init__(self, model):
        self._model = model

    @classmethod
    def train(cls, X: list, y: list) -> "FileScorer":
        from xgboost import XGBClassifier

        return cls(XGBClassifier(**XGB_PARAMS).fit(X, y))

    def save(self, model_path: str, training_samples: int) -> None:
        with open(model_path, "wb") as f:
            pickle.dump({"model": self._model, "feature_columns": FEATURE_COLUMNS,
                         "training_samples": training_samples}, f)

    @classmethod
    def load(cls, model_path: str) -> "FileScorer":
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        # Fail loudly on a different feature layout: a mismatch otherwise surfaces
        # as a ValueError inside predict_proba, which callers tend to swallow.
        if list(data.get("feature_columns", [])) != FEATURE_COLUMNS:
            raise ValueError(f"{model_path}: trained on a different feature layout "
                             f"than file_features.FEATURE_COLUMNS. Retrain the model.")
        return cls(data["model"])

    def predict(self, X: list) -> list[float]:
        """Probability of label 1 for each feature vector."""
        return self._model.predict_proba(X)[:, 1].tolist() if len(X) else []

    def score(self, rel_path: str) -> float:
        """Return a probability score in [0, 1] for a single file path."""
        return self.predict([features_to_vector(extract_features(rel_path))])[0]

    def rank(self, rel_paths: list[str]) -> list[tuple[float, str]]:
        """Score and rank relative paths. Returns [(score, path), ...] descending."""
        X = [features_to_vector(extract_features(p)) for p in rel_paths]
        return sorted(zip(self.predict(X), rel_paths), reverse=True)

    def filter(self, rel_path: str, threshold: float = 0.5) -> bool:
        """Return True if the file should be scanned."""
        return self.score(rel_path) >= threshold

    def report(self, X: list, y: list) -> list[float]:
        """Print the classification report and ROC-AUC at threshold 0.5;
        returns the predicted probabilities."""
        from sklearn.metrics import classification_report, roc_auc_score

        proba = self.predict(X)
        print(classification_report(y, [int(p >= 0.5) for p in proba], target_names=["exclude", "include"]))
        print(f"   ROC-AUC: {roc_auc_score(y, proba):.4f}")
        return proba

    def print_importances(self, top: int = 10) -> None:
        importances = self._model.feature_importances_
        print(f"\n🔍 Feature importances (top {top}):")
        for name, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)[:top]:
            print(f"  {name:<22} {imp:.4f}  {'█' * int(imp * 40)}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _load_data(data_path: str, sources: set[str] | None) -> tuple[list, list, list]:
    X, y, meta = load_jsonl(data_path, sources)
    if not X:
        print("❌ No valid records found.", file=sys.stderr)
        sys.exit(1)
    return X, y, meta


def cmd_train(data_path: str, model_path: str, test_split: float, sources: set[str] | None = None):
    from sklearn.model_selection import train_test_split

    print(f"📂 Loading training data from {data_path} "
          f"(sources: {','.join(sorted(sources)) if sources else 'all'}) ...")
    X, y, _ = _load_data(data_path, sources)
    n_pos = sum(y)
    print(f"   {len(X)} samples  |  include={n_pos}  exclude={len(y) - n_pos}  "
          f"({n_pos / len(y) * 100:.1f}% positive)")

    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y)
    else:
        X_train, y_train, X_test, y_test = X, y, [], []

    print(f"\n🏋️  Training on {len(X_train)} samples ...")
    scorer = FileScorer.train(X_train, y_train)
    if X_test:
        print("\n📊 Evaluation on held-out test set:")
        scorer.report(X_test, y_test)

    scorer.save(model_path, len(X_train))
    print(f"\n✅ Model saved: {model_path}")
    scorer.print_importances()


def cmd_eval(data_path: str, model_path: str, sources: set[str] | None = None):
    scorer = FileScorer.load(model_path)
    X, y, meta = _load_data(data_path, sources)
    print(f"\n📊 Evaluation on {len(X)} samples from {data_path}:")
    proba = scorer.report(X, y)

    # Worst mistakes: high-confidence wrong predictions
    mistakes = sorted(((p if p >= 0.5 else 1 - p), true, m["path"], m["reason"])
                      for true, p, m in zip(y, proba, meta) if true != int(p >= 0.5))[::-1]
    if mistakes:
        print("\n❌ Top 10 high-confidence mistakes:")
        for conf, true, path, reason in mistakes[:10]:
            print(f"  [{conf:.2f}] true={'include' if true else 'exclude'} "
                  f"pred={'exclude' if true else 'include'}  {path}")
            print(f"         reason: {reason}")


def cmd_score(model_path: str, top_k: int, threshold: float):
    """Read file paths from stdin, score them, print results sorted by score."""
    scorer = FileScorer.load(model_path)
    paths = [line.strip() for line in sys.stdin if line.strip()]
    if not paths:
        print("No file paths received on stdin.", file=sys.stderr)
        sys.exit(1)

    print(f"{'score':>6}  path")
    print("-" * 60)
    for prob, path in scorer.rank(paths)[:top_k or None]:
        print(f"{prob:>6.3f}  {'✅' if prob >= threshold else '  '} {path}")


def main():
    parser = argparse.ArgumentParser(description="Train and use the ML file scorer")
    sub = parser.add_subparsers(dest="command", required=True)
    sources_help = ("Comma-separated record sources to use, e.g. 'llm,auto' "
                    "(default: all). Rule labels are derived from the rule_score "
                    "feature, so mixing them in makes the model circular.")

    p_train = sub.add_parser("train", help="Train a model from JSONL training data")
    p_train.add_argument("--data", required=True, help="Path to training_data.jsonl")
    p_train.add_argument("--model", default="scorer.pkl", help="Output model file")
    p_train.add_argument("--test-split", type=float, default=0.2,
                         help="Fraction held out for evaluation (0 to skip)")
    p_train.add_argument("--sources", default=None, help=sources_help)

    p_eval = sub.add_parser("eval", help="Evaluate a saved model against a JSONL file")
    p_eval.add_argument("--data", required=True, help="Path to evaluation JSONL")
    p_eval.add_argument("--model", default="scorer.pkl")
    p_eval.add_argument("--sources", default=None, help=sources_help)

    p_score = sub.add_parser("score", help="Score file paths from stdin using a saved model")
    p_score.add_argument("--model", default="scorer.pkl")
    p_score.add_argument("--top-k", type=int, default=0, help="Only show top K results (0 = all)")
    p_score.add_argument("--threshold", type=float, default=0.5,
                         help="Score threshold for the include marker")

    args = parser.parse_args()
    sources = ({s.strip() for s in args.sources.split(",") if s.strip()}
               if getattr(args, "sources", None) else None)

    if args.command == "train":
        cmd_train(args.data, args.model, args.test_split, sources)
    elif args.command == "eval":
        cmd_eval(args.data, args.model, sources)
    elif args.command == "score":
        cmd_score(args.model, args.top_k, args.threshold)


if __name__ == "__main__":
    main()
