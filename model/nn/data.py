"""
data.py — Training repos for the neural scorer: RepoFeatures of every file
joined with its gold label (model/tmp/gold/<repo>.yaml) and LLM label
(generate_training_data.py JSONL).  Features are cached per repo under
model/tmp/nn_cache/, keyed on the checkout's commit and FEATURE_VERSION.
"""
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from model.nn.repo_features import FEATURE_VERSION, RepoFeatures, RepoTensors

MODEL_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = MODEL_DIR / "tmp" / "gold"
CACHE_DIR = MODEL_DIR / "tmp" / "nn_cache"
DEFAULT_LLM_LABELS = MODEL_DIR / "tmp" / "colab" / "llm_training_data_qwen3.5-2b.jsonl"
# Final test repos, never trained on while tuning (same as tmp/tune_xgb_gold.HOLDOUT)
HOLDOUT = ("httpx", "ollama")


@dataclass
class RepoData:
    name: str
    x: RepoTensors
    scored: np.ndarray      # bool: scored by collector (not a priority file)
    gold: np.ndarray        # 1 for core / useful
    core: np.ndarray        # 1 for core
    llm: np.ndarray         # LLM label, -1 where the file has none


def repo_files(root: Path) -> tuple[list[str], list[str]]:
    """All files and the priority files, as collector.py sees them."""
    import collector

    repo = collector.Repo(root)
    return repo.files, collector.FileSelector.priority(repo)


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_features(root: Path, name: str) -> tuple[RepoTensors, list[str]]:
    """RepoTensors of a checkout and its priority files, cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}.pkl"
    key = (FEATURE_VERSION, _commit(root))
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data["key"] == key:
            return data["x"], data["priority"]
    files, priority = repo_files(root)
    x = RepoFeatures(root, files, name).build()
    with open(cache, "wb") as f:
        pickle.dump({"key": key, "x": x, "priority": priority}, f)
    return x, priority


def load_llm_labels(path: Path | None) -> dict[tuple[str, str], int]:
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") in (0, 1) and r.get("source") in ("llm", "auto"):
                out[(r["repo"], r["path"].replace("\\", "/"))] = int(r["label"])
    return out


def load_repos(source: Path, llm_path: Path | None = DEFAULT_LLM_LABELS,
               names: list[str] | None = None) -> list[RepoData]:
    """Every gold repo with a checkout under `source` (or just `names`)."""
    llm = load_llm_labels(llm_path)
    out = []
    for path in sorted(GOLD_DIR.glob("*.yaml")):
        gold = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = gold["repo"]
        if names is not None and name not in names:
            continue
        root = source / name
        if not root.exists():
            print(f"skipping {name}: no checkout at {root}")
            continue
        x, priority = load_features(root, name)
        prio = {p.replace(os.sep, "/") for p in priority}
        core, useful = set(gold["core"]), set(gold.get("useful") or [])
        missing = (core | useful) - set(x.files)
        if missing:
            print(f"⚠️ {name}: gold files not in the checkout: {sorted(missing)}")
        out.append(RepoData(
            name=name, x=x,
            scored=np.array([f not in prio for f in x.files]),
            gold=np.array([f in core | useful for f in x.files], np.float32),
            core=np.array([f in core for f in x.files], np.float32),
            llm=np.array([llm.get((name, f), -1) for f in x.files], np.float32),
        ))
    return out
