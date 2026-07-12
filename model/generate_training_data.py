#!/usr/bin/env python3
"""
generate_training_data.py — Build labeled training data for the ML file scorer.

For each repository under --root, this script:
  1. Lists all files (respecting gitignore via `git ls-files`, or plain walk).
  2. Extracts rule-based features and scores for every file.
  3. Sends batches of files to the local LLM for include/exclude labeling.
  4. Writes one JSONL record per file:
       {"path": "src/api.py", "label": 1, "features": {...}, "llm_reason": "..."}

The JSONL output is then consumed by train_ml_scorer.py.

Usage:
    python3 generate_training_data.py \\
        --root /path/to/repos \\
        --output training_data.jsonl \\
        [--batch-size 40] \\
        [--rule-only]        # skip LLM, label using rule_score threshold instead

Environment variables:
    LOCAL_LLM_BASE_URL   e.g. http://localhost:1234/v1
    LOCAL_LLM_MODEL      e.g. qwen2.5-coder-7b-instruct
    LOCAL_LLM_API_KEY    e.g. lm-studio
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

from model.file_features import extract_features, score_file, _IGNORE_DIRS

# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def list_repo_files(repo_root: Path) -> list[str]:
    """
    Return relative file paths inside a repo. Prefers `git ls-files` (which
    respects .gitignore automatically); falls back to a recursive walk that
    manually skips common noise directories.
    """
    git_dir = repo_root / ".git"
    if git_dir.exists():
        try:
            out = subprocess.check_output(
                ["git", "ls-files"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            files = [l.strip() for l in out.splitlines() if l.strip()]
            if files:
                return files
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # git not available or not a proper repo, fall through

    # Plain recursive walk — skip obvious noise
    files = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        parts_lower = [part.lower() for part in p.relative_to(repo_root).parts]
        if any(part in _IGNORE_DIRS for part in parts_lower):
            continue
        files.append(str(p.relative_to(repo_root)))
    return sorted(files)


# ---------------------------------------------------------------------------
# LLM labeling
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a technical documentation expert. You will be given a list of file paths
from a software repository. Your task is to decide which files are worth scanning
to generate a developer cheatsheet or API reference.

A file is worth scanning (label=1) if it:
- Contains source code with public APIs, classes, or functions
- Is a README, usage guide, or example/tutorial
- Is a project manifest or config that describes capabilities or dependencies

A file is NOT worth scanning (label=0) if it:
- Is a test file (unit tests, fixtures, mocks)
- Is auto-generated, compiled, or a binary artifact
- Is deep inside a vendor/node_modules/build directory
- Contains only data (CSV, images, lock files)
- Is CI/CD pipeline config unrelated to the project's API

Respond with ONLY a JSON array (no markdown, no preamble). Each element:
{"path": "<path>", "label": <0 or 1>, "reason": "<one sentence>"}
"""


def _call_llm(file_list: list[str], base_url: str, model: str, api_key: str) -> list[dict]:
    """Send a batch of file paths to the LLM and parse its JSON response."""
    user_msg = "File paths to classify:\n" + "\n".join(f"- {p}" for p in file_list)

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,  # deterministic output for reproducible labels
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    raw = payload["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if the model wrapped its output
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()

    try:
        results = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned non-JSON output:\n{raw[:500]}"
        ) from e

    if not isinstance(results, list):
        raise RuntimeError(f"Expected a JSON array, got: {type(results)}")

    return results


def _label_via_llm(
    files: list[str], base_url: str, model: str, api_key: str, batch_size: int
) -> dict[str, dict]:
    """
    Run LLM labeling in batches. Returns a dict mapping path → {label, reason}.
    Files that the LLM didn't include in its response fall back to rule_score.
    """
    import re  # local import so the module is usable without it
    results = {}
    total = len(files)
    for i in range(0, total, batch_size):
        batch = files[i : i + batch_size]
        print(f"  LLM labeling batch {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}"
              f" ({len(batch)} files)...")
        try:
            llm_results = _call_llm(batch, base_url, model, api_key)
            for item in llm_results:
                path = item.get("path", "").strip()
                label = item.get("label")
                reason = item.get("reason", "")
                if path and label in (0, 1):
                    results[path] = {"label": label, "reason": reason}
        except RuntimeError as e:
            print(f"  ⚠️ LLM batch failed: {e} — falling back to rule score for this batch",
                  file=sys.stderr)
    return results


def _rule_label(rule_score: float, threshold: float = 0.45) -> int:
    """Convert a rule score to a binary label using a threshold."""
    return 1 if rule_score >= threshold else 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(root_dir: str, output_path: str, batch_size: int, rule_only: bool):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"❌ Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Collect all repo sub-directories (each direct child is treated as a repo)
    repo_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()
                 and not d.name.startswith(".")]
    if not repo_dirs:
        # root itself might be a single repo
        repo_dirs = [root]

    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
    model    = os.environ.get("LOCAL_LLM_MODEL", "gpt-3.5-turbo")
    api_key  = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")

    use_llm = (not rule_only) and bool(base_url)
    if not use_llm and not rule_only:
        print("⚠️ LOCAL_LLM_BASE_URL not set — labeling with rule score only.\n"
              "   Set LOCAL_LLM_BASE_URL or pass --rule-only to suppress this warning.",
              file=sys.stderr)

    total_records = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for repo_dir in repo_dirs:
            repo_name = repo_dir.name
            print(f"\n📂 Repo: {repo_name}")
            files = list_repo_files(repo_dir)
            print(f"   {len(files)} files found")
            if not files:
                continue

            # Extract features for all files
            features_map = {f: extract_features(f) for f in files}

            # LLM labels (optional)
            llm_labels: dict[str, dict] = {}
            if use_llm:
                llm_labels = _label_via_llm(files, base_url, model, api_key, batch_size)

            # Write one JSONL record per file
            for f in files:
                feats = features_map[f]
                rule_sc = feats["rule_score"]

                if f in llm_labels:
                    label  = llm_labels[f]["label"]
                    reason = llm_labels[f]["reason"]
                    source = "llm"
                else:
                    label  = _rule_label(rule_sc)
                    reason = f"rule_score={rule_sc:.3f}"
                    source = "rule"

                record = {
                    "repo":     repo_name,
                    "path":     f,
                    "label":    label,
                    "source":   source,    # "llm" or "rule" — useful for analysis
                    "reason":   reason,
                    "features": feats,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

    print(f"\n✅ Training data written: {output_path} ({total_records} records)")
    _print_stats(output_path)


def _print_stats(output_path: str):
    """Print a quick label distribution summary."""
    counts = {"llm": {0: 0, 1: 0}, "rule": {0: 0, 1: 0}}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "rule")
            lbl = r.get("label", 0)
            counts.setdefault(src, {0: 0, 1: 0})[lbl] += 1

    print("\n📊 Label distribution:")
    for src, dist in counts.items():
        total = dist[0] + dist[1]
        if total == 0:
            continue
        pct = dist[1] / total * 100
        print(f"  [{src}] include={dist[1]}  exclude={dist[0]}  "
              f"include_rate={pct:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled training data for the ML file scorer"
    )
    parser.add_argument("--root",       required=True,
                        help="Root folder containing one sub-directory per repo")
    parser.add_argument("--output",     default="training_data.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--batch-size", type=int, default=40,
                        help="Number of files sent to the LLM per request")
    parser.add_argument("--rule-only",  action="store_true",
                        help="Skip LLM; label using rule score threshold only")
    args = parser.parse_args()
    generate(args.root, args.output, args.batch_size, args.rule_only)


if __name__ == "__main__":
    main()
