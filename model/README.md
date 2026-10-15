# File Scoring

Ranks every file in a repo by how useful it is for a **user** cheatsheet, so
`collector.py` can pick which files to put in the LLM context.

- `file_features.py`              ← Path features, `rule_score`, import reference counts (shared)
- `generate_training_data.py`     ← Labels files (LLM or rules) and writes training JSONL
- `train_ml_scorer.py`            ← Training + evaluation + inference (`FileScorer`)
- `tmp/eval_llm_labels.py`        ← Regression check of the LLM labeling against a hand-labeled gold set

```bash
uv pip install scikit-learn python-dotenv
```

Repos used for training live under `source/`, one directory per repo:

```bash
git clone https://github.com/ggml-org/llama.cpp.git source/llama.cpp
git clone https://github.com/milvus-io/milvus.git source/milvus
```

# Step 1: Generate training data

```bash
# .env or environment
export LOCAL_LLM_BASE_URL=http://localhost:19001/v1   # any OpenAI-compatible endpoint
export LOCAL_LLM_MODEL=sonnet

python -m model.generate_training_data --root ./source \
    --output ./model/tmp/llm_training_data_v3.jsonl --max-llm-files 600

# Without an LLM: rule-based labels for a cold start
python -m model.generate_training_data --root ./source \
    --output ./model/tmp/training_data.jsonl --rule-only
```

| Option | Default | Meaning |
|---|---|---|
| `--max-llm-files N` | all | Per-repo cap on files sent to the LLM. Half the budget goes to files with `rule_score >= 0.3`, half is a random sample of the rest |
| `--batch-size N` | 20 | Files per LLM request |
| `--no-content` | off | Send paths only, without content snippets |
| `--no-verify` | off | Skip the second pass that re-checks label-1 files |
| `--cache PATH` | `model/tmp/llm_label_cache.jsonl` | Label cache, `''` to disable |
| `--seed N` | 42 | Seed for the `--max-llm-files` sample |

Each output record has a `source`:
- `auto` — certain negatives labeled without the LLM: tests, mocks, vendored / generated code (`.pb.go`, `DO NOT EDIT` headers), binaries, assets, lock / build / lint files, CI
- `llm`  — labeled by the LLM; the record also stores the LLM's `category`
- `rule` — not sent to the LLM (beyond `--max-llm-files`); label = `rule_score >= 0.45`

## How LLM labeling works

1. Each batch is sent with repo context (name, top-level entries, README excerpt) and, per file: path, size, **location facts** (`repo root`, `1 of 81 .py files in conversion/`, `inside a 'developer_guides' docs directory`) and a 400-char content snippet with the license header stripped.
2. The LLM does not answer 0/1. It puts each file into one of the categories in `CATEGORIES` (`user_doc`, `example_main`, `api_entry`, `user_script`, `install_manifest` → 1; `per_variant_impl`, `sdk_internal`, `internal_impl`, `tool_impl`, `ui_source`, `example_aux`, `maintainer_doc`, `ops_runbook`, `support_lib`, `test_build_ci`, `asset_generated` → 0) and the label is derived from the table. When a file fits both an include and an exclude category, exclude wins.
3. Rules decidable from the path are enforced in code (`_policy_override`): Go `*_example_test.go` and `examples/<x>/README*` are always `example_main`.
4. Files labeled 1 are re-labeled once more in a differently mixed batch (with some negatives as filler) and keep label 1 only when both passes agree. Errors are mostly false positives that flip with batch composition.
5. Batches are shuffled rather than path-sorted, so neighbouring files are not labeled by analogy. Items are referenced by number; malformed JSON is salvaged by regex and logged to `model/tmp/llm_failed_replies.log`.

Labels are cached per `PROMPT_VERSION`, model, and content / verify flags, so after changing `file_features.py` you can regenerate features and retrain without relabeling. **Bump `PROMPT_VERSION` whenever `_SYSTEM_PROMPT`, `CATEGORIES`, the per-file facts or `_policy_override` change.**

## Checking label quality

Run this after any change to the prompt or labeling code:

```bash
python -m model.tmp.eval_llm_labels --repeat 3   # accuracy / precision / recall + self-consistency
python -m model.tmp.eval_llm_labels --no-verify  # compare without the verification pass
```

To inspect a small real sample, label 25 files per repo and read the `llm` records:

```bash
python -m model.generate_training_data --root ./source --output ./model/tmp/llm_sample50.jsonl --max-llm-files 25
```

The run ends with an "LLM vs rule" summary; the disagreements are what the ML model can learn beyond `rule_score`.

# Step 2: Train

```bash
# LLM data — exclude `rule` records: their label is a threshold on the
# rule_score feature, so the model would just relearn that threshold
python -m model.train_ml_scorer train --data ./model/tmp/llm_training_data_v3.jsonl \
    --model llm_scorer_v3.pkl --sources llm,auto

# Rule-only data (cold start). The model then reproduces rule_score >= 0.45.
python -m model.train_ml_scorer train --data ./model/tmp/training_data_v3.jsonl --model scorer_v3.pkl

# Evaluate a model on a JSONL file
python -m model.train_ml_scorer eval --data ./model/tmp/llm_training_data_v3.jsonl \
    --model llm_scorer_v3.pkl --sources llm,auto
```

`FileScorer.load` refuses a model whose feature columns differ from `FEATURE_COLUMNS`. Any change to the feature set (`in_maintainer_docs`, `in_ops`, …) requires regenerating the training data and retraining.

# Step 3: Inference (integrated with git ls-files)

```bash
git -C source/llama.cpp ls-files | grep docs | python -m model.train_ml_scorer score --model llm_scorer_v3.pkl --top-k 15
```

The `score` command computes features from the path only (`ref_count = 0`).

# Step 4: collector.py

`collector.py` loads the first model in `SCORER_MODELS` that matches the current features (`scorer_v3.pkl`, `scorer_v2.pkl`, …), and falls back to `filescore.priority_score` when none loads. To use a newly trained model, put it first in `SCORER_MODELS`.

`collect_secondary_files` then:
1. computes `ref_count` (non-test importers) and example reference counts (importers under `examples/` or Go `*_example_test.go`) with `build_ref_counts`;
2. scores every file and keeps those `>= SECONDARY_THRESHOLD` (0.5);
3. ranks with tie-breakers: model score → `rule_score` → content signal (code blocks in docs, stub files last) → depth → `ref_count`;
4. reserves code slots: `MIN_API_CODE_FILES` (3) public-API files ranked by example references, then `MIN_EXAMPLE_CODE_FILES` (2) example programs;
5. fills `MAX_SECONDARY_FILES` (20), with caps of `MAX_PER_PARENT_DIR` (3) per directory and `MAX_PER_TOP_DIR` (5) per top-level directory, skipping files already collected as priority files.

```py
from model.train_ml_scorer import FileScorer
from model.file_features import build_ref_counts

scorer = FileScorer.load("llm_scorer_v3.pkl")
example_refs = {}
ref_counts = build_ref_counts(repo_root, all_files, example_counts=example_refs)
ranked = scorer.rank(all_files, ref_counts)   # [(score, path), ...] descending
```
