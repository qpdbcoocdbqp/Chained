# File Scoring

Ranks every file in a repo by how useful it is for a **user** cheatsheet, so
`collector.py` can pick which files to put in the LLM context.

`collector.py` uses the PyTorch scorer in [`nn/`](nn/README.md) (`nn_scorer.pt`),
trained on the gold lists in `tmp/gold/` with per-repo context features. The
XGBoost pipeline below (LLM labels → path features → `FileScorer`) is kept as
its baseline and as the fallback for a `.pkl` `SCORER_MODEL`.

- `nn/`                           ← Neural scorer: repo features, `RepoContextNet`, training / evaluation (see `nn/README.md`)
- `file_features.py`              ← `FeatureExtraction`: path features and `rule_score` (the model's inputs)
- `ref_counts.py`                 ← Import reference counts; `collector.py` tie-breakers, not a model feature
- `config.yaml`                   ← Directory / name / skip lists, patterns, `rule_score` weights and model `feature_columns`
- `generate_training_data.py`     ← Labels files (LLM or rules) and writes training JSONL
- `prompt.yaml`                   ← LLM labeling prompt, categories → label table, `version`
- `train_ml_scorer.py`            ← Training + evaluation + inference (`FileScorer`)
- `tmp/eval_llm_labels.py`        ← Regression check of the LLM labeling against a hand-labeled gold set
- `tmp/tune_xgb_gold.py`          ← Tunes `XGB_PARAMS` by PR-AUC against the cheatsheet gold lists in `tmp/gold/`

```bash
uv pip install torch scikit-learn xgboost python-dotenv pyyaml openai
```

Repos used for training live under `source/`, one directory per repo, each pinned to the commit its gold list in `tmp/gold/<repo>.yaml` refers to (`SOURCE_REPOS` in `colab_labeling.ipynb` holds the full hashes):

```bash
git clone https://github.com/ggml-org/llama.cpp.git source/llama.cpp && git -C source/llama.cpp checkout 07d9378
git clone https://github.com/milvus-io/milvus.git source/milvus && git -C source/milvus checkout 6c8462cefb
git clone --depth 1 --branch v9.23.0    https://github.com/redis/go-redis.git source/go-redis
git clone --depth 1 --branch 0.28.1     https://github.com/encode/httpx.git   source/httpx
git clone --depth 1 --branch curl-8_23_0 https://github.com/curl/curl.git     source/curl
git clone --depth 1 --branch v8.3.4     https://github.com/vitejs/vite.git    source/vite
git clone --depth 1 --branch v0.40.6    https://github.com/ollama/ollama.git  source/ollama
```

| Repo | Kind | Role |
|---|---|---|
| llama.cpp | C/C++ library + CLI tools | tuning |
| milvus | Go server + Go SDK | tuning |
| go-redis | Go library, API in the repo root | tuning |
| curl | C library + CLI, per-option reference docs | tuning |
| vite | TS monorepo, VitePress docs | tuning |
| httpx | Python library, implementation in `_*.py` | holdout |
| ollama | Go server + CLI + REST API | holdout |

Holdout repos are the final test set (see "Tuning against the gold lists").

# Step 1: Generate training data

```bash
# .env or environment
export LOCAL_LLM_BASE_URL=http://localhost:19001/v1   # any OpenAI-compatible endpoint
export LOCAL_LLM_MODEL=sonnet

python -m model.generate_training_data --root ./source \
    --output ./model/tmp/llm_training_data.jsonl --workers 8
```

| Option | Default | Meaning |
|---|---|---|
| `--batch-size N` | 20 | Files per LLM request |
| `--no-verify` | off | Skip the second pass that re-checks label-1 files |
| `--cache PATH` | `model/tmp/llm_label_cache.jsonl` | Label cache, `''` to disable |
| `--workers N` | 1 | LLM batches in flight at once; raise it (e.g. 8) for a batching server such as vLLM |

To label on a Colab GPU with an open model served by vLLM, open
[`colab_labeling.ipynb`](colab_labeling.ipynb) in Colab. It pins `source/` to the
local commits, keeps the label cache on Google Drive, and runs the gold-set check
before the full run. Labels from a different model do not reuse cached ones
(the model name is part of the cache key).

`LABEL_MODEL` in the notebook's section 2 picks the model: `'auto'` (Qwen2.5-7B /
14B / 32B-Instruct-AWQ by GPU memory), `'fast'` (Qwen2.5-3B-Instruct-AWQ, about
twice the batches in flight of the 7B on a T4), `'fastest'` (Qwen2.5-1.5B, for
experiments), or `'qwen3.5-4b'` / `'qwen3.5-2b'` / `'qwen3.5-0.8b'`. The Qwen3.5
models are hybrid linear-attention models served text only with their
multi-token-prediction head as speculative drafter (`qwen3_next_mtp`), and with
thinking switched off through `LOCAL_LLM_EXTRA_BODY`
(`{"chat_template_kwargs": {"enable_thinking": false}}`); 4B wants an L4 or
better, 2B fits ~68 batches at once even on a T4, where Qwen3.5 is untested. A
different model relabels every repo, so compare its gold-set check with the
7B's (accuracy 0.96, precision 0.91, recall 1.00) before training on its labels,
and set `LOCAL_LLM_MODEL` to its served name (e.g. `qwen3.5-2b`) when
regenerating from its cache.

Each output record has a `source`:
- `auto` — certain negatives labeled without the LLM: tests, mocks, vendored / generated code (`.pb.go`, `DO NOT EDIT` headers), binaries, assets, lock / build / lint files, CI
- `llm`  — labeled by the LLM; the record also stores the LLM's `category`

Every other file is sent to the LLM; files it fails to label are left out of the output.

## How LLM labeling works

1. Each batch is sent with repo context (name, top-level entries, README excerpt) and, per file: path, size, **location facts** (`repo root`, `1 of 81 .py files in conversion/`, `inside a 'developer_guides' docs directory`) and a 400-char content snippet with the license header stripped.
2. The LLM does not answer 0/1. It puts each file into one of the `categories` in `prompt.yaml` (`user_doc`, `example_main`, `api_entry`, `user_script`, `install_manifest` → 1; `per_variant_impl`, `sdk_internal`, `internal_impl`, `tool_impl`, `ui_source`, `example_aux`, `maintainer_doc`, `ops_runbook`, `support_lib`, `test_build_ci`, `asset_generated` → 0) and the label is derived from the table. When a file fits both an include and an exclude category, exclude wins.
3. Rules decidable from the path are enforced in code (`policy_override`), because the model applies them inconsistently across batches:
   - Go `*_example_test.go` and `examples/<x>/README*` are always `example_main`.
   - `.proto` files are never positive (`internal_impl`): the model called 15 of milvus `pkg/proto` `api_entry`.
   - `install_manifest` only for a root-level manifest (`names.user_config` in `config.yaml`, root `docker-compose*.yml`) or a config under `configs/`; other files become `test_build_ci` (`.devops/` Dockerfiles, nix files, nested requirements, cmake templates) or `user_doc` (READMEs).
   - Majority vote (`RepoLabeler.apply_policies`): in a directory with at least 10 LLM-labeled files of one extension whose most common category is `per_variant_impl`, the positives become `per_variant_impl` too (`conversion/*.py`).

   Policies are re-applied to cached labels on every run, so changing them needs no relabeling. Changed labels carry a `[policy: old -> new]` reason prefix.
4. Files labeled 1 are re-labeled once more in a differently mixed batch (with some negatives as filler) and keep label 1 only when both passes agree. Errors are mostly false positives that flip with batch composition.
5. Batches are shuffled rather than path-sorted, so neighbouring files are not labeled by analogy. Items are referenced by number; malformed JSON is salvaged by regex and logged to `model/tmp/llm_failed_replies.log`.

Labels are cached per prompt `version`, model and verify flag, so after changing `file_features.py` you can regenerate features and retrain without relabeling. **Bump `version` in `prompt.yaml` whenever the prompts, the categories or the per-file facts (`RepoLabeler.location_fact`) change.**

## Checking label quality

Run this after any change to the prompt or labeling code:

```bash
python -m model.tmp.eval_llm_labels --repeat 3   # accuracy / precision / recall + self-consistency
python -m model.tmp.eval_llm_labels --no-verify  # compare without the verification pass
```

# Step 2: Train

```bash
# --sources llm,auto drops `rule` records that older training data still
# contains: their label is a threshold on the rule_score feature
# --test-split 0: the saved model is trained on the training split only
python -m model.train_ml_scorer train \
    --data ./model/tmp/colab/llm_training_data_qwen2.5-7b-instruct-awq_v9.jsonl \
    --model llm_scorer_v8.pkl --sources llm,auto --test-split 0

# Evaluate a model on a JSONL file
python -m model.train_ml_scorer eval \
    --data ./model/tmp/colab/llm_training_data_qwen2.5-7b-instruct-awq_v9.jsonl \
    --model llm_scorer_v8.pkl --sources llm,auto
```

Labels in a training JSONL follow the policies of the run that wrote it. After changing `policy_override` or the majority vote, rerun Step 1: with every label cached it makes no LLM requests, but `LOCAL_LLM_MODEL` must match the cached model name (`qwen2.5-7b-instruct-awq` for the Colab cache):

```bash
LOCAL_LLM_BASE_URL=http://127.0.0.1:9/v1 LOCAL_LLM_MODEL=qwen2.5-7b-instruct-awq \
python -m model.generate_training_data --root ./source \
    --cache model/tmp/colab/llm_label_cache.jsonl \
    --output model/tmp/colab/llm_training_data_qwen2.5-7b-instruct-awq_v9.jsonl
```

## Tuning against the gold lists

`XGB_PARAMS` is picked by PR-AUC against `tmp/gold/<repo>.yaml` (`core` + `useful`), not against the LLM labels. Each gold repo is scored by a model trained on the other repos' labels, on the files `collector.py` scores (priority files excluded). Configs that leave fewer files at or above 0.5 than `MAX_SECONDARY_FILES` (or the repo's number of gold files, if smaller) in a repo are not picked. Check the result end to end with `tmp/eval_collector_gold.py`:

```bash
python -m model.tmp.tune_xgb_gold --data ./model/tmp/colab/llm_training_data_qwen2.5-7b-instruct-awq_v9.jsonl
python -m model.tmp.eval_collector_gold --model llm_scorer_v8.pkl
```

The holdout repos (`HOLDOUT` in `tmp/tune_xgb_gold.py`) are never trained on and take no part in the search; both scripts skip them unless `--report-holdout` is given. Pass it once, after tuning is done, and report that number. The production model can then be retrained on every repo.

| | gold PR-AUC (leave-one-repo-out, llama.cpp + milvus) | collector core recall | collector hit rate |
|---|---|---|---|
| v6: old params, labels before current policies | 0.25 | 0.32 | 0.40 |
| v8: tuned params, v9 labels | 0.36 | 0.37 | 0.49 |

The model uses only the 9 columns in `FEATURE_COLUMNS`: `rule_score`, `depth`, `src_depth`, `ext_group_other`, `in_examples`, `is_example_file`, `is_dev_config`, `is_internal_name`, `is_public_api_name`. `extract_features` still computes the other path flags, because `rule_score` and `collector.py` use them; import reference counts moved to `ref_counts.py`. Trained on one repo and tested on the other, the dropped flags either overfit to the training repos (`in_src`, the `in_*` directory flags, `ext_group_code`, …) or added nothing beyond `rule_score` (`ref_count`, `is_readme`, `is_public_api`, …). Removing them raised PR-AUC from 0.49 to 0.51 on llama.cpp and from 0.27 to 0.33 on milvus.

`FileScorer.load` rejects a model, and `load_jsonl` skips a training record, whose feature layout differs from `FEATURE_COLUMNS`. Changing `FEATURE_COLUMNS` requires retraining.

# Step 3: Inference (integrated with git ls-files)

```bash
git -C source/llama.cpp ls-files | grep docs | python -m model.train_ml_scorer score --model llm_scorer_v8.pkl --top-k 15
```

# Step 4: collector.py

`collector.py` loads `SCORER_MODEL` and falls back to `rule_score` when it is missing or has a different feature layout.

With the neural scorer (`.pt`, the default), `FileSelector.secondary` keeps its `MAX_SECONDARY_FILES` (20) best-scored files, skipping priority files and files below the scorer's floor (0.05); see `nn/README.md` for why the steps below are skipped for it.

With a `FileScorer` pickle (e.g. `SCORER_MODEL = "llm_scorer_v8.pkl"`), `FileSelector.secondary` instead:
1. computes `ref_count` (non-test importers) and example reference counts (importers under `examples/` or Go `*_example_test.go`) with `build_ref_counts`;
2. scores every file and keeps those `>= SECONDARY_THRESHOLD` (0.5);
3. ranks with tie-breakers: model score → `rule_score` → content signal (code blocks in docs, stub files last) → depth → `ref_count`;
4. reserves code slots: `MIN_API_CODE_FILES` (3) public-API files ranked by example references, then `MIN_EXAMPLE_CODE_FILES` (2) example programs;
5. fills `MAX_SECONDARY_FILES` (20), with caps of `MAX_PER_PARENT_DIR` (3) per directory and `MAX_PER_TOP_DIR` (5) per top-level directory, skipping files already collected as priority files.

```py
from model.train_ml_scorer import FileScorer
from model.ref_counts import build_ref_counts

scorer = FileScorer.load("llm_scorer_v8.pkl")
ranked = scorer.rank(all_files)   # [(score, path), ...] descending

# Tie-breakers used by collect_secondary_files
example_refs = {}
ref_counts = build_ref_counts(repo_root, all_files, example_counts=example_refs)
```
