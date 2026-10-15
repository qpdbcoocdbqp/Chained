#!/usr/bin/env python3
"""
generate_training_data.py — Build labeled training data for the ML file scorer.

For each repository under --root, this script:
  1. Lists all files (respecting gitignore via `git ls-files`, or plain walk).
  2. Extracts rule-based features and scores for every file.
  3. Labels obvious negatives (tests, vendor, binaries, lock files, CI) without
     the LLM ("auto"), then sends the remaining files — with a content snippet
     and repo context — in batches to the local LLM ("llm").  Labels are cached
     per prompt version, so feature changes do not require relabeling.
  4. Writes one JSONL record per file:
       {"path": "src/api.py", "label": 1, "source": "llm", "features": {...}, ...}

The JSONL output is then consumed by train_ml_scorer.py; train on
`--sources llm,auto` — "rule" labels are derived from the rule_score feature.

Usage:
    python3 generate_training_data.py \\
        --root /path/to/repos \\
        --output training_data.jsonl \\
        [--batch-size 20] \\
        [--max-llm-files 600]  # per-repo cap on LLM-labeled files
        [--no-content]         # send paths only
        [--rule-only]          # skip LLM, label using rule_score threshold instead

Environment variables:
    LOCAL_LLM_BASE_URL   e.g. http://localhost:1234/v1
    LOCAL_LLM_MODEL      e.g. qwen2.5-coder-7b-instruct
    LOCAL_LLM_API_KEY    e.g. lm-studio
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from filescore import SKIP_EXTS, SKIP_NAMES
from model.file_features import extract_features, _IGNORE_DIRS, _TEST_PATTERN, build_ref_counts

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

# Bump whenever _SYSTEM_PROMPT, CATEGORIES or the per-file context changes
# meaningfully: cached labels are only reused for the same version.
PROMPT_VERSION = "v9"

# The LLM only classifies each file into one of these categories; the label is
# derived here.  Asking for 0/1 directly let the model trade an include rule
# off against an exclude rule ("doc.go is an entry point" vs "internal support
# library") and pick 1 — a fixed category → label table enforces the policy.
CATEGORIES: dict[str, tuple[int, str]] = {
    # --- label 1 ---------------------------------------------------------
    "user_doc":         (1, "README, getting started, install / build instructions, user guide, "
                            "tutorial, FAQ, troubleshooting, docs of CLI tools / server endpoints / "
                            "config options, guide for migrating user code between versions"),
    "example_main":     (1, "the main program or README of an example / demo / sample notebook, "
                            "Go *_example_test.go"),
    "api_entry":        (1, "entry point of the PUBLIC library / SDK users import: headers in the "
                            "library's public include/ dir, __init__.py / doc.go / index.ts of the "
                            "public package, the SDK client constructor and its main operations "
                            "files (collections, search, insert ...)"),
    "user_script":      (1, "a script users run directly, located at the repo root "
                            "(e.g. convert_hf_to_gguf.py)"),
    "install_manifest": (1, "ROOT-level pyproject.toml / package.json / go.mod / requirements.txt, "
                            "or a documented example configuration file"),
    # --- label 0 ---------------------------------------------------------
    "per_variant_impl": (0, "one of many sibling files each implementing one model / backend / "
                            "index type / data type / operator (conversion/llama.py, index/hnsw.go, "
                            "src/models/qwen.cpp, ggml-cuda/*.cu)"),
    "sdk_internal":     (0, "SDK file that is not its main entry: option builders (*_options), "
                            "type / entity / column definitions, conversion, serialization, minor "
                            "or administrative operations"),
    "internal_impl":    (0, "implementation: backends, kernels, server internals, internal/ pkg/ "
                            "core/ packages of a server, helpers, headers next to the implementation "
                            "(src/*.h) — even when widely imported"),
    "tool_impl":        (0, "implementation source of a CLI tool or server (tools/x/x.cpp)"),
    "ui_source":        (0, "web / mobile / desktop app source: components, pages, stores, "
                            "services, utils, type definitions"),
    "example_aux":      (0, "anything in an example other than its main program / README: build "
                            "and project files, resources, helper headers, secondary source files"),
    "maintainer_doc":   (0, "design docs, architecture / data-flow docs, RFCs, ADRs, contributor or "
                            "developer guides, guides for coding agents, release process"),
    "ops_runbook":      (0, "operating a deployment: cluster upgrade, metadata / storage migration, "
                            "log export, monitoring / dashboards setup"),
    "support_lib":      (0, "code or docs of an internal support library (logging, errors, metrics, "
                            "tracing) of a server, including its doc.go / README"),
    "test_build_ci":    (0, "tests, mocks, fixtures, benchmarks, CI/CD, build scripts, linters, "
                            "nested manifests — but NOT Go *_example_test.go, which is example_main"),
    "asset_generated":  (0, "generated, vendored, binary, data, asset or template files "
                            "(e.g. chat-template .jinja), placeholders, near-empty files"),
}

_SYSTEM_PROMPT = """
You classify files of a software repository for building a USER CHEATSHEET: a
compact reference telling a new user what the project does, how to install
it, and how to use it (commands, APIs, configuration).

Put every file into exactly ONE category:

{categories}

Rules:
- When a file fits both an include category (user_doc, example_main,
  api_entry, user_script, install_manifest) and an exclude category, the
  EXCLUDE category wins.  Examples: doc.go of a server's logging package is
  support_lib; *_options.go of an SDK is sdk_internal; a grafana setup README
  is ops_runbook; a UI data-flow doc is maintainer_doc.
- Use the facts after each path: "repo root" vs a nested path decides
  user_script; "1 of N <ext> files in <dir>/" with a large N usually means
  per_variant_impl.
- First decide what kind of project this is from the repo context.  For a
  LIBRARY its exported entry points are api_entry.  For a SERVER or
  APPLICATION, users go through its client SDK, CLI, configuration and network
  API; the server's own packages are internal_impl or support_lib.
- When a content snippet is given, judge by the content; the path alone can
  mislead (a docs/ file may be a design record, a README may be a stub).
- Judge every file independently, not by analogy with its neighbours.
- Expect only about 5-15% of files in include categories.  When unsure
  between an include and an exclude category, choose the exclude one.

Respond with ONLY a JSON array with one object per input item, in input order:
[{{"i": <item number>, "category": "<category>", "reason": "<at most 15 words>"}}]
""".format(categories="\n".join(f"- {name} — {desc}" for name, (_, desc) in CATEGORIES.items()))

SNIPPET_CHARS = 400          # per-file content excerpt sent to the LLM
README_CONTEXT_CHARS = 1500  # repo README excerpt included in every batch
_TEXT_SNIFF_BYTES = 2048
_DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".adoc"}
_COMMENT_LINE = re.compile(r"^\s*(//|#|/\*|\*|\*/|--|;)")
_LICENSE_WORDS = re.compile(r"licen[cs]e|copyright|spdx", re.I)


def _strip_license_header(text: str) -> str:
    """Drop a leading comment block that is a license header — otherwise the
    Apache boilerplate fills the whole snippet of most source files.

    Only the first block, up to the first blank line, is considered: the
    package doc comment that usually follows (// Package milvusclient ...)
    is exactly what the labeler needs to see.  Call before blank lines are
    collapsed."""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and lines[k].strip() and _COMMENT_LINE.match(lines[k]):
        k += 1
    if k and _LICENSE_WORDS.search("\n".join(lines[:k])):
        return "\n".join(lines[k:])
    return text


def _read_snippet(repo_dir: Path, rel: str, limit: int) -> str | None:
    """First `limit` characters of a text file with blank runs collapsed, or
    None for binary / unreadable files."""
    p = repo_dir / rel
    try:
        with open(p, "rb") as f:
            head = f.read(_TEXT_SNIFF_BYTES)
            if b"\x00" in head:
                return None
            data = head + f.read(max(0, limit * 4, 8192 - len(head)))
    except OSError:
        return None
    text = data.decode("utf-8", errors="ignore").replace("\r\n", "\n").lstrip()
    if p.suffix.lower() not in _DOC_SUFFIXES:   # '#' is a heading in Markdown
        text = _strip_license_header(text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:limit]


def _repo_context(repo_dir: Path, files: list[str]) -> str:
    """Repo-level context shared by every batch: name, top-level layout and
    the start of the README, so the LLM knows what kind of project it is."""
    top = sorted({f.replace("\\", "/").split("/")[0] + ("/" if "/" in f.replace("\\", "/") else "")
                  for f in files})
    readme = next((f for f in files if f.lower() in ("readme.md", "readme.rst", "readme.txt", "readme")), None)
    parts = [f"Repository: {repo_dir.name}",
             "Top-level entries: " + ", ".join(top[:60])]
    if readme:
        excerpt = _read_snippet(repo_dir, readme, README_CONTEXT_CHARS)
        if excerpt:
            parts.append(f"README excerpt:\n{excerpt}")
    return "\n".join(parts)


def dir_ext_stats(all_files: list[str]) -> dict[str, dict[str, int]]:
    """dir → {extension: file count}, for the per-file location facts.
    Test files are not counted: 93 .go files in milvusclient/ are mostly
    *_test.go and would make the SDK entry look like one variant of many."""
    stats: dict[str, dict[str, int]] = {}
    for f in all_files:
        d, _, name = f.replace("\\", "/").rpartition("/")
        if _TEST_PATTERN.search(Path(name).stem.lower()):
            continue
        ext = Path(name).suffix.lower()
        stats.setdefault(d, {})
        stats[d][ext] = stats[d].get(ext, 0) + 1
    return stats


_MAINTAINER_DOC_PATH = re.compile(
    r"(^|/)docs?/(.+/)?(design[-_]?docs?|designs?|architecture|flows|rfcs?|adrs?|"
    r"development|developer[-_]guides?|internals)/", re.I)


def _location_fact(rel: str, dir_stats: dict[str, dict[str, int]] | None) -> str:
    """Objective placement facts the model otherwise guesses wrong: whether a
    script is at the repo root, and how many same-type siblings it has
    (one of 45 .py files in conversion/ is a per-variant implementation)."""
    d, _, name = rel.replace("\\", "/").rpartition("/")
    if not d:
        return "repo root"
    ext = Path(name).suffix.lower()
    n = (dir_stats or {}).get(d, {}).get(ext, 0)
    fact = f"1 of {n} {ext or 'extensionless'} files in {d}/" if n > 1 else f"in {d}/"
    m = _MAINTAINER_DOC_PATH.search(rel.replace("\\", "/"))
    if m:
        fact += f"; inside a '{m.group(3)}' docs directory, which is written for maintainers"
    return fact


def _format_batch(repo_dir: Path, batch: list[str], with_content: bool,
                  dir_stats: dict[str, dict[str, int]] | None = None) -> str:
    lines = []
    for i, rel in enumerate(batch, 1):
        try:
            size_kb = (repo_dir / rel).stat().st_size / 1024
            size = f"{size_kb:.1f} KB"
        except OSError:
            size = "?"
        lines.append(f"[{i}] {rel} ({size}; {_location_fact(rel, dir_stats)})")
        if with_content:
            snippet = _read_snippet(repo_dir, rel, SNIPPET_CHARS)
            if snippet is None:
                lines.append("    <binary or unreadable>")
            elif not snippet:
                lines.append("    <empty>")
            else:
                lines.append("    " + snippet.replace("\n", "\n    "))
    return "\n".join(lines)


_ITEM_PATTERN = re.compile(r'"i"\s*:\s*"?(\d+)"?\s*,\s*"category"\s*:\s*"([a-z_]+)"')
LLM_DEBUG_LOG = Path("model/tmp/llm_failed_replies.log")


def _log_bad_reply(raw: str, error: str) -> None:
    try:
        LLM_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"===== {error}\n{raw}\n")
    except OSError:
        pass


def _extract_json_array(raw: str):
    """Parse the model's reply, tolerating <think> blocks, markdown fences and
    chatter around the JSON array.  If the array itself is malformed (most
    often an unescaped quote inside a "reason"), fall back to pulling the
    i/category pairs out with a regex — those matter, the reasons do not."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            results = json.loads(raw[start:end + 1])
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            pass
    salvaged = [{"i": int(i), "category": cat, "reason": "(reason lost: malformed JSON)"}
                for i, cat in _ITEM_PATTERN.findall(raw)]
    if salvaged:
        _log_bad_reply(raw, f"salvaged {len(salvaged)} items")
        return salvaged
    _log_bad_reply(raw, "unparseable")
    first = raw.strip().splitlines()[0][:200] if raw.strip() else "<empty reply>"
    raise RuntimeError(f"LLM reply has no parseable labels: {first}")


def _call_llm(context: str, batch_text: str, base_url: str, model: str, api_key: str) -> list:
    """Send one batch to an OpenAI-compatible chat endpoint and parse the reply."""
    user_msg = f"{context}\n\nFiles to classify:\n{batch_text}"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,  # deterministic output for reproducible labels
        "max_tokens": 8192,
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    return _extract_json_array(payload["choices"][0]["message"]["content"])


_EXAMPLE_README = re.compile(r"(^|/)(examples?|samples?|demos?)/[^/]+/readme[^/]*$", re.I)


def _policy_override(rel: str, category: str) -> str:
    """Policies decidable from the path alone, which the model applies
    inconsistently across batches.  Content judgements stay with the model."""
    norm = rel.replace("\\", "/")
    # godoc runnable examples look like tests but are usage documentation
    if category == "test_build_ci" and norm.lower().endswith("example_test.go"):
        return "example_main"
    # the README of examples/<name>/ is that example's main document
    if category == "example_aux" and _EXAMPLE_README.search(norm):
        return "example_main"
    return category


def _label_batch(repo_dir: Path, context: str, batch: list[str], with_content: bool,
                 base_url: str, model: str, api_key: str,
                 dir_stats: dict[str, dict[str, int]] | None = None) -> dict[str, dict]:
    """Label one batch.  Items are referenced by number rather than path so
    that a model rewriting a path (slashes, ./ prefix) cannot drop a label.
    The model returns a category; the label comes from CATEGORIES.
    On failure the batch is split in half and retried, down to single files."""
    def again(sub):
        return _label_batch(repo_dir, context, sub, with_content, base_url, model, api_key, dir_stats)

    try:
        items = _call_llm(context, _format_batch(repo_dir, batch, with_content, dir_stats),
                          base_url, model, api_key)
    except RuntimeError as e:
        if len(batch) == 1:
            print(f"    ⚠️ giving up on {batch[0]}: {e}", file=sys.stderr)
            return {}
        print(f"    ⚠️ batch of {len(batch)} failed ({str(e).splitlines()[0]}); splitting",
              file=sys.stderr)
        mid = len(batch) // 2
        return {**again(batch[:mid]), **again(batch[mid:])}

    labels = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        category = str(item.get("category", "")).strip().lower()
        if 1 <= idx <= len(batch) and category in CATEGORIES:
            category = _policy_override(batch[idx - 1], category)
            labels[batch[idx - 1]] = {"label": CATEGORIES[category][0], "category": category,
                                      "reason": str(item.get("reason", ""))}
    missing = [f for f in batch if f not in labels]
    if missing and len(missing) < len(batch):
        # Model skipped some items or invented a category — ask again for those
        labels.update(again(missing))
    return labels


VERIFY_MIN_FILLER = 4   # negatives mixed into a verification batch


def label_and_verify(repo_dir: Path, context: str, batch: list[str], with_content: bool,
                     base_url: str, model: str, api_key: str,
                     dir_stats: dict[str, dict[str, int]] | None, rng: random.Random,
                     verify: bool = True) -> dict[str, dict]:
    """Label a batch, then re-label its positives once more in a different mix
    and keep label 1 only when both passes agree.

    Errors are almost all false positives that flip with batch composition
    (a maintainer guide called user_doc next to real user docs), so a second,
    differently-composed look at the ~10% positives buys precision cheaply.
    Negatives from the batch are mixed in as filler so the verifier does not
    see a list made only of plausible files."""
    labels = _label_batch(repo_dir, context, batch, with_content, base_url, model, api_key, dir_stats)
    positives = [f for f, lab in labels.items() if lab["label"] == 1]
    if not verify or not positives:
        return labels
    negatives = [f for f, lab in labels.items() if lab["label"] == 0]
    filler = rng.sample(negatives, min(len(negatives), max(VERIFY_MIN_FILLER, len(positives))))
    check = positives + filler
    rng.shuffle(check)
    second = _label_batch(repo_dir, context, check, with_content, base_url, model, api_key, dir_stats)
    for f in positives:
        lab2 = second.get(f)
        if lab2 is not None and lab2["label"] == 0:
            first_cat = labels[f]["category"]
            labels[f] = {**lab2, "reason": f"[verify: {first_cat} -> {lab2['category']}] {lab2['reason']}"}
    return labels


# ---------------------------------------------------------------------------
# Label cache — labels depend on the prompt and model, not on file_features,
# so feature changes can be retrained without relabeling.
# ---------------------------------------------------------------------------

def _cache_key(repo: str, path: str, model: str, with_content: bool, verify: bool) -> str:
    return "\t".join([repo, path.replace("\\", "/"), PROMPT_VERSION, model,
                      str(int(with_content)), "verify" if verify else "single"])


def _load_cache(cache_path: str | None) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_path or not Path(cache_path).exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                cache[r["key"]] = {"label": r["label"], "category": r.get("category", ""),
                                   "reason": r.get("reason", "")}
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _label_via_llm(
    repo_dir: Path, files: list[str], all_files: list[str], base_url: str, model: str,
    api_key: str, batch_size: int, with_content: bool, cache: dict[str, dict],
    cache_path: str | None, verify: bool = True,
) -> dict[str, dict]:
    """Label `files` with the LLM, reusing and extending the on-disk cache.
    Returns path → {label, reason}; files the LLM never labeled are absent."""
    repo = repo_dir.name
    results = {}
    todo = []
    for f in files:
        hit = cache.get(_cache_key(repo, f, model, with_content, verify))
        if hit:
            results[f] = hit
        else:
            todo.append(f)
    print(f"   LLM labels: {len(results)} cached, {len(todo)} to label", flush=True)
    if not todo:
        return results

    # Mixed batches: path-sorted batches put 20 sibling files together and
    # the model labels them by analogy instead of one by one.
    rng = random.Random(f"{repo}:{PROMPT_VERSION}")
    rng.shuffle(todo)

    context = _repo_context(repo_dir, all_files)
    dir_stats = dir_ext_stats(all_files)
    cache_f = open(cache_path, "a", encoding="utf-8") if cache_path else None
    try:
        n_batches = (len(todo) + batch_size - 1) // batch_size
        for b, i in enumerate(range(0, len(todo), batch_size), 1):
            batch = todo[i:i + batch_size]
            print(f"   batch {b}/{n_batches} ({len(batch)} files)...", flush=True)
            labels = label_and_verify(repo_dir, context, batch, with_content, base_url, model,
                                      api_key, dir_stats, rng, verify)
            results.update(labels)
            if cache_f:
                for path, lab in labels.items():
                    key = _cache_key(repo, path, model, with_content, verify)
                    cache[key] = lab
                    cache_f.write(json.dumps({"key": key, **lab}, ensure_ascii=False) + "\n")
                cache_f.flush()   # keep progress if the run is interrupted
    finally:
        if cache_f:
            cache_f.close()
    return results


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

# Asset / IDE-project extensions not covered by filescore.SKIP_EXTS
_ASSET_EXTS = {".webp", ".avif", ".tiff", ".icns", ".ttf", ".otf", ".woff", ".woff2",
               ".eot", ".mp4", ".webm", ".ogg", ".flac", ".pbxproj", ".xcworkspacedata",
               ".xcscheme", ".plist", ".storyboard", ".xib", ".jar", ".keystore"}
# Vendored dependency trees, often nested inside examples/
_VENDORED_DIRS = {"third_party", "third-party", "thirdparty", "3rdparty", "deps", "external_deps"}
# Generated sources: protobuf / gRPC stubs, codegen output
_GENERATED_NAME = re.compile(
    r"(\.pb\.(go|h|cc|cpp|gw\.go)|_pb2(_grpc)?\.py|\.pb\.validate\.go|_grpc\.pb\.go|"
    r"[._-]gen(erated)?\.(go|py|ts|js|h|cpp|rs)|\.g\.dart|_string\.go)$", re.I)
# "Code generated ... DO NOT EDIT." (Go convention) and common equivalents.
# Only checked on the leading comment lines: generator *programs* mention
# "autogenerated" in their string literals.
_GENERATED_MARKER = re.compile(
    r"do not (edit|modify)|@generated|"
    r"this (file|code) (is|was|has been) (automatically |auto-?)?generated|"
    r"(automatically|auto-?)generated (by|from|file|code)", re.I)


_PREPROCESSOR = re.compile(r"^\s*#\s*(include|import|pragma|define|if|ifdef|ifndef|else|endif)\b")


def _has_generated_header(head: str) -> bool:
    for line in head.splitlines()[:15]:
        stripped = line.strip()
        if not stripped:
            continue
        if not _COMMENT_LINE.match(line) or _PREPROCESSOR.match(line):
            break                       # first non-comment line ends the header
        if _GENERATED_MARKER.search(stripped):
            return True
    return False
# Android resource dirs (res/values, res/mipmap-*, res/layout, ...)
_ANDROID_RES = re.compile(r"(^|/)res/(values|mipmap|drawable|layout|xml)[^/]*/", re.I)


def _auto_negative(rel: str, feats: dict, repo_dir: Path | None = None) -> str | None:
    """Reason string when a file is certainly not cheatsheet material, so it
    can be labeled without spending an LLM call; None otherwise."""
    p = Path(rel)
    name, ext = p.name.lower(), p.suffix.lower()
    norm = rel.replace("\\", "/")
    if feats["in_ignored"]:
        return "vendored / generated / build-output path"
    if any(part.lower() in _VENDORED_DIRS for part in p.parts[:-1]):
        return "vendored dependency"
    if name in SKIP_NAMES:
        return "build, lint or lock file"
    if ext in SKIP_EXTS or ext in _ASSET_EXTS or feats["ext_group_data"]:
        return "binary, asset or data file"
    if _ANDROID_RES.search(norm):
        return "mobile app resource file"
    if repo_dir is not None and _read_snippet(repo_dir, rel, 1) is None:
        return "binary file"
    if feats["in_devops"]:
        return "CI / devops path"
    if (feats["in_tests"] or feats["is_test_file"]) and not feats["is_example_file"]:
        return "test code"
    if re.search(r"(^|[_\-.])mocks?([_\-.]|$)", p.stem.lower()):
        return "test mock"
    if _GENERATED_NAME.search(name):
        return "generated code"
    if repo_dir is not None and feats["ext_group_code"]:
        try:
            with open(repo_dir / rel, "rb") as f:
                head = f.read(1024).decode("utf-8", errors="ignore")
        except OSError:
            head = ""
        if _has_generated_header(head):
            return "generated code"
    return None


def _sample_for_llm(files: list[str], features_map: dict, limit: int | None, seed: int) -> list[str]:
    """Cap the number of LLM-labeled files per repo.  Half the budget goes to
    files the rules consider plausible (rule_score >= 0.3), where labels are
    most informative; the rest is a random sample of everything else so the
    model also learns where the rules are wrong in the other direction."""
    if limit is None or len(files) <= limit:
        return files
    rng = random.Random(seed)
    hi = [f for f in files if features_map[f]["rule_score"] >= 0.3]
    lo = [f for f in files if features_map[f]["rule_score"] < 0.3]
    rng.shuffle(hi)
    rng.shuffle(lo)
    take_hi = hi[: max(limit // 2, limit - len(lo))]
    take_lo = lo[: limit - len(take_hi)]
    return sorted(take_hi + take_lo)


def _rule_label(rule_score: float, threshold: float = 0.45) -> int:
    """Convert a rule score to a binary label using a threshold."""
    return 1 if rule_score >= threshold else 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(root_dir: str, output_path: str, batch_size: int, rule_only: bool,
             with_content: bool = True, max_llm_files: int | None = None,
             cache_path: str | None = None, seed: int = 42, verify: bool = True):
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
    cache = _load_cache(cache_path) if use_llm else {}

    total_records = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for repo_dir in repo_dirs:
            repo_name = repo_dir.name
            print(f"\n📂 Repo: {repo_name}")
            files = list_repo_files(repo_dir)
            print(f"   {len(files)} files found")
            if not files:
                continue

            # Extract features for all files (with cross-file reference counts)
            print(f"   Building reference counts...")
            ref_counts = build_ref_counts(str(repo_dir), files)
            features_map = {f: extract_features(f, ref_count=ref_counts.get(f, 0)) for f in files}

            auto_neg: dict[str, str] = {}
            llm_labels: dict[str, dict] = {}
            if use_llm:
                auto_neg = {f: r for f in files if (r := _auto_negative(f, features_map[f], repo_dir))}
                candidates = _sample_for_llm([f for f in files if f not in auto_neg],
                                             features_map, max_llm_files, seed)
                print(f"   {len(auto_neg)} auto-negative, {len(candidates)} sent to LLM")
                llm_labels = _label_via_llm(repo_dir, candidates, files, base_url, model, api_key,
                                            batch_size, with_content, cache, cache_path, verify)

            # Write one JSONL record per file
            for f in files:
                feats = features_map[f]
                rule_sc = feats["rule_score"]

                category = ""
                if f in llm_labels:
                    label    = llm_labels[f]["label"]
                    reason   = llm_labels[f]["reason"]
                    category = llm_labels[f].get("category", "")
                    source   = "llm"
                elif f in auto_neg:
                    label  = 0
                    reason = auto_neg[f]
                    source = "auto"
                else:
                    # Train on LLM data with `--sources llm,auto`: rule labels
                    # are a function of the rule_score feature (circular).
                    label  = _rule_label(rule_sc)
                    reason = f"rule_score={rule_sc:.3f}"
                    source = "rule"

                record = {
                    "repo":       repo_name,
                    "path":       f,
                    "label":      label,
                    "source":     source,    # "llm", "auto" or "rule"
                    "reason":     reason,
                    "category":   category,  # LLM category the label was derived from
                    "rule_label": _rule_label(rule_sc),
                    "features":   feats,
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

    # Where the LLM disagrees with the rules is exactly what the ML model can
    # learn beyond rule_score — and a quick sanity check of the labels.
    agree = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    flips: dict[tuple, list] = {(0, 1): [], (1, 0): []}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "llm" or "rule_label" not in r:
                continue
            key = (r["rule_label"], r["label"])
            agree[key] += 1
            if key in flips:
                flips[key].append(r["path"])
    n = sum(agree.values())
    if n:
        print(f"\n🤝 LLM vs rule on {n} LLM-labeled files: "
              f"agree={(agree[(0, 0)] + agree[(1, 1)]) / n * 100:.1f}%  "
              f"rule0→llm1={agree[(0, 1)]}  rule1→llm0={agree[(1, 0)]}")
        for (rl, ll), paths in flips.items():
            if paths:
                print(f"   e.g. rule={rl} llm={ll}: " + ", ".join(paths[:5]))


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
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of files sent to the LLM per request")
    parser.add_argument("--rule-only",  action="store_true",
                        help="Skip LLM; label using rule score threshold only")
    parser.add_argument("--no-content", action="store_true",
                        help="Send paths only, without per-file content snippets")
    parser.add_argument("--max-llm-files", type=int, default=None,
                        help="Per-repo cap on files sent to the LLM (stratified sample)")
    parser.add_argument("--cache", default="model/tmp/llm_label_cache.jsonl",
                        help="LLM label cache (reused across runs); '' to disable")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the second LLM pass that re-checks label-1 files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --max-llm-files sampling")
    args = parser.parse_args()
    generate(args.root, args.output, args.batch_size, args.rule_only,
             with_content=not args.no_content, max_llm_files=args.max_llm_files,
             cache_path=args.cache or None, seed=args.seed, verify=not args.no_verify)


if __name__ == "__main__":
    main()
