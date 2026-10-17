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
  4. Writes one JSONL record per labeled file:
       {"path": "src/api.py", "label": 1, "source": "llm", "features": {...}, ...}

The prompt and the category → label table live in prompt.yaml.  The JSONL
output is consumed by train_ml_scorer.py.

Usage:
    python3 generate_training_data.py \\
        --root /path/to/repos \\
        --output training_data.jsonl \\
        [--batch-size 20] [--workers 8]

Environment variables (required):
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv
load_dotenv()

from filescore import SKIP_EXTS, SKIP_NAMES
from model.file_features import extract_features, features

PROMPT = yaml.safe_load(Path(__file__).with_name("prompt.yaml").read_text(encoding="utf-8"))
PROMPT_VERSION: str = PROMPT["version"]
CATEGORIES: dict[str, int] = {name: c["label"] for name, c in PROMPT["categories"].items()}
SYSTEM_PROMPT: str = PROMPT["system"].replace("{categories}", "\n".join(
    f"- {name} — {c['desc']}" for name, c in PROMPT["categories"].items()))


def list_repo_files(repo_root: Path) -> list[str]:
    """
    Return relative file paths inside a repo. Prefers `git ls-files` (which
    respects .gitignore automatically); falls back to a recursive walk that
    manually skips common noise directories.
    """
    if (repo_root / ".git").exists():
        try:
            out = subprocess.check_output(["git", "ls-files"], cwd=repo_root,
                                          stderr=subprocess.DEVNULL, text=True)
            files = [l.strip() for l in out.splitlines() if l.strip()]
            if files:
                return files
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # git not available or not a proper repo, fall through

    return sorted(str(p.relative_to(repo_root)) for p in repo_root.rglob("*")
                  if p.is_file() and not features.dirs["ignored"]
                  & {part.lower() for part in p.relative_to(repo_root).parts})


# ---------------------------------------------------------------------------
# Content snippets
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM labeling
# ---------------------------------------------------------------------------

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


def llm_from_env() -> tuple[openai.OpenAI, str]:
    """Client for the OpenAI-compatible endpoint (vLLM, LM Studio, ...) in
    LOCAL_LLM_BASE_URL, and the model name."""
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
    if not base_url:
        sys.exit("❌ LOCAL_LLM_BASE_URL is not set")
    client = openai.OpenAI(base_url=base_url, api_key=os.environ.get("LOCAL_LLM_API_KEY", "not-needed"),
                           timeout=300)
    return client, os.environ.get("LOCAL_LLM_MODEL", "gpt-3.5-turbo")


_MAINTAINER_DOC_PATH = re.compile(
    r"(^|/)docs?/(.+/)?(design[-_]?docs?|designs?|architecture|flows|rfcs?|adrs?|"
    r"development|developer[-_]guides?|internals)/", re.I)
_EXAMPLE_README = re.compile(r"(^|/)(examples?|samples?|demos?)/[^/]+/readme[^/]*$", re.I)
_ROOT_COMPOSE = re.compile(r"^docker-compose[\w.-]*\.ya?ml$", re.I)
_CONFIG_EXAMPLE = re.compile(r"(^|/)configs?/[^/]+\.(ya?ml|toml|json|ini|conf)$", re.I)
VERIFY_MIN_FILLER = 4   # negatives mixed into a verification batch
VOTE_MIN_FILES = 10     # directory size for the per_variant_impl majority vote


def policy_override(rel: str, category: str) -> str:
    """Policies decidable from the path alone, which the model applies
    inconsistently across batches.  Content judgements stay with the model."""
    norm = rel.replace("\\", "/")
    name = norm.rpartition("/")[2].lower()
    # godoc runnable examples look like tests but are usage documentation
    if category == "test_build_ci" and name.endswith("example_test.go"):
        return "example_main"
    # the README of examples/<name>/ is that example's main document
    if _EXAMPLE_README.search(norm):
        return "example_main"
    # protobuf schemas are a server's internal RPC definitions; users go
    # through the client SDK (the model called 15 of milvus pkg/proto api_entry)
    if name.endswith(".proto") and CATEGORIES[category]:
        return "internal_impl"
    # install_manifest as prompt.yaml defines it: a root-level manifest or an
    # example configuration under configs/.  The model also gave it to
    # .devops/ Dockerfiles, nix files, nested requirements and cmake templates.
    if category == "install_manifest" and not (
            ("/" not in norm and (name in features.names["user_config"] or _ROOT_COMPOSE.match(name)))
            or _CONFIG_EXAMPLE.search(norm)):
        return "user_doc" if name.startswith("readme") else "test_build_ci"
    return category


class RepoLabeler:
    """Labels files of one repo with the LLM.

    Every batch carries the same repo context (name, top-level layout, README
    excerpt) so the LLM knows what kind of project it is, and per-file
    location facts computed from all files of the repo.
    """

    def __init__(self, repo_dir: Path, all_files: list[str], client: openai.OpenAI, model: str,
                 verify: bool = True):
        self.repo_dir, self.client, self.model, self.verify = repo_dir, client, model, verify
        self.context = self._repo_context(all_files)
        self.dir_stats = self._dir_ext_stats(all_files)

    def _repo_context(self, files: list[str]) -> str:
        top = sorted({f.replace("\\", "/").split("/")[0] + ("/" if "/" in f.replace("\\", "/") else "")
                      for f in files})
        readme = next((f for f in files if f.lower() in ("readme.md", "readme.rst", "readme.txt", "readme")), None)
        parts = [f"Repository: {self.repo_dir.name}", "Top-level entries: " + ", ".join(top[:60])]
        excerpt = readme and _read_snippet(self.repo_dir, readme, README_CONTEXT_CHARS)
        if excerpt:
            parts.append(f"README excerpt:\n{excerpt}")
        return "\n".join(parts)

    @staticmethod
    def _dir_ext_stats(files: list[str]) -> dict[str, Counter]:
        """dir → {extension: file count}, for the per-file location facts.
        Test files are not counted: 93 .go files in milvusclient/ are mostly
        *_test.go and would make the SDK entry look like one variant of many."""
        stats: dict[str, Counter] = {}
        for f in files:
            d, _, name = f.replace("\\", "/").rpartition("/")
            if not features.patterns["test"].search(Path(name).stem.lower()):
                stats.setdefault(d, Counter())[Path(name).suffix.lower()] += 1
        return stats

    def location_fact(self, rel: str) -> str:
        """Objective placement facts the model otherwise guesses wrong: whether a
        script is at the repo root, and how many same-type siblings it has
        (one of 45 .py files in conversion/ is a per-variant implementation)."""
        d, _, name = rel.replace("\\", "/").rpartition("/")
        if not d:
            return "repo root"
        ext = Path(name).suffix.lower()
        n = self.dir_stats.get(d, {}).get(ext, 0)
        fact = f"1 of {n} {ext or 'extensionless'} files in {d}/" if n > 1 else f"in {d}/"
        m = _MAINTAINER_DOC_PATH.search(rel.replace("\\", "/"))
        if m:
            fact += f"; inside a '{m.group(3)}' docs directory, which is written for maintainers"
        return fact

    def format_batch(self, batch: list[str]) -> str:
        lines = []
        for i, rel in enumerate(batch, 1):
            try:
                size = f"{(self.repo_dir / rel).stat().st_size / 1024:.1f} KB"
            except OSError:
                size = "?"
            lines.append(f"[{i}] {rel} ({size}; {self.location_fact(rel)})")
            snippet = _read_snippet(self.repo_dir, rel, SNIPPET_CHARS)
            lines.append("    <binary or unreadable>" if snippet is None else
                         "    <empty>" if not snippet else
                         "    " + snippet.replace("\n", "\n    "))
        return "\n".join(lines)

    def classify(self, batch: list[str]) -> list:
        """Send one batch with the system prompt and parse the reply."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT["user"].format(
                        context=self.context, files=self.format_batch(batch))},
                ],
                **PROMPT["request"],
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f"LLM request failed: {e}") from e
        return _extract_json_array(resp.choices[0].message.content or "")

    def label_batch(self, batch: list[str]) -> dict[str, dict]:
        """Label one batch.  Items are referenced by number rather than path so
        that a model rewriting a path (slashes, ./ prefix) cannot drop a label.
        The model returns a category; the label comes from CATEGORIES.
        On failure the batch is split in half and retried, down to single files."""
        try:
            items = self.classify(batch)
        except RuntimeError as e:
            if len(batch) == 1:
                print(f"    ⚠️ giving up on {batch[0]}: {e}", file=sys.stderr)
                return {}
            print(f"    ⚠️ batch of {len(batch)} failed ({str(e).splitlines()[0]}); splitting",
                  file=sys.stderr)
            mid = len(batch) // 2
            return {**self.label_batch(batch[:mid]), **self.label_batch(batch[mid:])}

        labels = {}
        for item in items:
            try:
                idx = int(item.get("i"))
            except (AttributeError, TypeError, ValueError):
                continue
            category = str(item.get("category", "")).strip().lower()
            if 1 <= idx <= len(batch) and category in CATEGORIES:
                category = policy_override(batch[idx - 1], category)
                labels[batch[idx - 1]] = {"label": CATEGORIES[category], "category": category,
                                          "reason": str(item.get("reason", ""))}
        missing = [f for f in batch if f not in labels]
        if missing and len(missing) < len(batch):
            # Model skipped some items or invented a category — ask again for those
            labels.update(self.label_batch(missing))
        return labels

    def label_and_verify(self, batch: list[str], rng: random.Random) -> dict[str, dict]:
        """Label a batch, then re-label its positives once more in a different mix
        and keep label 1 only when both passes agree.

        Errors are almost all false positives that flip with batch composition
        (a maintainer guide called user_doc next to real user docs), so a second,
        differently-composed look at the ~10% positives buys precision cheaply.
        Negatives from the batch are mixed in as filler so the verifier does not
        see a list made only of plausible files."""
        labels = self.label_batch(batch)
        positives = [f for f, lab in labels.items() if lab["label"] == 1]
        if not self.verify or not positives:
            return labels
        negatives = [f for f, lab in labels.items() if lab["label"] == 0]
        check = positives + rng.sample(negatives, min(len(negatives), max(VERIFY_MIN_FILLER, len(positives))))
        rng.shuffle(check)
        second = self.label_batch(check)
        for f in positives:
            lab2 = second.get(f)
            if lab2 is not None and lab2["label"] == 0:
                labels[f] = {**lab2, "reason": f"[verify: {labels[f]['category']} -> {lab2['category']}] "
                                               f"{lab2['reason']}"}
        return labels

    def label(self, files: list[str], cache: "LabelCache", batch_size: int,
              workers: int = 1) -> dict[str, dict]:
        """Label `files`, reusing and extending the cache.
        Returns path → {label, category, reason}; files never labeled are absent."""
        repo = self.repo_dir.name
        results = {f: hit for f in files if (hit := cache.get(repo, f))}
        todo = [f for f in files if f not in results]
        print(f"   LLM labels: {len(results)} cached, {len(todo)} to label", flush=True)
        if todo:
            self._label_uncached(todo, results, cache, batch_size, workers)
        return self.apply_policies(results)

    @staticmethod
    def apply_policies(labels: dict[str, dict]) -> dict[str, dict]:
        """Re-apply policy_override (so cached labels follow the current rules)
        and the per_variant_impl majority vote.  Changed labels get a
        "[policy: old -> new]" prefix on the reason."""
        def relabel(f: str, category: str) -> None:
            lab = labels[f]
            if category != lab["category"]:
                labels[f] = {"label": CATEGORIES[category], "category": category,
                             "reason": f"[policy: {lab['category']} -> {category}] {lab['reason']}"}

        labels = dict(labels)
        for f, lab in list(labels.items()):
            relabel(f, policy_override(f, lab["category"]))

        # The model sees ~20 of the 81 conversion/*.py at a time and calls some
        # api_entry or example_main.  When it calls most files of a directory
        # (same extension) per_variant_impl, its positives there are the slips.
        groups: dict[tuple, list[str]] = {}
        for f in labels:
            d, _, name = f.replace("\\", "/").rpartition("/")
            groups.setdefault((d, Path(name).suffix.lower()), []).append(f)
        for fs in groups.values():
            if (len(fs) >= VOTE_MIN_FILES and
                    Counter(labels[f]["category"] for f in fs).most_common(1)[0][0] == "per_variant_impl"):
                for f in fs:
                    if labels[f]["label"] == 1:
                        relabel(f, "per_variant_impl")
        return labels

    def _label_uncached(self, todo: list[str], results: dict[str, dict], cache: "LabelCache",
                        batch_size: int, workers: int) -> None:
        repo = self.repo_dir.name
        # Mixed batches: path-sorted batches put 20 sibling files together and
        # the model labels them by analogy instead of one by one.
        rng = random.Random(f"{repo}:{PROMPT_VERSION}")
        rng.shuffle(todo)
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        # One RNG per batch, drawn up front, so the verification filler does not
        # depend on the order in which parallel batches finish.
        batch_rngs = [random.Random(rng.random()) for _ in batches]
        # Batches run concurrently (--workers) so a batching server such as vLLM
        # stays busy; cache writes stay on this thread.
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            futures = [pool.submit(self.label_and_verify, b, r) for b, r in zip(batches, batch_rngs)]
            for n, fut in enumerate(as_completed(futures), 1):
                labels = fut.result()
                print(f"   batch {n}/{len(batches)} done ({len(labels)} labels)", flush=True)
                results.update(labels)
                cache.add(repo, labels)
        finally:
            pool.shutdown(cancel_futures=True)


class LabelCache:
    """LLM labels on disk, keyed by repo, path, prompt version, model and the
    verify flag.  Labels do not depend on file_features, so feature changes
    can be retrained without relabeling."""

    def __init__(self, path: str | None, model: str, verify: bool):
        self.path = path
        # "1": content snippets were sent — kept so existing cache entries match
        self._suffix = [PROMPT_VERSION, model, "1", "verify" if verify else "single"]
        self._labels: dict[str, dict] = {}
        if path and Path(path).exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._labels[r["key"]] = {"label": r["label"], "category": r.get("category", ""),
                                                  "reason": r.get("reason", "")}
                    except (json.JSONDecodeError, KeyError):
                        continue

    def key(self, repo: str, path: str) -> str:
        return "\t".join([repo, path.replace("\\", "/"), *self._suffix])

    def get(self, repo: str, path: str) -> dict | None:
        return self._labels.get(self.key(repo, path))

    def add(self, repo: str, labels: dict[str, dict]) -> None:
        """Store labels, appending them to the file at once so an interrupted
        run keeps its progress."""
        lines = []
        for path, lab in labels.items():
            key = self.key(repo, path)
            self._labels[key] = lab
            lines.append(json.dumps({"key": key, **lab}, ensure_ascii=False) + "\n")
        if self.path and lines:
            with open(self.path, "a", encoding="utf-8") as f:
                f.writelines(lines)


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
# Android resource dirs (res/values, res/mipmap-*, res/layout, ...)
_ANDROID_RES = re.compile(r"(^|/)res/(values|mipmap|drawable|layout|xml)[^/]*/", re.I)


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


def auto_negative(rel: str, feats: dict, repo_dir: Path | None = None) -> str | None:
    """Reason string when a file is certainly not cheatsheet material, so it
    can be labeled without spending an LLM call; None otherwise."""
    p = Path(rel)
    name, ext = p.name.lower(), p.suffix.lower()
    if feats["in_ignored"]:
        return "vendored / generated / build-output path"
    if any(part.lower() in _VENDORED_DIRS for part in p.parts[:-1]):
        return "vendored dependency"
    if name in SKIP_NAMES:
        return "build, lint or lock file"
    if ext in SKIP_EXTS or ext in _ASSET_EXTS or feats["ext_group_data"]:
        return "binary, asset or data file"
    if _ANDROID_RES.search(rel.replace("\\", "/")):
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(root_dir: str, output_path: str, batch_size: int, cache_path: str | None = None,
             verify: bool = True, workers: int = 1):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        sys.exit(f"❌ Root not found: {root}")

    # Each direct child is a repo; otherwise root itself is a single repo
    repo_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")] or [root]
    client, model = llm_from_env()
    cache = LabelCache(cache_path, model, verify)

    records = []
    for repo_dir in repo_dirs:
        print(f"\n📂 Repo: {repo_dir.name}")
        files = list_repo_files(repo_dir)
        print(f"   {len(files)} files found")
        if not files:
            continue
        features_map = {f: extract_features(f) for f in files}
        auto_neg = {f: r for f in files if (r := auto_negative(f, features_map[f], repo_dir))}
        candidates = [f for f in files if f not in auto_neg]
        print(f"   {len(auto_neg)} auto-negative, {len(candidates)} sent to LLM")
        llm_labels = RepoLabeler(repo_dir, files, client, model, verify).label(
            candidates, cache, batch_size, workers)
        if len(llm_labels) < len(candidates):
            print(f"   ⚠️ {len(candidates) - len(llm_labels)} files left unlabeled", file=sys.stderr)

        labels = {**{f: {"label": 0, "category": "", "reason": r, "source": "auto"} for f, r in auto_neg.items()},
                  **{f: {**lab, "source": "llm"} for f, lab in llm_labels.items()}}
        records += [{"repo": repo_dir.name, "path": f, "label": lab["label"], "source": lab["source"],
                     "reason": lab["reason"],
                     "category": lab.get("category", ""),   # LLM category the label was derived from
                     "features": features_map[f]}
                    for f in files if (lab := labels.get(f))]

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    print(f"\n✅ Training data written: {output_path} ({len(records)} records)")
    print("\n📊 Label distribution:")
    for src in ("llm", "auto"):
        dist = Counter(r["label"] for r in records if r["source"] == src)
        if dist:
            print(f"  [{src}] include={dist[1]}  exclude={dist[0]}  "
                  f"include_rate={dist[1] / (dist[0] + dist[1]) * 100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate labeled training data for the ML file scorer")
    parser.add_argument("--root", required=True,
                        help="Root folder containing one sub-directory per repo")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of files sent to the LLM per request")
    parser.add_argument("--cache", default="model/tmp/llm_label_cache.jsonl",
                        help="LLM label cache (reused across runs); '' to disable")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the second LLM pass that re-checks label-1 files")
    parser.add_argument("--workers", type=int, default=1,
                        help="LLM batches in flight at once (e.g. 8 for vLLM)")
    args = parser.parse_args()
    generate(args.root, args.output, args.batch_size, cache_path=args.cache or None,
             verify=not args.no_verify, workers=args.workers)


if __name__ == "__main__":
    main()
