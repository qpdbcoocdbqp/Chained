"""
collector.py
Responsible for: git clone target repository, scanning key files, packaging into a "context text"
for subsequent analysis using the Claude API.
"""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from filescore import priority_score
from model.train_ml_scorer import FileScorer


# Tried in order; the first one compatible with the current feature layout wins.
SCORER_MODELS = ["scorer_v3.pkl", "scorer_v2.pkl", "llm_scorer.pkl", "scorer.pkl"]


def _load_scorer() -> FileScorer | None:
    base = Path(__file__).resolve().parent
    for name in SCORER_MODELS:
        try:
            loaded = FileScorer.load(str(base / name))
        except SystemExit:      # _load_model exits when the file is missing
            continue
        except Exception as err:
            print(f"Skipping score model {name}: {err}")
            continue
        print(f"Loaded score model: {name}")
        return loaded
    print("No score model loaded; falling back to filescore.priority_score.")
    return None


scorer = _load_scorer()

# High priority filenames (case-insensitive matching)
PRIORITY_FILES = [
    "readme.md", "readme", "readme.rst", "readme.txt",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "cargo.toml", "go.mod", "gemfile", "pom.xml", "build.gradle",
    "contributing.md", "usage.md", "getting_started.md", "quickstart.md",
    "docs/readme.md",
]

# Directories worth overall inclusion (examples/docs)
INTEREST_DIRS = ["examples", "example", "docs", "sample", "samples"]

MAX_FILE_CHARS = 6000       # Maximum slice length of a single file
MAX_TOTAL_CHARS = 60000     # Upper limit of total packaged content to avoid exceeding context
IGNORE_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv"}


def clone_repo(repo_url: str) -> str:
    """Shallow clone the repository to a temporary directory and return the local path"""
    tmp_dir = tempfile.mkdtemp(prefix="repo_cheatsheet_")
    print(f"Cloning {repo_url} -> {tmp_dir}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, tmp_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {result.stderr}")
    return tmp_dir


def get_dir_tree(root: str, max_depth: int = 2) -> str:
    """Generate directory structure (limited depth) to help LLM understand the project layout"""
    lines = []
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        name = os.path.basename(dirpath) or dirpath
        lines.append(f"{indent}{name}/")
        if depth == max_depth:
            continue
        for f in sorted(filenames):
            if not f.startswith("."):
                lines.append(f"{indent}  {f}")
    return "\n".join(lines[:300])  # Prevent screen flooding from huge repositories


def read_file_safe(path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars)
        return content
    except Exception as e:
        return f"[Cannot read: {e}]"


def collect_priority_files(root: str) -> dict:
    """Search for priority files in the repository root directory and one level of subdirectories"""
    found = {}
    lower_map = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth > 1:
            dirnames[:] = []
            continue
        for f in filenames:
            rel_path = os.path.normpath(os.path.join(rel_dir, f)) if rel_dir != "." else f
            lower_map[rel_path.lower()] = rel_path

    for target in PRIORITY_FILES:
        target_norm = target.lower()
        if target_norm in lower_map:
            rel_path = lower_map[target_norm]
            found[rel_path] = read_file_safe(os.path.join(root, rel_path))
    return found


MAX_SECONDARY_FILES = 20    # Maximum number of secondary files to include
SECONDARY_THRESHOLD = 0.5  # Minimum score to be considered
SECONDARY_FILE_CHARS = 2048
MAX_PER_PARENT_DIR = 3      # e.g. at most 3 files from examples/simple/
MAX_PER_TOP_DIR = 5         # e.g. at most 5 files from examples/ overall
# Slots reserved for source code; docs outscore code on rule_score, so without
# a quota READMEs take every slot.  Split so that example code (which already
# shares examples/ with READMEs) cannot also take the public-API slots.
MIN_API_CODE_FILES = 3      # public headers / package entry files, most-imported first
MIN_EXAMPLE_CODE_FILES = 2  # runnable samples
SCORE_TIE_DECIMALS = 2      # model probabilities saturate near 1.0; treat close scores as ties


RULE_SCORE_BUCKET = 0.05      # rule scores this close are compared on content
STUB_FILE_BYTES = 300         # READMEs smaller than this are usually placeholders
_DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".adoc"}


def _content_signal(abs_path: Path) -> int:
    """Cheap content-based usefulness hint for files that already passed the
    path-based scorer.

    For docs: the number of fenced code blocks / shell prompts, since a
    cheatsheet is built from runnable commands and snippets.  Near-empty
    stubs get -1.  Other files: 0 (ref_count covers code).
    """
    try:
        size = abs_path.stat().st_size
    except OSError:
        return 0
    if size < STUB_FILE_BYTES:
        return -1
    if abs_path.suffix.lower() not in _DOC_SUFFIXES:
        return 0
    text = read_file_safe(str(abs_path), max_chars=64 * 1024)
    fences = sum(1 for line in text.splitlines() if line.lstrip().startswith(("```", "~~~")))
    prompts = sum(1 for line in text.splitlines() if line.lstrip().startswith("$ "))
    return min(fences // 2 + prompts, 30)


def _rank_key(score: float, feats: dict, content: int, rel: str) -> tuple:
    """Sort key: model score first, then deterministic tie-breakers.

    Without tie-breakers a saturated model (hundreds of files at ~1.00) falls
    back to reverse-alphabetical path order, which is meaningless.
    """
    return (
        -round(score, SCORE_TIE_DECIMALS),
        -round(feats["rule_score"] / RULE_SCORE_BUCKET),   # finer-grained heuristic signal
        -content,                  # docs with more runnable snippets first
        feats["depth"],            # shallower first
        -feats["ref_count"],       # more widely imported first
        rel,
    )


def _api_rank_key(score: float, feats: dict, example_refs: int, rel: str) -> tuple:
    """For public-API code, what the examples import is the best usefulness
    hint (llama.h, milvusclient), then the overall import count; ref_count
    alone favours internal hubs such as src/models/models.h or milvus pkg/."""
    return (
        -round(score, SCORE_TIE_DECIMALS),
        -example_refs,
        -feats["ref_count"],
        -round(feats["rule_score"] / RULE_SCORE_BUCKET),
        feats["depth"],
        rel,
    )


def _is_api_code(feats: dict) -> bool:
    return bool(feats["ext_group_code"] and (feats["is_public_api"] or feats["is_cli_entry"])
                and not (feats["in_examples"] or feats["is_example_file"]))


def _is_example_code(feats: dict) -> bool:
    return bool(feats["ext_group_code"] and (feats["in_examples"] or feats["is_example_file"]))


def _dir_keys(rel: str) -> tuple[str, str]:
    parts = Path(rel).parts
    return str(Path(rel).parent), (parts[0] if len(parts) > 1 else "")


def _select_diverse(ranked: list[str], limit: int, picked: list[str] | None = None) -> list[str]:
    """Pick up to `limit` paths, capping how many come from the same directory
    so that one directory with many similar files (e.g. 27 examples/*/README.md)
    cannot crowd out everything else.

    `picked` are paths already chosen (e.g. reserved code slots); they count
    toward the limit and the per-directory caps."""
    picked = list(picked or [])
    per_parent: dict[str, int] = {}
    per_top: dict[str, int] = {}
    for rel in picked:
        parent, top = _dir_keys(rel)
        per_parent[parent] = per_parent.get(parent, 0) + 1
        if top:
            per_top[top] = per_top.get(top, 0) + 1
    for rel in ranked:
        if len(picked) >= limit:
            break
        if rel in picked:
            continue
        parent, top = _dir_keys(rel)
        if per_parent.get(parent, 0) >= MAX_PER_PARENT_DIR:
            continue
        if top and per_top.get(top, 0) >= MAX_PER_TOP_DIR:
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        if top:
            per_top[top] = per_top.get(top, 0) + 1
        picked.append(rel)
    return picked


def collect_secondary_files(root: str, exclude: set[str] | None = None) -> dict:
    """Score every file in the repo and return the top candidates.

    Instead of restricting to INTEREST_DIRS upfront (which misses important
    files in src/, shallow package roots, CLI entry points, etc.), we walk the
    entire tree, score each file with the ML/rule-based scorer, and keep only
    the highest-ranked ones.  INTEREST_DIRS is kept as a soft prior via the
    scorer's `in_examples` / `in_docs` features, so example files still rank
    well without being the *only* things considered.

    Reference counts (how many other files import each file) are computed once
    for the whole repo and injected into the scorer as an extra signal, so
    heavily-imported API modules rank higher than isolated helper files.

    `exclude` holds paths already collected as priority files, so the same
    README is not packed twice.
    """
    from model.file_features import build_ref_counts, extract_features, features_to_vector

    exclude_norm = {os.path.normcase(os.path.normpath(p)) for p in (exclude or ())}

    # Collect all candidate file paths first
    all_rel: list[str] = []
    abs_map: dict[str, Path] = {}   # rel_path str → abs Path
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            abs_path = Path(dirpath) / fname
            rel = str(abs_path.relative_to(root))
            all_rel.append(rel)
            abs_map[rel] = abs_path

    # Compute cross-file reference counts for the whole repo in one pass
    print(f"Building reference counts for {len(all_rel)} files...")
    example_refs: dict[str, int] = {}
    ref_counts = build_ref_counts(root, all_rel, example_counts=example_refs)

    candidates = [r for r in all_rel if os.path.normcase(r) not in exclude_norm]
    feats_map = {r: extract_features(r, ref_count=ref_counts.get(r, 0)) for r in candidates}

    if scorer is None:
        scores = {r: min(priority_score(r) / 100.0, 1.0) for r in candidates}
    else:
        probas = scorer._pipeline.predict_proba(
            [features_to_vector(feats_map[r]) for r in candidates]
        )[:, 1] if candidates else []
        scores = dict(zip(candidates, (float(p) for p in probas)))

    passing = [r for r in candidates if scores[r] >= SECONDARY_THRESHOLD]
    content_sig = {r: _content_signal(abs_map[r]) for r in passing}
    passing.sort(key=lambda r: _rank_key(scores[r], feats_map[r], content_sig[r], r))

    # Docs carry a higher base rule_score than code, so without a quota the
    # public headers / API modules lose every slot to READMEs.
    api_code = sorted((r for r in passing if _is_api_code(feats_map[r])),
                      key=lambda r: _api_rank_key(scores[r], feats_map[r], example_refs.get(r, 0), r))
    example_code = [r for r in passing if _is_example_code(feats_map[r])]   # already in rank order
    reserved = _select_diverse(api_code, MIN_API_CODE_FILES)
    reserved = _select_diverse(example_code, len(reserved) + MIN_EXAMPLE_CODE_FILES, picked=reserved)
    top_files = _select_diverse(passing, MAX_SECONDARY_FILES, picked=reserved)
    order = {r: i for i, r in enumerate(passing)}
    top_files.sort(key=order.__getitem__)

    print(f"Secondary files: {len(passing)} candidates >= {SECONDARY_THRESHOLD} "
          f"({len(api_code)} API code, {len(example_code)} example code), "
          f"keeping top {len(top_files)} ({len(reserved)} reserved for code)")

    collected = {}
    for rel in top_files:
        content = read_file_safe(str(abs_map[rel]), max_chars=SECONDARY_FILE_CHARS)
        collected[rel] = content

    return collected


def build_context_bundle(repo_url: str):
    """Main function: clone + scan + package into a single text

    Returns (bundle, scanned_files, truncated):
        bundle        -- Packaged context text
        scanned_files -- List of files actually read this time,
                         each item is a dict: {"path": relative path, "category": "priority"|"example", "chars": actual character count}
        truncated     -- bool, whether the packaged content was truncated due to exceeding MAX_TOTAL_CHARS
    """
    local_path = clone_repo(repo_url)
    try:
        tree = get_dir_tree(local_path)
        priority_files = collect_priority_files(local_path)
        example_files = collect_secondary_files(local_path, exclude=set(priority_files))

        parts = [f"# Repository: {repo_url}\n", "## Directory Structure (Partial)\n```\n" + tree + "\n```\n"]
        scanned_files = []

        for path, content in priority_files.items():
            parts.append(f"## File: {path}\n```\n{content}\n```\n")
            scanned_files.append({"path": path, "category": "priority", "chars": len(content)})

        for path, content in example_files.items():
            parts.append(f"## Example File: {path}\n```\n{content}\n```\n")
            scanned_files.append({"path": path, "category": "example", "chars": len(content)})

        bundle = "\n".join(parts)
        truncated = False
        if len(bundle) > MAX_TOTAL_CHARS:
            bundle = bundle[:MAX_TOTAL_CHARS] + "\n\n[Content too long, truncated]"
            truncated = True

        return bundle, scanned_files, truncated
    finally:
        shutil.rmtree(local_path, ignore_errors=True)

