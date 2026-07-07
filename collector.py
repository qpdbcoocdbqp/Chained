"""
collector.py
Responsible for: git clone target repository, scanning key files, packaging into a "context text"
for subsequent analysis using the Claude API.
"""
import os
import subprocess
import tempfile
import shutil

# High priority filenames (case-insensitive matching)
PRIORITY_FILES = [
    "readme.md", "readme", "readme.rst", "readme.txt",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
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


def collect_interest_dirs(root: str) -> dict:
    """Collect a small number of representative files under directories like examples/docs"""
    collected = {}
    total_chars = 0
    for d in INTEREST_DIRS:
        dir_path = os.path.join(root, d)
        if not os.path.isdir(dir_path):
            continue
        count = 0
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirnames[:] = [x for x in dirnames if x not in IGNORE_DIRS and not x.startswith(".")]
            for f in sorted(filenames):
                if count >= 5:  # Take at most 5 file examples per directory
                    break
                if total_chars > MAX_TOTAL_CHARS:
                    return collected
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root)
                content = read_file_safe(full_path, max_chars=2000)
                collected[rel_path] = content
                total_chars += len(content)
                count += 1
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
        example_files = collect_interest_dirs(local_path)

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

