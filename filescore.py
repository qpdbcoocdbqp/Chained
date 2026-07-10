from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

KEEP_EXTS = {
    ".md",
    ".py",
    ".ipynb",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}

SKIP_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".ico",
    ".bmp",
    ".mp4",
    ".mov",
    ".avi",
    ".mp3",
    ".wav",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".onnx",
    ".gguf",
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".parquet",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
    ".pytest_cache",
}

PATH_SCORE = {
    "readme": 100,
    "docs": 95,
    "examples": 90,
    "example": 90,
    "tutorials": 88,
    "guides": 88,
    "tests": 75,
    ".github/workflows": 70,
    "scripts": 65,
    "src": 60,
    "config": 55,
}

EXT_SCORE = {
    ".md": 20,
    ".py": 15,
    ".ipynb": 15,
    ".yaml": 12,
    ".yml": 12,
    ".toml": 10,
    ".json": 5,
    ".js": 10,
    ".ts": 10,
    ".sh": 8,
}

NAME_SCORE = {
    "architecture": 30,
    "design": 30,
    "getting_started": 30,
    "quickstart": 28,
    "tutorial": 25,
    "example": 20,
    "api": 20,
    "cli": 18,
    "config": 15,
    "main": 25,
    "server": 20,
    "agent": 15,
    "rag": 15,
    "tool": 15,
}

NEGATIVE_NAME_SCORE = {
    "benchmark": -5,
    "helper": -5,
    "helpers": -5,
    "utils": -5,
    "generated": -20,
    "snapshot": -20,
    "output": -20,
    "cache": -20,
    "dataset": -30,
    "embedding": -30,
}

JSON_KEEP = {
    "config",
    "request",
    "response",
    "schema",
    "prompt",
    "template",
    "workflow",
    "tool",
    "agent",
    "openapi",
    "swagger",
}

JSON_SKIP = {
    "dataset",
    "embedding",
    "cache",
    "output",
    "result",
    "snapshot",
    "train",
    "eval",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _size_score(size: int | None) -> int:
    if size is None:
        return 0

    kb = size / 1024

    if kb < 20:
        return 10
    if kb < 100:
        return 5
    if kb < 500:
        return 0
    if kb < 2048:
        return -10
    return -30


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def priority_score(path: str | Path, size_bytes: int | None = None) -> int:
    """
    Compute priority score (0~100+) for LLM indexing.
    """

    p = Path(path)

    if any(part in SKIP_DIRS for part in p.parts):
        return 0

    ext = p.suffix.lower()
    stem = p.stem.lower()
    path_str = str(p).replace("\\", "/").lower()

    if ext in SKIP_EXTS:
        return 0

    score = 0

    # Path
    if path_str.startswith("readme"):
        score += PATH_SCORE["readme"]

    for k, v in PATH_SCORE.items():
        if k == "readme":
            continue
        if k in path_str:
            score += v
            break

    # Extension
    score += EXT_SCORE.get(ext, 0)

    # Filename keywords
    for k, v in NAME_SCORE.items():
        if k in stem:
            score += v

    for k, v in NEGATIVE_NAME_SCORE.items():
        if k in stem:
            score += v

    # JSON heuristic
    if ext == ".json":
        if any(k in stem for k in JSON_KEEP):
            score += 15

        if any(k in stem for k in JSON_SKIP):
            score -= 40

    # Size
    score += _size_score(size_bytes)

    return max(score, 0)


def should_scan(
    path: str | Path,
    size_bytes: int | None = None,
    threshold: int = 70,
) -> bool:
    """
    Whether the file should be summarized by the LLM.
    """
    return priority_score(path, size_bytes) >= threshold


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    samples = [
        ("README.md", 6000),
        ("docs/architecture.md", 18000),
        ("examples/basic.py", 4000),
        ("examples/request.json", 2000),
        ("examples/output.json", 2000),
        ("tests/test_agent.py", 9000),
        ("src/main.py", 7000),
        ("src/utils.py", 12000),
        ("assets/logo.png", 80000),
        ("models/model.gguf", 5_000_000_000),
    ]

    for path, size in samples:
        print(
            f"{priority_score(path, size):3d}  "
            f"{should_scan(path, size)}  "
            f"{path}"
        )
