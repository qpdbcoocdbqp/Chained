#!/usr/bin/env python3
"""
semantic_search.py — Semantic chunking, indexing, and search over a folder of
<repo_name>/CHEATSHEET.md (or AGENT_GUIDE.md) files.

Usage:
    # 1. Build an index (recursively scans the folder for CHEATSHEET.md / AGENT_GUIDE.md)
    python3 semantic_search.py build --root /path/to/cheatsheets --backend json --index index.json
    python3 semantic_search.py build --root /path/to/cheatsheets --backend lancedb --uri ./lancedb_data

    # 2. Search
    python3 semantic_search.py search --backend json --index index.json --query "how do I do vector search in Milvus" --top-k 5
    python3 semantic_search.py search --backend lancedb --uri ./lancedb_data --query "..." --top-k 5

Storage backends:
    --backend json      Simple JSON file index (default). Good for small/local use, no
                         extra services required.
    --backend lancedb    LanceDB table. --uri can be a local path (e.g. ./lancedb_data)
                         or an S3-compatible URI (e.g. s3://bucket/prefix) backed by
                         MinIO — see MinIO environment variables below.

Embedding backend priority:
    1. fastembed (default) — local ONNX quantized model. The model is downloaded once
       to a local cache (~/.cache/fastembed) on first run, then works fully offline,
       with no API key and no server to run.
       Default model is the multilingual paraphrase-multilingual-MiniLM-L12-v2, which
       supports both English and Chinese.
    2. Local OpenAI-compatible /embeddings endpoint (if LOCAL_EMBEDDING_BASE_URL is
       set, this overrides fastembed and uses your own local server instead)
    3. If neither is available (fastembed not installed, and no local endpoint set),
       an error is raised with installation instructions.

Environment variables (optional, following the project's existing LOCAL_LLM_* naming
convention):
    LOCAL_EMBEDDING_BASE_URL   e.g. http://localhost:1234/v1  (only takes effect if set)
    LOCAL_EMBEDDING_MODEL      e.g. text-embedding-nomic-embed-text-v1.5
    LOCAL_EMBEDDING_API_KEY    e.g. lm-studio (some local servers ignore this but still
                                require a value to be sent)
    FASTEMBED_MODEL            override the default fastembed model name (see fastembed's
                                supported model list)

MinIO / S3-compatible storage environment variables (only used with --backend lancedb
and an s3:// --uri):
    MINIO_ENDPOINT      e.g. http://localhost:9000
    MINIO_ACCESS_KEY    access key / username
    MINIO_SECRET_KEY    secret key / password
    MINIO_REGION        default: us-east-1 (required by the S3 protocol even though
                         MinIO itself doesn't really use regions)
    MINIO_ALLOW_HTTP    default: true (set to "false" if your MinIO is behind HTTPS)

Install fastembed (recommended for first-time use):
    pip install fastembed --break-system-packages

Install lancedb (only needed for --backend lancedb):
    pip install lancedb --break-system-packages
"""

import argparse
import json
import os
import re
import sys
import math
import urllib.request
import urllib.error
from pathlib import Path

# --------------------------------------------------------------------------
# Chunking: split by Markdown headings (## / ###), keeping the heading
# hierarchy as context for each chunk.
# --------------------------------------------------------------------------

def chunk_markdown(text: str, min_chars: int = 40):
    """
    Split a Markdown document into chunks by heading. Each chunk carries a
    "breadcrumb" of the heading path it belongs to, so search results show
    which section the content came from.
    """
    lines = text.splitlines()
    chunks = []
    current_heading_stack = []  # [(level, title), ...]
    buf = []

    def flush():
        content = "\n".join(buf).strip()
        if len(content) >= min_chars:
            breadcrumb = " > ".join(h for _, h in current_heading_stack)
            chunks.append({"heading": breadcrumb, "content": content})
        buf.clear()

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    in_code_block = False
    fence_re = re.compile(r"^\s*```")
    for line in lines:
        if fence_re.match(line):
            in_code_block = not in_code_block
            buf.append(line)
            continue
        # Skip heading detection inside fenced code blocks, so that e.g. a
        # "# comment" in a Python snippet isn't mistaken for a Markdown heading.
        m = None if in_code_block else heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop headings that are the same level or deeper than the new one
            current_heading_stack = [h for h in current_heading_stack if h[0] < level]
            current_heading_stack.append((level, title))
            buf.append(line)  # keep the heading line itself in this chunk
        else:
            buf.append(line)
    flush()

    # If the whole document has no headings at all, treat it as a single chunk
    if not chunks and text.strip():
        chunks.append({"heading": "(whole document)", "content": text.strip()})

    return chunks


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------

class EmbeddingError(RuntimeError):
    pass


def _embed_via_local_api(texts, base_url, model, api_key):
    url = base_url.rstrip("/") + "/embeddings"
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise EmbeddingError(
            f"Could not connect to local embedding endpoint {url}: {e}\n"
            f"Check that LOCAL_EMBEDDING_BASE_URL / LOCAL_EMBEDDING_MODEL are set "
            f"correctly and that the server is running at that address."
        ) from e
    try:
        data = payload["data"]
        return [item["embedding"] for item in data]
    except (KeyError, TypeError) as e:
        raise EmbeddingError(f"Unexpected response format from local embedding endpoint: {payload}") from e


_fastembed_model_cache = {}

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _embed_via_fastembed(texts, model_name=None):
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise EmbeddingError(
            "fastembed is not installed, and LOCAL_EMBEDDING_BASE_URL is not set.\n"
            "Pick one:\n"
            "  1) Run `pip install fastembed --break-system-packages` "
            "(recommended — local ONNX model, no API key, model auto-downloads on "
            "first run); or\n"
            "  2) Set LOCAL_EMBEDDING_BASE_URL / LOCAL_EMBEDDING_MODEL to point at "
            "your own local OpenAI-compatible embedding endpoint."
        ) from e

    name = model_name or os.environ.get("FASTEMBED_MODEL", DEFAULT_FASTEMBED_MODEL)
    if name not in _fastembed_model_cache:
        try:
            _fastembed_model_cache[name] = TextEmbedding(model_name=name)
        except Exception as e:
            raise EmbeddingError(
                f"fastembed failed to initialize model '{name}': {e}\n"
                f"First use requires internet access to download the model file "
                f"(cached to ~/.cache/fastembed). If this environment has no internet "
                f"access, use LOCAL_EMBEDDING_BASE_URL to point at a local server instead."
            ) from e

    model = _fastembed_model_cache[name]
    vectors = list(model.embed(texts))
    return [v.tolist() for v in vectors]


def embed_texts(texts):
    """
    Get embedding vectors for a batch of texts.
    Defaults to fastembed (local ONNX, no API key required); if
    LOCAL_EMBEDDING_BASE_URL is set, that local server is used instead (override).
    """
    base_url = os.environ.get("LOCAL_EMBEDDING_BASE_URL")
    if base_url:
        model = os.environ.get("LOCAL_EMBEDDING_MODEL", "text-embedding-ada-002")
        api_key = os.environ.get("LOCAL_EMBEDDING_API_KEY", "not-needed")
        # Call in batches to avoid sending too much at once
        out = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            out.extend(_embed_via_local_api(texts[i:i + batch_size], base_url, model, api_key))
        return out
    else:
        return _embed_via_fastembed(texts)


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

TARGET_FILENAMES = {"cheatsheet.md", "agent_guide.md"}


def find_cheatsheet_files(root: Path):
    """Recursively find files matching the <repo_name>/CHEATSHEET.md pattern (case-insensitive)."""
    found = []
    for p in root.rglob("*.md"):
        if p.name.lower() in TARGET_FILENAMES:
            found.append(p)
    return sorted(found)


def guess_repo_name(file_path: Path, root: Path):
    """Use the file's immediate parent directory name as the repo name; if the
    file sits directly under root, fall back to the filename without extension."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = file_path
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return file_path.stem


def collect_chunks(root_dir: str):
    """Scan root_dir for cheat sheet files, chunk them, and return the list of
    chunk dicts (without embeddings yet)."""
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"❌ Folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = find_cheatsheet_files(root)
    if not files:
        print(f"⚠️ No CHEATSHEET.md / AGENT_GUIDE.md found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"📂 Found {len(files)} file(s), chunking...")

    all_chunks = []  # list of dict: repo, file, heading, content
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        repo = guess_repo_name(f, root)
        for c in chunk_markdown(text):
            all_chunks.append({
                "repo": repo,
                "file": str(f),
                "heading": c["heading"],
                "content": c["content"],
            })

    print(f"🧩 Split into {len(all_chunks)} semantic chunk(s).")
    return all_chunks


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def print_results(query, scored, top_k):
    """scored: list of (score, chunk_dict) sorted descending, chunk_dict has
    repo/heading/content/file keys."""
    print(f"\n🔍 Query: \"{query}\"  Top {top_k} results:\n")
    for rank, (sim, c) in enumerate(scored[:top_k], 1):
        print(f"[{rank}] score={sim:.3f}  repo={c['repo']}  section={c['heading']}")
        preview = c["content"].strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"    {preview}")
        print(f"    source file: {c['file']}\n")


# --------------------------------------------------------------------------
# JSON backend
# --------------------------------------------------------------------------

def build_index_json(root_dir: str, index_path: str):
    all_chunks = collect_chunks(root_dir)

    contents = [c["content"] for c in all_chunks]
    try:
        vectors = embed_texts(contents)
    except EmbeddingError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    for c, v in zip(all_chunks, vectors):
        c["embedding"] = v

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False)

    print(f"✅ Index written: {index_path} ({len(all_chunks)} chunk(s))")


def search_index_json(index_path: str, query: str, top_k: int = 5, repo_filter: str = None):
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

    if repo_filter:
        chunks = [c for c in chunks if c["repo"] == repo_filter]
        if not chunks:
            print(f"⚠️ No chunks found for repo '{repo_filter}'", file=sys.stderr)
            return

    try:
        q_vec = embed_texts([query])[0]
    except EmbeddingError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    scored = [(cosine_sim(q_vec, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    print_results(query, scored, top_k)


# --------------------------------------------------------------------------
# LanceDB (+ optional MinIO) backend
# --------------------------------------------------------------------------

DEFAULT_LANCEDB_TABLE = "cheatsheet_chunks"


def _lancedb_storage_options():
    """Build storage_options for connecting LanceDB to an S3-compatible
    endpoint such as MinIO, from environment variables. Returns None if no
    MinIO endpoint is configured (i.e. the --uri is a plain local path)."""
    endpoint = os.environ.get("MINIO_ENDPOINT")
    if not endpoint:
        return None
    options = {
        "endpoint": endpoint,
        "region": os.environ.get("MINIO_REGION", "us-east-1"),
        "allow_http": os.environ.get("MINIO_ALLOW_HTTP", "true"),
    }
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    if access_key:
        options["aws_access_key_id"] = access_key
    if secret_key:
        options["aws_secret_access_key"] = secret_key
    return options


def _lancedb_connect(uri: str):
    try:
        import lancedb
    except ImportError as e:
        raise RuntimeError(
            "lancedb is not installed. Run `pip install lancedb --break-system-packages`."
        ) from e

    storage_options = _lancedb_storage_options() if uri.startswith("s3://") else None
    if storage_options:
        return lancedb.connect(uri, storage_options=storage_options)
    return lancedb.connect(uri)


def build_index_lancedb(root_dir: str, uri: str, table_name: str = DEFAULT_LANCEDB_TABLE):
    all_chunks = collect_chunks(root_dir)

    contents = [c["content"] for c in all_chunks]
    try:
        vectors = embed_texts(contents)
    except EmbeddingError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for c, v in zip(all_chunks, vectors):
        rows.append({
            "repo": c["repo"],
            "file": c["file"],
            "heading": c["heading"],
            "content": c["content"],
            "vector": v,
        })

    print(f"🗄️  Connecting to LanceDB at {uri} ...")
    try:
        db = _lancedb_connect(uri)
        # mode="overwrite" so re-running build refreshes the table from scratch
        db.create_table(table_name, data=rows, mode="overwrite")
    except Exception as e:
        print(f"❌ Failed to write to LanceDB: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ LanceDB table '{table_name}' written at {uri} ({len(rows)} chunk(s))")


def search_index_lancedb(uri: str, query: str, top_k: int = 5, repo_filter: str = None,
                          table_name: str = DEFAULT_LANCEDB_TABLE):
    try:
        db = _lancedb_connect(uri)
        tbl = db.open_table(table_name)
    except Exception as e:
        print(f"❌ Failed to open LanceDB table '{table_name}' at {uri}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        q_vec = embed_texts([query])[0]
    except EmbeddingError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    search_q = tbl.search(q_vec).limit(top_k)
    if repo_filter:
        search_q = search_q.where(f"repo = '{repo_filter}'")

    results = search_q.to_list()
    if not results:
        print(f"⚠️ No results found" + (f" for repo '{repo_filter}'" if repo_filter else ""))
        return

    # LanceDB returns a `_distance` column (lower = more similar for the default
    # L2 metric); convert to a "higher is better" score for consistent display.
    scored = []
    for r in results:
        score = 1.0 / (1.0 + r.get("_distance", 0.0))
        scored.append((score, r))
    print_results(query, scored, top_k)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic search over cheat sheets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Scan a folder and build an index")
    p_build.add_argument("--root", required=True, help="Root folder containing <repo_name>/CHEATSHEET.md files")
    p_build.add_argument("--backend", choices=["json", "lancedb"], default="json", help="Storage backend")
    p_build.add_argument("--index", default="index.json", help="[json backend] output index file path")
    p_build.add_argument("--uri", default="./lancedb_data", help="[lancedb backend] local path or s3://bucket/prefix (MinIO)")
    p_build.add_argument("--table", default=DEFAULT_LANCEDB_TABLE, help="[lancedb backend] table name")

    p_search = sub.add_parser("search", help="Run a semantic search against an existing index")
    p_search.add_argument("--backend", choices=["json", "lancedb"], default="json", help="Storage backend")
    p_search.add_argument("--index", default="index.json", help="[json backend] index file path")
    p_search.add_argument("--uri", default="./lancedb_data", help="[lancedb backend] local path or s3://bucket/prefix (MinIO)")
    p_search.add_argument("--table", default=DEFAULT_LANCEDB_TABLE, help="[lancedb backend] table name")
    p_search.add_argument("--query", required=True, help="Search query string")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    p_search.add_argument("--repo", default=None, help="Restrict search to a specific repo name")

    args = parser.parse_args()

    if args.command == "build":
        if args.backend == "json":
            build_index_json(args.root, args.index)
        else:
            build_index_lancedb(args.root, args.uri, args.table)
    elif args.command == "search":
        if args.backend == "json":
            search_index_json(args.index, args.query, args.top_k, args.repo)
        else:
            search_index_lancedb(args.uri, args.query, args.top_k, args.repo, args.table)


if __name__ == "__main__":
    main()
