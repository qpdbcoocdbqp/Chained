#!/usr/bin/env python3
"""
semantic_search.py — 對一批 <repo_name>/CHEATSHEET.md（或 AGENT_GUIDE.md）
做語意分塊索引與搜尋。

用法：
    # 1. 建立索引（掃描資料夾，遞迴尋找 CHEATSHEET.md / AGENT_GUIDE.md）
    python3 semantic_search.py build --root /path/to/cheatsheets --index index.json

    # 2. 搜尋
    python3 semantic_search.py search --index index.json --query "Milvus 向量搜尋怎麼用" --top-k 5

Embedding 後端優先順序：
    1. fastembed（預設）— 本地 ONNX 量化模型，第一次執行會自動下載模型到本地
       快取（~/.cache/fastembed），之後完全離線、免 API key、免起任何伺服器。
       預設模型為多語言的 paraphrase-multilingual-MiniLM-L12-v2，中英文皆可用。
    2. 本地 OpenAI-compatible /embeddings 端點（若設定了 LOCAL_EMBEDDING_BASE_URL，
       會覆蓋 fastembed，改用你自己起的伺服器）
    3. 都不可用（fastembed 未安裝、且沒設本地端點）則報錯並提示安裝方式

環境變數（與專案既有的 LOCAL_LLM_* 命名風格一致，皆為選用）：
    LOCAL_EMBEDDING_BASE_URL   e.g. http://localhost:1234/v1  （設了才會啟用）
    LOCAL_EMBEDDING_MODEL      e.g. text-embedding-nomic-embed-text-v1.5
    LOCAL_EMBEDDING_API_KEY    e.g. lm-studio（有些本地伺服器不檢查，但仍需給值）
    FASTEMBED_MODEL            覆蓋 fastembed 預設模型名稱（見 fastembed 支援清單）

安裝 fastembed（首次使用建議）：
    pip install fastembed --break-system-packages
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
# 分塊：依 Markdown 標題 (## / ###) 切分，保留標題階層作為 context
# --------------------------------------------------------------------------

def chunk_markdown(text: str, min_chars: int = 40):
    """
    依標題切分成塊。每個 chunk 帶著它所屬的標題路徑（heading breadcrumb），
    這樣搜尋結果才知道是哪個章節底下的內容。
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
        m = None if in_code_block else heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # 彈出比目前層級深或同層的標題
            current_heading_stack = [h for h in current_heading_stack if h[0] < level]
            current_heading_stack.append((level, title))
            buf.append(line)  # 標題本身也放進這個 chunk 的開頭
        else:
            buf.append(line)
    flush()

    # 若整份文件完全沒有標題也切不出東西，就整篇當一塊
    if not chunks and text.strip():
        chunks.append({"heading": "(整份文件)", "content": text.strip()})

    return chunks


# --------------------------------------------------------------------------
# Embedding 後端
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
            f"無法連線到本地 embedding 端點 {url}：{e}\n"
            f"請確認 LOCAL_EMBEDDING_BASE_URL / LOCAL_EMBEDDING_MODEL 設定正確，"
            f"且該伺服器有跑在這個位址上。"
        ) from e
    try:
        data = payload["data"]
        return [item["embedding"] for item in data]
    except (KeyError, TypeError) as e:
        raise EmbeddingError(f"本地 embedding 端點回傳格式不符預期：{payload}") from e


_fastembed_model_cache = {}

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _embed_via_fastembed(texts, model_name=None):
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise EmbeddingError(
            "未安裝 fastembed，且未設定 LOCAL_EMBEDDING_BASE_URL。\n"
            "請擇一：\n"
            "  1) 執行 `pip install fastembed --break-system-packages`"
            "（推薦，本地 ONNX 模型，免 API key，首次執行會自動下載模型）；或\n"
            "  2) 設定環境變數 LOCAL_EMBEDDING_BASE_URL / LOCAL_EMBEDDING_MODEL "
            "指向你自己起的本地 OpenAI-compatible embedding 端點。"
        ) from e

    name = model_name or os.environ.get("FASTEMBED_MODEL", DEFAULT_FASTEMBED_MODEL)
    if name not in _fastembed_model_cache:
        try:
            _fastembed_model_cache[name] = TextEmbedding(model_name=name)
        except Exception as e:
            raise EmbeddingError(
                f"fastembed 初始化模型 '{name}' 失敗：{e}\n"
                f"首次使用需要能連網下載模型檔（存到 ~/.cache/fastembed）。"
                f"若環境無法連網，請改用 LOCAL_EMBEDDING_BASE_URL 指向本地伺服器。"
            ) from e

    model = _fastembed_model_cache[name]
    vectors = list(model.embed(texts))
    return [v.tolist() for v in vectors]


def embed_texts(texts):
    """
    對一批文字取得向量。
    預設用 fastembed（本地 ONNX，免 API key）；
    若設定了 LOCAL_EMBEDDING_BASE_URL，則改用該本地伺服器（override）。
    """
    base_url = os.environ.get("LOCAL_EMBEDDING_BASE_URL")
    if base_url:
        model = os.environ.get("LOCAL_EMBEDDING_MODEL", "text-embedding-ada-002")
        api_key = os.environ.get("LOCAL_EMBEDDING_API_KEY", "not-needed")
        # 批次呼叫，避免一次塞太多
        out = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            out.extend(_embed_via_local_api(texts[i:i + batch_size], base_url, model, api_key))
        return out
    else:
        return _embed_via_fastembed(texts)


# --------------------------------------------------------------------------
# 索引建立
# --------------------------------------------------------------------------

TARGET_FILENAMES = {"cheatsheet.md", "agent_guide.md"}

def find_cheatsheet_files(root: Path):
    """遞迴尋找符合 <repo_name>/CHEATSHEET.md 樣式的檔案（不分大小寫）。"""
    found = []
    for p in root.rglob("*.md"):
        if p.name.lower() in TARGET_FILENAMES:
            found.append(p)
    return sorted(found)


def guess_repo_name(file_path: Path, root: Path):
    """用檔案所在的第一層目錄名當作 repo 名稱；若就在 root 底下，用檔名去掉副檔名。"""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = file_path
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return file_path.stem


def build_index(root_dir: str, index_path: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"❌ 找不到資料夾：{root}", file=sys.stderr)
        sys.exit(1)

    files = find_cheatsheet_files(root)
    if not files:
        print(f"⚠️ 在 {root} 底下沒找到任何 CHEATSHEET.md / AGENT_GUIDE.md", file=sys.stderr)
        sys.exit(1)

    print(f"📂 找到 {len(files)} 個檔案，開始分塊與 embedding...")

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

    print(f"🧩 共切出 {len(all_chunks)} 個語意區塊，開始呼叫 embedding...")

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

    print(f"✅ 索引已建立：{index_path}（{len(all_chunks)} 筆）")


# --------------------------------------------------------------------------
# 搜尋
# --------------------------------------------------------------------------

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search_index(index_path: str, query: str, top_k: int = 5, repo_filter: str = None):
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

    if repo_filter:
        chunks = [c for c in chunks if c["repo"] == repo_filter]
        if not chunks:
            print(f"⚠️ 找不到 repo 名稱為 '{repo_filter}' 的區塊", file=sys.stderr)
            return

    try:
        q_vec = embed_texts([query])[0]
    except EmbeddingError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    scored = []
    for c in chunks:
        sim = cosine_sim(q_vec, c["embedding"])
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n🔍 查詢：「{query}」　Top {top_k} 結果：\n")
    for rank, (sim, c) in enumerate(scored[:top_k], 1):
        print(f"[{rank}] score={sim:.3f}  repo={c['repo']}  section={c['heading']}")
        preview = c["content"].strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"    {preview}")
        print(f"    來源檔案: {c['file']}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cheat sheet 語意搜尋工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="掃描資料夾並建立索引")
    p_build.add_argument("--root", required=True, help="包含 <repo_name>/CHEATSHEET.md 的根資料夾")
    p_build.add_argument("--index", default="index.json", help="輸出的索引檔路徑")

    p_search = sub.add_parser("search", help="對已建好的索引做語意搜尋")
    p_search.add_argument("--index", default="index.json", help="索引檔路徑")
    p_search.add_argument("--query", required=True, help="搜尋查詢字串")
    p_search.add_argument("--top-k", type=int, default=5, help="回傳前 K 筆結果")
    p_search.add_argument("--repo", default=None, help="只在指定 repo 名稱底下搜尋")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.root, args.index)
    elif args.command == "search":
        search_index(args.index, args.query, args.top_k, args.repo)


if __name__ == "__main__":
    main()
