# Semantic search

```bash
# install package
uv pip install fastembed lancedb

# download embed model
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

```bash
# 本地檔案模式 (不需要 MinIO,先驗證流程能跑)
# build
python semantic_search.py build --root db/codebase-memory-mcp --backend lancedb --uri ./storage
python semantic_search.py build --root db/milvus --backend lancedb --uri ./storage
# search
python semantic_search.py search --backend lancedb --uri ./storage --query "vector search" --top-k 3
python semantic_search.py search --backend lancedb --uri ./storage --query "code memory" --top-k 3
```
