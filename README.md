# Chained

Thinking cheat sheets write. Playing with [Chained](https://www.youtube.com/watch?v=iO4YnxDHnig)

* **About Chained**

  > Tatsuya Kitani, natori
  >
  > いらないもの

## Cheat sheets

輸入一個 GitHub 倉庫網址，自動生成：
1. **`<repo>_CHEATSHEET.md`** — 適合人類閱讀的快速上手速查表
2. **`<repo>_AGENT_GUIDE.md`** — 適合 LLM agent 閱讀的結構化使用說明
3. **`<repo>_SCANNED_FILES.md`** — 記錄本次生成時實際讀取過哪些檔案（路徑 + 收錄字元數），
   方便追溯 cheatsheet 的資訊來源，也能看出是否因內容過多而被截斷

## 使用前準備

需要 Python 3 和 git。分析部分呼叫**本地模型**（OpenAI 相容介面，例如 vLLM / Ollama / LM Studio），
透過環境變數配置：

```bash
# 本地服務的 base url（注意不含 /chat/completions，腳本會自動拼接）
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1   # vLLM / LM Studio 常見位址
# export LOCAL_LLM_BASE_URL=http://localhost:11434/v1  # Ollama (OpenAI 相容模式)

# 本地服務中載入的模型名稱（需與服務端一致，例如 Ollama 的 "qwen2.5:7b" 這類）
export LOCAL_LLM_MODEL=your-model-name

# 大多數本地服務不校驗 key，不設定也無妨；若您的服務需要鑑權再設定
export LOCAL_LLM_API_KEY=xxx
```

不設定這些環境變數時，預設使用 `http://localhost:8000/v1` 和模型名稱 `local-model`。

## 使用方法

```bash
python main.py --repo-url https://github.com/psf/requests
# 或指定輸出目錄
python main.py --repo-url https://github.com/psf/requests --out-dir ./output
```

## 工作原理

- `collector.py`：使用 `git clone --depth 1` 淺複製倉庫，掃描 README、依賴清單
  （package.json / pyproject.toml / setup.py / Cargo.toml / go.mod 等）、
  examples/docs 目錄下的範例檔案，以及倉庫目錄結構，打包成一份上下文文字。
- `generator.py`：將上下文文字透過 OpenAI 相容介面（`/chat/completions`）發送給本地模型，
  分別以兩套 prompt 生成「人類速查表」與「LLM agent 使用指南」。
- `main.py`：命令列入口，串聯以上兩步並寫入檔案。

## 可調整之處

- `collector.py` 中的 `PRIORITY_FILES` / `INTEREST_DIRS`：可以按需增刪要掃描的檔案類型
- `collector.py` 中的 `MAX_TOTAL_CHARS`：如果倉庫很大、本地模型上下文視窗較小，建議調小
- `generator.py` 中的兩個 PROMPT：可以按您的團隊習慣調整輸出格式
- `generator.py` 中的 `API_BASE_URL` / `MODEL` 預設值：可以直接修改程式碼中的預設值，不一定要用環境變數

## 常見問題

- **連線失敗 / Connection refused**：確認本地模型服務已啟動，且 `LOCAL_LLM_BASE_URL` 指向正確的連接埠
- **回傳內容被截斷**：本地模型的上下文視窗可能較小，試試調小 `collector.py` 的 `MAX_TOTAL_CHARS`，
  或調大 `generator.py` 中的 `max_tokens`
- **模型輸出格式跑偏（尤其是本地小模型）**：可以在 prompt 末尾加入更強的格式約束，或換一個指令遵循能力更好的模型
