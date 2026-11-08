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
3. **`<repo>_SCANNED_FILES.md`** — 記錄本次生成時實際讀取過哪些檔案（路徑 + 收錄字元數與 token 數），
   方便追溯 cheatsheet 的資訊來源；超出 token 預算而整份未收錄的檔案會另列一區
4. **`<repo>_VERIFY.md`** — 檢查上面兩份文件中的 API、參數、旗標是否出現在送給模型的內容裡；
   兩處都找不到的標為 ⚠️ suspect（多半是模型編造或拼錯）。
   文件中每個 `[來源: 檔案路徑]` 也會被檢查：路徑必須是送給模型的檔案，旁邊的識別字必須出現在該檔案的摘錄裡

重新檢查之前產生的文件（需要該倉庫的本地 checkout；若生成時用了 `--focus`，這裡要給同樣的值）：

```bash
python verifier.py --repo source/milvus db/milvus/CHEATSHEET.md db/milvus/AGENT_GUIDE.md
```

## 使用前準備

需要 Python 3 和 git。分析部分呼叫**本地模型**（OpenAI 相容介面，例如 vLLM / Ollama / LM Studio），
透過環境變數配置：

```bash
# 本地服務的 base url（注意不含 /chat/completions，腳本會自動拼接）
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1   # vLLM / LM 

# 本地服務中載入的模型名稱（需與服務端一致，例如 Ollama 的 "qwen2.5:7b" 這類）
export LOCAL_LLM_MODEL=your-model-name

# 大多數本地服務不校驗 key，不設定也無妨；若您的服務需要鑑權再設定
export LOCAL_LLM_API_KEY=xxx

# 模型的 context window（token 數）。送給模型的倉庫內容預算 = 這個值 - 4000（輸出）- 1500（prompt）
export LOCAL_LLM_CONTEXT=32768
```

不設定這些環境變數時，預設使用 `http://localhost:8000/v1`、模型名稱 `local-model` 和 24576 token 的 context window。

## 使用方法

* Generate cheat sheet

  ```bash
  python main.py --repo-url https://github.com/psf/requests
  # 或指定輸出目錄
  python main.py --repo-url https://github.com/psf/requests --out-dir ./output
  # 或使用本地已 checkout 的倉庫（不會 clone，也不會刪除該目錄）
  python main.py --repo source/requests
  ```

  `--repo-url` 與 `--repo` 擇一必填。輸出資料夾以倉庫名（本地目錄名）命名。

* 限定範圍與額外指示（大型 monorepo 建議使用）

  ```bash
  # 只從這些目錄 / 檔案挑選範例與次要檔案（可重複）；README、依賴清單等核心檔案照常讀取
  python main.py --repo source/milvus --focus client/ --focus docs/user_guides/
  # 在兩份 prompt 的最後附加額外指示
  python main.py --repo source/milvus --focus client/ --instructions "只寫 Go client，不要內部開發規範"
  ```

  `--focus` 的路徑不存在於倉庫時會直接報錯。目錄結構也只列出 focus 的路徑。

* [semantic search](docs/semantic_search.md)


## 工作原理

- `collector.py`：使用 `git clone --depth 1` 淺複製倉庫，掃描 README、依賴清單
  （package.json / pyproject.toml / setup.py / Cargo.toml / go.mod 等）、
  examples/docs 目錄下的範例檔案，以及倉庫目錄結構，打包成一份上下文文字。
  打包依 token 預算進行：先放目錄結構，再依序放核心檔案與評分最高的次要檔案；
  某檔案超出剩餘預算時會以剩餘空間重新摘錄，仍放不下就整份略過（記錄在 `SCANNED_FILES.md`），不會把檔案切在一半。
- `generator.py`：將上下文文字透過 OpenAI 相容介面（`/chat/completions`）發送給本地模型，
  分別以兩套 prompt 生成「人類速查表」與「LLM agent 使用指南」；prompt 要求每個指令 / API 附 `[來源: 檔案路徑]`，
  找不到依據的內容列在「待確認」。
- `verifier.py`：不經 LLM，檢查識別字與 `[來源: ...]` 是否站得住腳，寫出 `VERIFY.md`。
- `main.py`：命令列入口，串聯以上步驟並寫入檔案。

## 可調整之處

- `collector.py` 中的 `PRIORITY_FILES` / `INTEREST_DIRS`：可以按需增刪要掃描的檔案類型
- `LOCAL_LLM_CONTEXT`（環境變數）：本地模型的 context window，決定倉庫內容的 token 預算；
  `collector.py` 的 `OUTPUT_TOKENS` / `PROMPT_TOKENS` 是預留給輸出與 prompt 的空間
- 安裝 `tiktoken` 時以它計 token，否則用字元數估算（偏保守）；兩者都和模型自己的 tokenizer 有些差距
- `generator.py` 中的兩個 PROMPT：可以按您的團隊習慣調整輸出格式
- `generator.py` 中的 `API_BASE_URL` / `MODEL` 預設值：可以直接修改程式碼中的預設值，不一定要用環境變數

## 常見問題

- **連線失敗 / Connection refused**：確認本地模型服務已啟動，且 `LOCAL_LLM_BASE_URL` 指向正確的連接埠
- **回傳內容被截斷**：試試調大 `generator.py` 中的 `max_tokens`；
  若是倉庫內容放不下（`SCANNED_FILES.md` 有「Not Included」區塊），
  依模型實際的 context window 設定 `LOCAL_LLM_CONTEXT`，或用 `--focus` 縮小範圍
- **`VERIFY.md` 顯示「No `[來源: path]` marks」**：模型沒有遵守出處規則，換一個指令遵循能力更好的模型，
  或用 `--instructions` 再強調一次
- **模型輸出格式跑偏（尤其是本地小模型）**：可以在 prompt 末尾加入更強的格式約束，或換一個指令遵循能力更好的模型
