"""
generator.py
Sends the context text packaged by collector to a local LLM (OpenAI-compatible interface, e.g., vLLM / Ollama / LM Studio),
and generates (1) Human Cheatsheet (2) LLM agent readable guide.
"""
import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible API base URL of the local LLM, can be overridden by environment variable
# Common defaults:
#   vLLM / LM Studio: http://localhost:8000/v1 or http://localhost:1234/v1
#   Ollama (OpenAI-compatible mode): http://localhost:11434/v1
API_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
API_URL = API_BASE_URL.rstrip("/") + "/chat/completions"
MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
# Most local model services do not validate the API key, but the API format requires this field, so a placeholder is provided.
# If your service requires a real key, pass it via the LOCAL_LLM_API_KEY environment variable.
API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")

# The repository context comes first and is identical in every call, so the
# second call reuses the KV cache of the first one (vLLM automatic prefix
# caching, llama.cpp server / Ollama prompt cache).  Only the task prompt after
# it differs.
CONTEXT_PREFIX = """倉庫資訊如下:
{context}

---

"""

HUMAN_CHEATSHEET_PROMPT = """你是一個資深工程師，請根據上面提供的倉庫資訊，寫一份專注於「指令操作與 API 呼叫」的簡潔"速查表"(cheat sheet)。
目標讀者是已經配置好環境、想快速上手進行開發或運維的開發者。

請完全忽略任何關於安裝、部屬、初始化或環境設定的內容。

要求:
- 用 Markdown 格式，多用表格、粗體與程式碼區塊，確保極高可讀性
- 包含:
  1. 倉庫一句話核心用途簡介
  2. 常用指令 / API 分類速查（請用表格呈現：欄位包含指令/API、核心參數、功能說明、來源）
  3. 3-5 個高頻實戰場景的代碼片段或指令組合技（附程式碼區塊）
  4. 操作時的常見坑或注意事項（如果有）
  5. 待確認（見下方出處規則；沒有就省略這一段）
- 內容盡量精煉，像一頁紙能看完的速查表，去除所有過渡性廢話，直接給出核心乾貨
- 不要編造倉庫中沒有出現的信息，如果信息不足，如實說明
- 出處規則:
  - 每個指令、API、參數後面用 [來源: <檔案路徑>] 標出它出現在哪個檔案（表格放在「來源」欄，
    每一格都要寫成 [來源: <檔案路徑>]，連同方括號與「來源:」，不要只寫路徑；
    路徑照抄倉庫資訊中 `## File:` 或 `## Example File:` 標題後的路徑）
  - 倉庫資訊中找不到依據的內容不要寫進表格，改列在最後的「待確認」段落
- 請使用繁體中文輸出
"""

AGENT_GUIDE_PROMPT = """你是一個為 LLM agent 準備工具說明文檔的助手。請根據上面提供的倉庫資訊，
生成一份結構化的 Markdown 文檔，目的是讓另一個 LLM agent 讀了之後，能夠知道如何正確調用此工具。

請完全忽略任何關於安裝（Installation）、環境配置或部屬的事項。專注於可操作的介面與參數。

請用如下結構輸出 (嚴格按此結構，方便程式解析):

# AGENT GUIDE: <repo 名稱>

## Summary
(一兩句話描述這個倉庫的用途與核心操作場景)

## Capabilities
對每個可操作的功能、命令或 API（如函數名、CLI 子命令名、或 API endpoint），用以下格式列出：

### <功能名>
- description: (該操作的具體用途)
- invocation: `具體呼叫或執行方式，如命令列或函數簽名`
- parameters: (詳細列出必要的參數、選填的 Flags 及其型態/意義)
- returns: (執行後的返回數據、輸出格式或預期結果)
- source: [來源: <檔案路徑>]
- example:

```

具體示例代碼或指令操作

```

## Operational Notes
(任何 LLM agent 在執行指令時需要注意的限制、依賴的環境變數、認證權限、或高風險的操作限制)

## 待確認
(倉庫資訊中找不到依據、無法確定的功能或細節；沒有就寫"無")

不要編造倉庫中沒有出現的功能。如果資訊不足以確定某個操作細節，請明確寫"未在倉庫資訊中找到"。
每個功能的 source 填它出現的檔案，路徑照抄倉庫資訊中 `## File:` 或 `## Example File:` 標題後的路徑；
找不到依據的功能不要列為 Capability，改列在「待確認」。
請使用繁體中文輸出。
"""

# Appended after the task prompt, so the context before it stays identical in every call (prefix cache)
INSTRUCTIONS_TEMPLATE = """
額外指示（使用者指定，請在遵守以上格式與規則的前提下照做）:
{instructions}
"""


def _call_local_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "max_tokens": 4000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
        # Thinking off: passed to the chat template by llama.cpp server / vLLM
        # (Qwen3, Gemma 4 templates); servers that do not know it ignore it
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Local LLM request failed: {e.code} {e.read().decode('utf-8')}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to connect to local LLM service ({API_URL}): {e.reason}\n"
            f"Please ensure the service is running, or specify the correct address using the LOCAL_LLM_BASE_URL environment variable."
        )

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Invalid format returned from local LLM: {json.dumps(data, ensure_ascii=False)[:500]}")


def _task(prompt: str, instructions: str) -> str:
    instructions = instructions.strip()
    return prompt + INSTRUCTIONS_TEMPLATE.format(instructions=instructions) if instructions else prompt


def generate_human_cheatsheet(context: str, instructions: str = "") -> str:
    return _call_local_llm(CONTEXT_PREFIX.format(context=context) + _task(HUMAN_CHEATSHEET_PROMPT, instructions))


def generate_agent_guide(context: str, instructions: str = "") -> str:
    return _call_local_llm(CONTEXT_PREFIX.format(context=context) + _task(AGENT_GUIDE_PROMPT, instructions))
