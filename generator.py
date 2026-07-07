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

HUMAN_CHEATSHEET_PROMPT = """你是一個資深工程師，請根據下面提供的倉庫資訊，寫一份專注於「指令操作與 API 呼叫」的簡潔"速查表"(cheat sheet)。
目標讀者是已經配置好環境、想快速上手進行開發或運維的開發者。

請完全忽略任何關於安裝、部屬、初始化或環境設定的內容。

要求:
- 用 Markdown 格式，多用表格、粗體與程式碼區塊，確保極高可讀性
- 包含: 
  1. 倉庫一句話核心用途簡介
  2. 常用指令 / API 分類速查（請用表格呈現：欄位包含指令/API、核心參數、功能說明）
  3. 3-5 個高頻實戰場景的代碼片段或指令組合技（附程式碼區塊）
  4. 操作時的常見坑或注意事項（如果有）
- 內容盡量精煉，像一頁紙能看完的速查表，去除所有過渡性廢話，直接給出核心乾貨
- 不要編造倉庫中沒有出現的信息，如果信息不足，如實說明
- 請使用繁體中文輸出

倉庫資訊如下:
{context}
"""

AGENT_GUIDE_PROMPT = """你是一個為 LLM agent 準備工具說明文檔的助手。請根據下面提供的倉庫資訊，
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
- example:

```

具體示例代碼或指令操作

```

## Operational Notes
(任何 LLM agent 在執行指令時需要注意的限制、依賴的環境變數、認證權限、或高風險的操作限制)

不要編造倉庫中沒有出現的功能。如果資訊不足以確定某個操作細節，請明確寫"未在倉庫資訊中找到"。
請使用繁體中文輸出。

倉庫資訊如下:
{context}
"""


def _call_local_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "max_tokens": 4000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
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


def generate_human_cheatsheet(context: str) -> str:
    prompt = HUMAN_CHEATSHEET_PROMPT.format(context=context)
    return _call_local_llm(prompt)


def generate_agent_guide(context: str) -> str:
    prompt = AGENT_GUIDE_PROMPT.format(context=context)
    return _call_local_llm(prompt)
