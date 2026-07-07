"""
generator.py
把 collector 打包好的上下文文本丢给本地模型(OpenAI 兼容接口, 如 vLLM / Ollama / LM Studio),
生成 (1) 人类速查表 (2) LLM agent 可解析版本
"""
import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

# 本地模型的 OpenAI 兼容接口地址, 可用环境变量覆盖
# 常见默认值:
#   vLLM / LM Studio: http://localhost:8000/v1 或 http://localhost:1234/v1
#   Ollama (OpenAI 兼容模式): http://localhost:11434/v1
API_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
API_URL = API_BASE_URL.rstrip("/") + "/chat/completions"
MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
# 大多数本地模型服务不校验 key，但接口格式要求这个字段存在，给个占位符即可；
# 如果你的服务需要真实 key，用 LOCAL_LLM_API_KEY 环境变量传入
API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")

HUMAN_CHEATSHEET_PROMPT = """你是一个资深工程师,请根据下面提供的仓库信息,写一份简洁的"速查表"(cheat sheet),
目标读者是第一次接触这个项目、想快速上手的开发者。

要求:
- 用 Markdown 格式
- 包含: 项目一句话简介 / 安装方式 / 3-6 个最常用的命令或 API 调用示例(附代码块)/ 常见坑或注意事项(如果有)
- 内容尽量精炼,像一页纸能看完的速查表,不要写成完整教程
- 不要编造仓库中没有出现的信息,如果信息不足,如实说明

仓库信息如下:
{context}
"""

AGENT_GUIDE_PROMPT = """你是一个为 LLM agent 准备工具说明文档的助手。请根据下面提供的仓库信息,
生成一份结构化的 Markdown 文档,目的是让另一个 LLM agent 读了之后,能够知道:
1. 这个仓库/工具能做什么
2. 有哪些可调用的功能/命令/API(尽量列出函数名、CLI 子命令名、或 API endpoint)
3. 每个功能的参数是什么、返回什么
4. 一个可以直接照抄使用的调用范例

请用如下结构输出(严格按此结构,方便程序解析):

# AGENT GUIDE: <repo 名称>

## Summary
(一两句话描述这个仓库的用途)

## Installation
(安装/初始化步骤,简洁的命令)

## Capabilities
对每个功能,用以下格式列出:

### <功能名>
- description: ...
- invocation: `具体调用方式,如命令行或函数签名`
- parameters: ...
- returns: ...
- example:
```
具体示例代码或命令
```

## Notes
(任何 LLM agent 需要注意的限制、依赖环境变量、认证方式等)

不要编造仓库中没有出现的功能。如果信息不足以确定某个细节,请明确写"未在仓库信息中找到"。

仓库信息如下:
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
        raise RuntimeError(f"本地模型请求失败: {e.code} {e.read().decode('utf-8')}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接到本地模型服务 ({API_URL}): {e.reason}\n"
            f"请确认服务已启动,或用 LOCAL_LLM_BASE_URL 环境变量指定正确地址"
        )

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"本地模型返回格式异常: {json.dumps(data, ensure_ascii=False)[:500]}")


def generate_human_cheatsheet(context: str) -> str:
    prompt = HUMAN_CHEATSHEET_PROMPT.format(context=context)
    return _call_local_llm(prompt)


def generate_agent_guide(context: str) -> str:
    prompt = AGENT_GUIDE_PROMPT.format(context=context)
    return _call_local_llm(prompt)
