# repo-cheatsheet

输入一个 GitHub 仓库地址,自动生成:
1. **`<repo>_CHEATSHEET.md`** — 给人看的快速上手速查表
2. **`<repo>_AGENT_GUIDE.md`** — 给 LLM agent 看的结构化使用说明

## 使用前准备

需要 Python 3 和 git。分析部分调用**本地模型**(OpenAI 兼容接口,如 vLLM / Ollama / LM Studio),
通过环境变量配置:

```bash
# 本地服务的 base url(注意不含 /chat/completions,脚本会自动拼接)
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1   # vLLM / LM Studio 常见地址
# export LOCAL_LLM_BASE_URL=http://localhost:11434/v1  # Ollama (OpenAI 兼容模式)

# 本地服务里加载的模型名(需和服务端一致,比如 Ollama 是 "qwen2.5:7b" 这类)
export LOCAL_LLM_MODEL=your-model-name

# 大多数本地服务不校验 key,不设置也没关系;如果你的服务需要鉴权再设置
export LOCAL_LLM_API_KEY=xxx
```

不设置这些环境变量时,默认使用 `http://localhost:8000/v1` 和模型名 `local-model`。

## 使用方法

```bash
python main.py https://github.com/psf/requests
# 或指定输出目录
python main.py https://github.com/psf/requests --out-dir ./output
```

## 工作原理

- `collector.py`: 用 `git clone --depth 1` 浅克隆仓库,扫描 README、依赖清单
  (package.json / pyproject.toml / setup.py / Cargo.toml / go.mod 等)、
  examples/docs 目录下的示例文件,以及仓库目录结构,打包成一份上下文文本。
- `generator.py`: 把上下文文本通过 OpenAI 兼容接口(`/chat/completions`)发给本地模型,
  分别用两套 prompt 生成"人类速查表"和"LLM agent 使用指南"。
- `main.py`: 命令行入口,串联以上两步并写文件。

## 可调整的地方

- `collector.py` 里的 `PRIORITY_FILES` / `INTEREST_DIRS`:可以按需增删要扫描的文件类型
- `collector.py` 里的 `MAX_TOTAL_CHARS`:如果仓库很大、本地模型上下文窗口较小,建议调小
- `generator.py` 里的两个 PROMPT:可以按你的团队习惯调整输出格式
- `generator.py` 里的 `API_BASE_URL` / `MODEL` 默认值:也可以直接改代码里的默认值,不一定要用环境变量

## 常见问题

- **连接失败 / Connection refused**:确认本地模型服务已启动,且 `LOCAL_LLM_BASE_URL` 指向正确的端口
- **返回内容被截断**:本地模型的上下文窗口可能比较小,试试调小 `collector.py` 的 `MAX_TOTAL_CHARS`,
  或调大 `generator.py` 里 `max_tokens`
- **模型输出格式跑偏(尤其是本地小模型)**:可以在 prompt 末尾加更强的格式约束,或换一个指令遵循能力更好的模型
