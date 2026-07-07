#!/usr/bin/env python3
"""
repo-cheatsheet CLI

用法:
    python main.py <github_repo_url> [--out-dir OUTPUT_DIR]

示例:
    python main.py https://github.com/psf/requests
    python main.py https://github.com/psf/requests --out-dir ./output

生成两份文件:
    CHEATSHEET.md   -- 给人看的速查表
    AGENT_GUIDE.md  -- 给 LLM agent 看的结构化说明
"""
import argparse
import os
import sys

from collector import build_context_bundle
from generator import generate_human_cheatsheet, generate_agent_guide


def main():
    parser = argparse.ArgumentParser(description="生成 GitHub 仓库的速查表 + LLM agent 使用指南")
    parser.add_argument("--repo-url", default="https://github.com/psf/requests", help="GitHub 仓库地址,例如 https://github.com/psf/requests")
    parser.add_argument("--out-dir", default=".", help="输出目录,默认当前目录")
    args = parser.parse_args()

    repo_name = args.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, repo_name), exist_ok=True)

    print(f"[1/3] 正在克隆并扫描仓库: {args.repo_url}")
    try:
        context = build_context_bundle(args.repo_url)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("[2/3] 正在生成人类速查表 (调用 Claude API)...")
    try:
        human_doc = generate_human_cheatsheet(context)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("[3/3] 正在生成 LLM agent 使用指南 (调用 Claude API)...")
    try:
        agent_doc = generate_agent_guide(context)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    human_path = os.path.join(out_dir, repo_name, "CHEATSHEET.md")
    agent_path = os.path.join(out_dir, repo_name, "AGENT_GUIDE.md")

    with open(human_path, "w", encoding="utf-8") as f:
        f.write(human_doc)
    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(agent_doc)

    print(f"\n完成!\n  人类速查表: {human_path}\n  Agent 指南: {agent_path}")


if __name__ == "__main__":
    main()
