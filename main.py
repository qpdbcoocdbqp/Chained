#!/usr/bin/env python3
"""
repo-cheatsheet CLI

Usage:
    python main.py --repo-url <github_repo_url> [--out-dir OUTPUT_DIR]

Example:
    python main.py --repo-url https://github.com/psf/requests
    python main.py --repo-url https://github.com/psf/requests --out-dir ./output

Generates three files:
    CHEATSHEET.md   -- Cheatsheet for humans
    AGENT_GUIDE.md  -- Structured guide for LLM agents
    SCANNED_FILES.md -- List of files scanned and included in the analysis context
"""
import argparse
import os
import sys

from collector import build_context_bundle
from generator import generate_human_cheatsheet, generate_agent_guide


def build_scanned_files_report(repo_url: str, scanned_files: list, truncated: bool) -> str:
    """Generates a Markdown file recording the files actually scanned/read during the cheatsheet generation."""
    lines = [f"# Scanned Files Record: {repo_url}\n"]

    if not scanned_files:
        lines.append("(No matching priority files or example files were found in this run; the cheatsheet might be generated based on the directory structure only)\n")
        return "\n".join(lines)

    priority = [f for f in scanned_files if f["category"] == "priority"]
    examples = [f for f in scanned_files if f["category"] == "example"]

    if priority:
        lines.append("## Core Files (README / Dependencies / Documentation, etc.)\n")
        lines.append("| File Path | Collected Characters |")
        lines.append("|---|---|")
        for f in priority:
            lines.append(f"| `{f['path']}` | {f['chars']} |")
        lines.append("")

    if examples:
        lines.append("## Sampled Files in Examples/Documentation Directories\n")
        lines.append("| File Path | Collected Characters |")
        lines.append("|---|---|")
        for f in examples:
            lines.append(f"| `{f['path']}` | {f['chars']} |")
        lines.append("")

    lines.append(f"A total of {len(scanned_files)} files were read and included in the analysis context.")

    if truncated:
        lines.append(
            "\n⚠️ Warning: The total length of the packaged context exceeded the limit and has been truncated. "
            "The content of the files near the end of the list above might not have been fully sent to the model for analysis. "
            "To cover them completely, you can increase `MAX_TOTAL_CHARS` in `collector.py`."
        )

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate a cheatsheet + LLM agent guide for a GitHub repository")
    parser.add_argument("--repo-url", help="GitHub repository URL, e.g., https://github.com/psf/requests")
    parser.add_argument("--out-dir", default="./db", help="Output directory, defaults to the current directory")
    args = parser.parse_args()

    repo_name = args.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, repo_name), exist_ok=True)

    print(f"[1/4] Cloning and scanning repository: {args.repo_url}")
    try:
        context, scanned_files, truncated = build_context_bundle(args.repo_url)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("[2/4] Generating human cheatsheet (calling local LLM)...")
    try:
        human_doc = generate_human_cheatsheet(context)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("[3/4] Generating LLM agent guide (calling local LLM)...")
    try:
        agent_doc = generate_agent_guide(context)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    human_path = os.path.join(out_dir, repo_name, "CHEATSHEET.md")
    agent_path = os.path.join(out_dir, repo_name, "AGENT_GUIDE.md")
    scanned_path = os.path.join(out_dir, repo_name, "SCANNED_FILES.md")

    with open(human_path, "w", encoding="utf-8") as f:
        f.write(human_doc)
    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(agent_doc)
    with open(scanned_path, "w", encoding="utf-8") as f:
        f.write(build_scanned_files_report(args.repo_url, scanned_files, truncated))

    print(
        f"\nDone!\n"
        f"  Human cheatsheet: {human_path}\n"
        f"  Agent guide: {agent_path}\n"
        f"  Scanned files record: {scanned_path}"
    )



if __name__ == "__main__":
    main()
