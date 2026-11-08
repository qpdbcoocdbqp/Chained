#!/usr/bin/env python3
"""
repo-cheatsheet CLI

Usage:
    python main.py (--repo-url <github_repo_url> | --repo <local_repo_path>) [--out-dir OUTPUT_DIR]
                   [--focus PATH ...] [--instructions TEXT]

Example:
    python main.py --repo-url https://github.com/psf/requests
    python main.py --repo source/requests
    python main.py --repo-url https://github.com/psf/requests --out-dir ./output
    python main.py --repo source/milvus --focus client/ --instructions "Only the Go client, no internal development rules"

Generates four files:
    CHEATSHEET.md   -- Cheatsheet for humans
    AGENT_GUIDE.md  -- Structured guide for LLM agents
    SCANNED_FILES.md -- List of files scanned and included in the analysis context
    VERIFY.md       -- Whether the identifiers and [來源: path] citations in the documents hold up
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

from collector import Repo, bundle_repo, context_token_budget, count_tokens
from generator import generate_human_cheatsheet, generate_agent_guide
from verifier import build_report, check_citations, summary, verify


def build_scanned_files_report(repo_url: str, scanned_files: list, skipped: list, total_tokens: int,
                               budget: int, focus: tuple = ()) -> str:
    """Generates a Markdown file recording the files actually scanned/read during the cheatsheet generation."""
    lines = [f"# Scanned Files Record: {repo_url}\n"]
    if focus:
        lines.append(f"Focus: {', '.join(f'`{f}`' for f in focus)}\n")
    lines.append(f"Context size: {total_tokens} of a {budget} token budget.\n")

    if not scanned_files:
        lines.append("(No matching priority files or example files were found in this run; the cheatsheet might be generated based on the directory structure only)\n")
        return "\n".join(lines)

    priority = [f for f in scanned_files if f["category"] == "priority"]
    examples = [f for f in scanned_files if f["category"] == "example"]

    if priority:
        lines.append("## Core Files (README / Dependencies / Documentation, etc.)\n")
        lines.append("| File Path | Collected Characters | Tokens |")
        lines.append("|---|---|---|")
        for f in priority:
            lines.append(f"| `{f['path']}` | {f['chars']} | {f['tokens']} |")
        lines.append("")

    if examples:
        lines.append("## Sampled Files in Examples/Documentation Directories\n")
        lines.append("| File Path | Collected Characters | Tokens |")
        lines.append("|---|---|---|")
        for f in examples:
            lines.append(f"| `{f['path']}` | {f['chars']} | {f['tokens']} |")
        lines.append("")

    lines.append(f"A total of {len(scanned_files)} files were read and included in the analysis context.")

    if skipped:
        lines.append("\n## Not Included (over the token budget)\n")
        lines.append("| File Path | Tokens it would have taken |")
        lines.append("|---|---|")
        for f in skipped:
            lines.append(f"| `{f['path']}` | {f['tokens'] if f['tokens'] is not None else '—'} |")
        lines.append(
            f"\n⚠️ Warning: {len(skipped)} selected files did not fit the context budget and were left out whole "
            "(none was cut in the middle). To include them, raise `LOCAL_LLM_CONTEXT` if the model's window allows, "
            "or narrow the scope with `--focus`."
        )

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate a cheatsheet + LLM agent guide for a GitHub repository")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-url", help="GitHub repository URL, e.g., https://github.com/psf/requests")
    source.add_argument("--repo", help="Path to a local repository checkout, e.g., source/milvus")
    parser.add_argument("--out-dir", default="./db", help="Output directory, defaults to ./db")
    parser.add_argument("--focus", action="append", default=[], metavar="PATH",
                        help="Only pick example/secondary files under this repo-relative directory or file "
                             "(repeatable, e.g., --focus client/ --focus docs/user_guide/); README and "
                             "dependency files are read as usual")
    parser.add_argument("--instructions", default="",
                        help="Extra instructions added to the end of both prompts, "
                             "e.g., \"Only the Python SDK, no internal development rules\"")
    args = parser.parse_args()

    if args.repo:
        local = Path(args.repo).resolve()
        if not local.is_dir():
            parser.error(f"--repo: not a directory: {args.repo}")
        repo_name, title = local.name, str(local)
        open_repo = contextlib.nullcontext(Repo(local))     # a local checkout is left as it is
        step = f"Scanning local repository: {local}"
    else:
        repo_name, title = args.repo_url.rstrip("/").split("/")[-1].replace(".git", ""), args.repo_url
        open_repo = Repo.clone(args.repo_url)
        step = f"Cloning and scanning repository: {args.repo_url}"
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, repo_name), exist_ok=True)

    # The checkout stays until the documents are verified against the whole repo
    print(f"[1/5] {step}")
    try:
        budget = context_token_budget()
        with open_repo as repo:
            context, scanned_files, skipped = bundle_repo(repo, title, focus=args.focus, max_tokens=budget)
            if not scanned_files:
                # From the directory tree alone the LLM writes the cheatsheet from file names
                raise RuntimeError("No files were selected for the context (or none fit the token budget); "
                                   "not calling the LLM.")

            print("[2/5] Generating human cheatsheet (calling local LLM)...")
            human_doc = generate_human_cheatsheet(context, args.instructions)

            print("[3/5] Generating LLM agent guide (calling local LLM)...")
            agent_doc = generate_agent_guide(context, args.instructions)

            print("[4/5] Verifying generated documents against the repository...")
            docs = {"CHEATSHEET.md": human_doc, "AGENT_GUIDE.md": agent_doc}
            results = {name: verify(doc, context, repo) for name, doc in docs.items()}
            citations = {name: check_citations(doc, scanned_files) for name, doc in docs.items()}
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    human_path = os.path.join(out_dir, repo_name, "CHEATSHEET.md")
    agent_path = os.path.join(out_dir, repo_name, "AGENT_GUIDE.md")
    scanned_path = os.path.join(out_dir, repo_name, "SCANNED_FILES.md")
    verify_path = os.path.join(out_dir, repo_name, "VERIFY.md")

    with open(human_path, "w", encoding="utf-8") as f:
        f.write(human_doc)
    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(agent_doc)
    with open(scanned_path, "w", encoding="utf-8") as f:
        f.write(build_scanned_files_report(title, scanned_files, skipped, count_tokens(context), budget,
                                           tuple(args.focus)))
    with open(verify_path, "w", encoding="utf-8") as f:
        f.write(build_report(results, repo_checked=True, citations=citations))

    print(
        f"\n[5/5] Done!\n"
        f"  Human cheatsheet: {human_path}\n"
        f"  Agent guide: {agent_path}\n"
        f"  Scanned files record: {scanned_path}\n"
        f"  Verification report: {verify_path} ({summary(results, citations)})"
    )



if __name__ == "__main__":
    main()
