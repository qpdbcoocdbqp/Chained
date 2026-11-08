#!/usr/bin/env python3
"""
verifier.py — Check that the APIs, flags and parameters a generated document
mentions appear in the context the LLM was given.  No LLM is involved.

  Checked   inline code (`create_collection()`, `--top-k`), and in code blocks
            the names called (foo(), obj.method()), the keyword arguments
            (limit=2) and the flags (--force).  Only tokens that look like code
            (snake_case, CamelCase, digits, dotted, flags) are checked: plain
            words such as search() or print() say nothing.
  Found     a token is found when it occurs in the bundle as a whole word; a
            dotted name a.b.c when its last part does (a is usually a variable
            the example made up).  Names the document assigns or defines
            itself (query_vectors = ..., def fetch_data()) are skipped.
  Repo      with a local checkout, tokens missing from the bundle are looked up
            in the whole repo, which tells "the model was not shown it" from
            "the repo does not have it".
  Citations the [來源: path] marks the prompts ask for: the path must be a file
            of the bundle, and the identifiers next to the mark (the inline code
            of its line, or of the invocation line of its section) must occur
            in that file's excerpt.

Usage (re-check documents generated earlier against a local checkout; give the
--focus the documents were generated with, so that the bundle is the same):
    python verifier.py --repo source/milvus db/milvus/CHEATSHEET.md db/milvus/AGENT_GUIDE.md
"""
import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from collector import _FENCE, _INLINE_CODE, _TOKEN, Repo, _technical, bundle_repo

_CALL = re.compile(r"(?<![\w.])((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\s*\(")
_KWARG = re.compile(r"[(,]\s*([A-Za-z_]\w*)=(?!=)")
_FLAG = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*")
_ASSIGNED = re.compile(r"^\s*(?:(?:var|let|const|val)\s+)?([A-Za-z_]\w*)(?:\s*,\s*[A-Za-z_]\w*)*\s*(?::=|=(?!=))"
                       r"|\b(?:def|func|function|fn|class|struct)\s+([A-Za-z_]\w*)", re.M)
_WORD = re.compile(r"[A-Za-z_]\w*")
_CODE_WORDS = {"__init__", "__name__", "__main__"}  # Python dunders a snippet may use without the repo naming them

_CITATION = re.compile(r"\[來源[:：]\s*([^\]]+?)\s*\]")
_PATH_SEPARATOR = re.compile(r"[,，、;；]")
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
_SECTION_HEADING = re.compile(r"#{2,4}\s")


@dataclass
class Finding:
    token: str
    in_bundle: bool
    repo_path: str | None = None   # first repo file containing it, when a checkout was given

    @property
    def status(self) -> str:
        if self.in_bundle:
            return "ok"
        if self.repo_path:
            return "not in context"     # real, but the model guessed it from elsewhere
        return "suspect"                # nowhere to be found: likely made up


def doc_tokens(doc: str) -> list[str]:
    """Code-like tokens of a document, in order of first appearance."""
    blocks = [m.group(2) for m in _FENCE.finditer(doc)]
    prose = _FENCE.sub("", doc)
    assigned = {m.group(1) or m.group(2) for b in blocks for m in _ASSIGNED.finditer(b)}

    tokens = []
    for code in _INLINE_CODE.findall(prose):
        tokens += _TOKEN.findall(code)
    for block in blocks:
        for line in block.split("\n"):
            tokens += _CALL.findall(line) + _KWARG.findall(line) + _FLAG.findall(line)
    tokens = [t for t in tokens if _technical(t) and t not in assigned and t not in _CODE_WORDS]
    return list(dict.fromkeys(tokens))


def _key(token: str) -> str:
    """What to look for: a flag as is, else the last part of a dotted name."""
    return token if token.startswith("-") else token.rsplit(".", 1)[-1]


def _vocabulary(text: str) -> set[str]:
    return set(_WORD.findall(text)) | set(_FLAG.findall(text))


def verify(doc: str, bundle: str, repo: Repo | None = None) -> list[Finding]:
    vocab = _vocabulary(bundle)
    findings = [Finding(t, _key(t) in vocab) for t in doc_tokens(doc)]
    missing = {_key(f.token): f for f in findings if not f.in_bundle}
    if repo is not None and missing:
        for rel in repo.files:
            text = repo.read(rel)
            if not text:
                continue
            for key in [k for k in missing if missing[k].repo_path is None]:
                if key in _vocabulary(text):
                    missing[key].repo_path = rel.replace(os.sep, "/")
            if all(f.repo_path for f in missing.values()):
                break
        # Dotted names sharing a last part share the lookup
        for f in findings:
            if not f.in_bundle:
                f.repo_path = missing[_key(f.token)].repo_path
    return findings


@dataclass(frozen=True)
class CitationProblem:
    path: str                       # the cited path(s)
    token: str | None               # the identifier looked up; None when the path itself is wrong
    status: str                     # "unknown file" | "not in excerpt"
    elsewhere: str | None = None    # another bundle file whose excerpt has the identifier


@dataclass
class CitationCheck:
    citations: int = 0              # [來源: ...] marks in the document
    checked: int = 0                # identifiers looked up in a cited file
    problems: list[CitationProblem] = field(default_factory=list)


def _norm_path(path: str) -> str:
    return _LINE_SUFFIX.sub("", path.strip().strip("`'\"")).replace("\\", "/").removeprefix("./")


def _citation_tokens(lines: list[str], i: int) -> list[str]:
    """The identifiers the citation on lines[i] is about: the inline code of its line, else that of the
    `- invocation:` line of its section (an agent guide capability)."""
    def inline(line: str) -> list[str]:
        return [t for code in _INLINE_CODE.findall(_CITATION.sub("", line))
                for t in _TOKEN.findall(code) if _technical(t)]

    tokens = inline(lines[i])
    if not tokens:
        start = i
        while start > 0 and not _SECTION_HEADING.match(lines[start]):
            start -= 1
        for line in lines[start + 1:]:
            if _SECTION_HEADING.match(line):
                break
            if "invocation" in line.lower():
                tokens = inline(line)
                break
    return list(dict.fromkeys(tokens))


def check_citations(doc: str, scanned_files: list[dict]) -> CitationCheck:
    """Check the [來源: path] marks of `doc` against `scanned_files`, the files of the bundle (each a dict with
    "path" and "text", as bundle_repo returns them)."""
    vocabularies = {_norm_path(f["path"]): _vocabulary(f["text"]) for f in scanned_files}
    lines = doc.split("\n")
    check = CitationCheck()
    problems = []
    for i, line in enumerate(lines):
        for m in _CITATION.finditer(line):
            check.citations += 1
            cited = [_norm_path(p) for p in _PATH_SEPARATOR.split(m.group(1)) if p.strip()]
            problems += [CitationProblem(p, None, "unknown file") for p in cited if p not in vocabularies]
            known = [p for p in cited if p in vocabularies]
            if not known:
                continue
            for token in _citation_tokens(lines, i):
                check.checked += 1
                key = _key(token)
                if not any(key in vocabularies[p] for p in known):
                    elsewhere = next((p for p, v in vocabularies.items() if key in v), None)
                    problems.append(CitationProblem(", ".join(known), token, "not in excerpt", elsewhere))
    check.problems = list(dict.fromkeys(problems))
    return check


def _citation_lines(check: CitationCheck) -> list[str]:
    if not check.citations:
        return ["### Citations\n",
                "No `[來源: path]` marks: the model did not follow the citation rule, so nothing could be checked.\n"]
    lines = ["### Citations\n",
             f"{check.citations} citations, {check.checked} identifiers looked up in the cited files, "
             f"{len(check.problems)} problems.\n"]
    if check.problems:
        lines += ["| Cited | Identifier | Status |", "|---|---|---|"]
        for p in check.problems:
            if p.status == "unknown file":
                status = "⚠️ not a file of the context"
            else:
                status = "not in the cited excerpt" + (f" (found in `{p.elsewhere}`)" if p.elsewhere else "")
            lines.append(f"| `{p.path}` | {f'`{p.token}`' if p.token else '—'} | {status} |")
        lines.append("")
    return lines


def build_report(results: dict[str, list[Finding]], repo_checked: bool,
                 citations: dict[str, CitationCheck] | None = None) -> str:
    lines = ["# Verification Report\n",
             "Code-like tokens in the generated documents (inline code, names called, keyword arguments, "
             "flags) checked against the context sent to the LLM"
             + (", and the `[來源: path]` citations against the files they name.\n" if citations else ".\n")]
    for name, findings in results.items():
        bad = [f for f in findings if not f.in_bundle]
        lines.append(f"## {name}\n")
        lines.append(f"{len(findings) - len(bad)} / {len(findings)} tokens found in the context.\n")
        if bad and repo_checked:
            lines += ["| Token | Found in repo | Status |", "|---|---|---|"]
            lines += [f"| `{f.token}` | {f'`{f.repo_path}`' if f.repo_path else '—'} | "
                      f"{'⚠️ suspect' if f.status == 'suspect' else 'not in context'} |" for f in bad]
        elif bad:
            lines += ["| Token | Status |", "|---|---|"]
            lines += [f"| `{f.token}` | ⚠️ not in context |" for f in bad]
        if bad:
            lines.append("")
        if citations and name in citations:
            lines += _citation_lines(citations[name])
    if repo_checked:
        lines.append("**suspect**: in neither the context nor the repo, likely made up by the model. "
                     "**not in context**: exists in the repo, but the model was not shown it.")
    else:
        lines.append("Tokens not in the context may be made up by the model, or come from its own knowledge "
                     "of the project. Re-run with `python verifier.py --repo <checkout> ...` to tell them apart.")
    return "\n".join(lines) + "\n"


def summary(results: dict[str, list[Finding]], citations: dict[str, CitationCheck] | None = None) -> str:
    parts = [f"{name}: {sum(not f.in_bundle for f in fs)} / {len(fs)} tokens not in context"
             + (f", {len(citations[name].problems)} citation problems in {citations[name].citations} citations"
                if citations and name in citations else "")
             for name, fs in results.items()]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Check generated cheatsheets against the repository context")
    parser.add_argument("docs", nargs="+", help="Generated Markdown files, e.g. db/milvus/CHEATSHEET.md")
    parser.add_argument("--repo", required=True, help="Local checkout of the repository the documents describe")
    parser.add_argument("--focus", action="append", default=[],
                        help="The --focus main.py was run with (repeatable); it changes which files the LLM saw")
    parser.add_argument("--out", help="Report path, defaults to VERIFY.md next to the first document")
    args = parser.parse_args()

    repo = Repo(args.repo)
    bundle, scanned_files, _ = bundle_repo(repo, repo.name, focus=args.focus)
    texts = {Path(d).name: Path(d).read_text(encoding="utf-8") for d in args.docs}
    results = {name: verify(text, bundle, repo) for name, text in texts.items()}
    citations = {name: check_citations(text, scanned_files) for name, text in texts.items()}

    out = args.out or str(Path(args.docs[0]).with_name("VERIFY.md"))
    Path(out).write_text(build_report(results, repo_checked=True, citations=citations), encoding="utf-8")
    print(summary(results, citations))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
