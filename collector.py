"""
collector.py — Clone a repository, pick the files worth reading and pack
excerpts of them into one context text for the cheatsheet LLM.

  Repo          a local checkout: its files, their text, a directory tree
  FileSelector  priority files (PRIORITY_FILES at the root or one level down)
                and secondary files: every file is scored by the model in
                model/ and the best MAX_SECONDARY_FILES are kept (with
                per-directory caps and slots reserved for API and example code
                when the model is a path-only FileScorer)
  Excerpter     cuts each file to its character budget by keeping what a user
                needs rather than its first N characters
  Language      per-language import lines and public declarations, for
                Excerpter's code excerpts

build_context_bundle() runs them in order on a fresh clone.
"""
import ast
import contextlib
import functools
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from model.file_features import extract_features, features, features_to_vector
from model.ref_counts import build_ref_counts
from model.train_ml_scorer import FileScorer

SCORER_MODEL = "nn_scorer.pt"     # model/nn; a FileScorer .pkl also works

# High priority filenames (case-insensitive matching)
PRIORITY_FILES = [
    "readme.md", "readme", "readme.rst", "readme.txt",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "cargo.toml", "go.mod", "gemfile", "pom.xml", "build.gradle",
    "contributing.md", "usage.md", "getting_started.md", "quickstart.md",
    "docs/readme.md",
]
IGNORE_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv"}

MAX_FILE_CHARS = 6000       # Character budget of a priority file
SECONDARY_FILE_CHARS = 2048  # Character budget of a secondary file
MAX_TOTAL_CHARS = 60000     # Upper limit of total packaged content to avoid exceeding context
MAX_READ_BYTES = 512 * 1024

MAX_SECONDARY_FILES = 20    # Maximum number of secondary files to include
SECONDARY_THRESHOLD = 0.5   # Minimum score to be considered
MAX_PER_PARENT_DIR = 3      # e.g. at most 3 files from one examples/<name>/
MAX_PER_TOP_DIR = 5         # e.g. at most 5 files from examples/ overall
# Slots reserved for source code; docs outscore code on rule_score, so without
# a quota READMEs take every slot.  Split so that example code (which already
# shares examples/ with READMEs) cannot also take the public-API slots.
MIN_API_CODE_FILES = 3      # public headers / package entry files, most-imported first
MIN_EXAMPLE_CODE_FILES = 2  # runnable samples
SCORE_TIE_DECIMALS = 2      # model probabilities saturate near 1.0; treat close scores as ties
RULE_SCORE_BUCKET = 0.05    # rule scores this close are compared on content
STUB_FILE_BYTES = 300       # READMEs smaller than this are usually placeholders

DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".adoc"}
YAML_SUFFIXES = {".yaml", ".yml"}
SCRIPT_SUFFIXES = {".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1"}
C_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cu", ".m", ".mm"}
JS_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
CODE_SUFFIXES = features.exts["code"] | C_SUFFIXES
TABLE_ROWS = 8              # rows kept from a long markdown table
LINE_CHARS = 200            # longer lines are cut


def _is_example(feats: dict) -> bool:
    return bool(feats["in_examples"] or feats["is_example_file"])


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class Repo:
    """A local checkout; files under IGNORE_DIRS and dot-dirs are left out.
    `name` is the repository name (the checkout directory's by default)."""

    def __init__(self, root: str | Path, name: str | None = None):
        self.root = Path(root)
        self.name = name or self.root.name

    @classmethod
    @contextlib.contextmanager
    def clone(cls, url: str) -> Iterator["Repo"]:
        """Shallow clone into a temporary directory, removed on exit."""
        tmp_dir = tempfile.mkdtemp(prefix="repo_cheatsheet_")
        print(f"Cloning {url} -> {tmp_dir}")
        try:
            result = subprocess.run(["git", "clone", "--depth", "1", url, tmp_dir], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr}")
            yield cls(tmp_dir, url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _walk(self) -> Iterator[tuple[str, list[str], list[str]]]:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            yield dirpath, dirnames, filenames

    @functools.cached_property
    def files(self) -> list[str]:
        """Relative paths of all files."""
        return [os.path.relpath(os.path.join(d, f), self.root) for d, _, names in self._walk() for f in names]

    def read(self, rel: str, limit: int = MAX_READ_BYTES) -> str | None:
        """The first `limit` bytes of a file as text, or None for binary / unreadable files."""
        try:
            with open(self.root / rel, "rb") as f:
                data = f.read(limit)
        except OSError:
            return None
        return None if b"\x00" in data[:4096] else data.decode("utf-8", errors="ignore").replace("\r\n", "\n")

    def tree(self, max_depth: int = 2) -> str:
        """Directory structure (limited depth) to help the LLM understand the project layout"""
        lines = []
        root_depth = str(self.root).rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in self._walk():
            depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            if depth > max_depth:
                dirnames[:] = []
                continue
            indent = "  " * depth
            lines.append(f"{indent}{os.path.basename(dirpath) or dirpath}/")
            if depth < max_depth:
                lines += [f"{indent}  {f}" for f in sorted(filenames) if not f.startswith(".")]
        return "\n".join(lines[:300])  # Prevent screen flooding from huge repositories


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Candidate:
    """A file that passed the score threshold, with the hints that order
    files of about the same score."""
    rel: str
    score: float
    feats: dict
    refs: int           # files importing it
    example_refs: int   # example files importing it
    content: int        # FileSelector.content_signal()

    @property
    def is_api_code(self) -> bool:
        return bool(self.feats["ext_group_code"] and (self.feats["is_public_api"] or self.feats["is_cli_entry"])
                    and not _is_example(self.feats))

    @property
    def is_example_code(self) -> bool:
        return bool(self.feats["ext_group_code"] and _is_example(self.feats))

    def rank_key(self) -> tuple:
        """Model score first, then deterministic tie-breakers.

        Without tie-breakers a saturated model (hundreds of files at ~1.00)
        falls back to reverse-alphabetical path order, which is meaningless."""
        return (
            -round(self.score, SCORE_TIE_DECIMALS),
            -round(self.feats["rule_score"] / RULE_SCORE_BUCKET),   # finer-grained heuristic signal
            -self.content,             # docs with more runnable snippets first
            self.feats["depth"],       # shallower first
            -self.refs,                # more widely imported first
            self.rel,
        )

    def api_rank_key(self) -> tuple:
        """For public-API code, how many example files import it is the best
        usefulness hint (the public header or SDK package users start from),
        then the overall import count; the import count alone favours internal
        hubs that every module of the project includes."""
        return (
            -round(self.score, SCORE_TIE_DECIMALS),
            -self.example_refs,
            -self.refs,
            -round(self.feats["rule_score"] / RULE_SCORE_BUCKET),
            self.feats["depth"],
            self.rel,
        )


class FileSelector:
    """Picks the files to pack.  `model` is next to this file: a neural
    scorer (.pt, model/nn/) or a FileScorer pickle; without one, files are
    scored by their rule_score.  `threshold` defaults to the one saved with
    a neural scorer, SECONDARY_THRESHOLD otherwise."""

    def __init__(self, model: str = SCORER_MODEL, threshold: float | None = None):
        path = Path(__file__).resolve().parent / model
        try:
            if path.suffix == ".pt":
                from model.nn.scorer import NNScorer     # torch is only needed for this one
                self.scorer = NNScorer.load(path)
            else:
                self.scorer = FileScorer.load(str(path))
        except SystemExit:      # FileScorer.load exits when the file is missing
            self.scorer = None
        except Exception as err:
            print(f"Skipping score model {model}: {err}")
            self.scorer = None
        print(f"Loaded score model: {model}" if self.scorer else
              "No score model loaded; falling back to rule_score.")
        self.threshold = threshold if threshold is not None else getattr(self.scorer, "threshold", SECONDARY_THRESHOLD)

    @staticmethod
    def priority(repo: Repo) -> list[str]:
        """The PRIORITY_FILES found at the repo root or one directory down."""
        by_name = {rel.replace(os.sep, "/").lower(): rel for rel in repo.files if rel.count(os.sep) <= 1}
        return [by_name[t] for t in PRIORITY_FILES if t in by_name]

    def secondary(self, repo: Repo, exclude: Iterable[str] = ()) -> list[str]:
        """The top-scoring files of the repo.

        Every file is scored with the model, so important files in src/,
        package roots or CLI entry points compete with docs and examples.
        `exclude` holds paths already collected as priority files, so the
        same README is not packed twice.

        The neural scorer's ranking is used as is.  For the path-only models,
        ties are broken by rule_score, content (runnable snippets in docs),
        depth and how many files import them, slots are reserved for code and
        directories are capped.
        """
        exclude = {os.path.normcase(os.path.normpath(p)) for p in exclude}
        rels = [r for r in repo.files if os.path.normcase(r) not in exclude]
        feats = [extract_features(r) for r in rels]
        scores = self.score(repo, rels, feats)
        if self.scorer is not None and not isinstance(self.scorer, FileScorer):
            return self._top(rels, scores)

        print(f"Building reference counts for {len(repo.files)} files...")
        example_refs: dict[str, int] = {}
        refs = build_ref_counts(str(repo.root), repo.files, example_counts=example_refs)
        passing = [Candidate(r, s, f, refs.get(r, 0), example_refs.get(r, 0), self.content_signal(repo, r))
                   for r, s, f in zip(rels, scores, feats) if s >= self.threshold]
        passing.sort(key=Candidate.rank_key)

        # Docs carry a higher base rule_score than code, so without a quota the
        # public headers / API modules lose every slot to READMEs.
        api_code = sorted((c for c in passing if c.is_api_code), key=Candidate.api_rank_key)
        example_code = [c for c in passing if c.is_example_code]    # already in rank order
        reserved = self._diverse(api_code, MIN_API_CODE_FILES)
        reserved = self._diverse(example_code, len(reserved) + MIN_EXAMPLE_CODE_FILES, picked=reserved)
        top = self._diverse(passing, MAX_SECONDARY_FILES, picked=reserved)
        top.sort(key=passing.index)

        print(f"Secondary files: {len(passing)} candidates >= {self.threshold} "
              f"({len(api_code)} API code, {len(example_code)} example code), "
              f"keeping top {len(top)} ({len(reserved)} reserved for code)")
        return [c.rel for c in top]

    def _top(self, rels: list[str], scores: list[float]) -> list[str]:
        """The neural scorer's MAX_SECONDARY_FILES best files.  It was trained
        on the gold lists with every file seen next to its directory and the
        repo, so it already weighs API code, examples and docs against each
        other; the code quotas and directory caps of the path-only models
        lowered its leave-one-repo-out core recall from 0.62 to 0.45 and its
        hit rate from 0.62 to 0.40 (model/nn/README.md)."""
        ranked = sorted((-s, r.count(os.sep), r) for s, r in zip(scores, rels) if s >= self.threshold)
        top = [r for _, _, r in ranked[:MAX_SECONDARY_FILES]]
        print(f"Secondary files: {len(ranked)} candidates >= {self.threshold}, keeping top {len(top)}")
        return top

    def score(self, repo: Repo, rels: list[str], feats: list[dict]) -> list[float]:
        if self.scorer is None:
            return [f["rule_score"] for f in feats]
        if isinstance(self.scorer, FileScorer):
            return self.scorer.predict([features_to_vector(f) for f in feats])
        # The neural scorer compares each file with its directory and the
        # whole repo, so it scores every file, priority files included
        scores = self.scorer.score_repo(repo.root, repo.files, repo.name)
        return [scores[r] for r in rels]

    @staticmethod
    def content_signal(repo: Repo, rel: str) -> int:
        """Cheap content-based usefulness hint for files that already passed
        the path-based scorer.

        For docs: the number of fenced code blocks / shell prompts, since a
        cheatsheet is built from runnable commands and snippets.  Near-empty
        stubs get -1.  Other files: 0 (refs cover code).
        """
        try:
            if (repo.root / rel).stat().st_size < STUB_FILE_BYTES:
                return -1
        except OSError:
            return 0
        if Path(rel).suffix.lower() not in DOC_SUFFIXES:
            return 0
        lines = (repo.read(rel) or "")[:64 * 1024].splitlines()
        fences = sum(1 for line in lines if line.lstrip().startswith(("```", "~~~")))
        prompts = sum(1 for line in lines if line.lstrip().startswith("$ "))
        return min(fences // 2 + prompts, 30)

    @staticmethod
    def _diverse(ranked: list[Candidate], limit: int, picked: list[Candidate] = ()) -> list[Candidate]:
        """Pick up to `limit` files, capping how many come from the same
        directory so that one directory with many similar files (e.g. one
        README per example under examples/) cannot crowd out everything else.

        `picked` are files already chosen (e.g. reserved code slots); they
        count toward the limit and the per-directory caps."""
        picked = list(picked)
        per_parent, per_top = Counter(), Counter()

        def dir_keys(c: Candidate) -> tuple[str, str]:
            parts = Path(c.rel).parts
            return str(Path(c.rel).parent), (parts[0] if len(parts) > 1 else "")

        for c in picked:
            parent, top = dir_keys(c)
            per_parent[parent] += 1
            per_top[top] += 1
        for c in ranked:
            if len(picked) >= limit:
                break
            parent, top = dir_keys(c)
            if c in picked or per_parent[parent] >= MAX_PER_PARENT_DIR or (top and per_top[top] >= MAX_PER_TOP_DIR):
                continue
            per_parent[parent] += 1
            per_top[top] += 1
            picked.append(c)
        return picked


# ---------------------------------------------------------------------------
# Excerpts
#
# A file's first N characters are often its least useful part: a license
# header, imports, rarely used rows at the top of an auto-generated option
# table, the first section of a long config file.  Excerpter keeps what a
# user needs instead:
#
#   docs     sections with commands and code first; long option tables keep
#            the rows whose flags the repo's docs and examples actually use
#   code     license and imports dropped; examples keep their bodies minus
#            repeated boilerplate; other code is reduced to declarations
#            (signatures, struct fields, argparse options) ranked by usage
#   config   large YAML files become an outline of sections and settings
#   scripts  functions, case labels, usage messages and the commands they run
#
# "Usage" is Excerpter.terms: for each identifier or command-line flag, the
# number of example files and doc code blocks that mention it.
# ---------------------------------------------------------------------------

# Flags, dotted calls (pkg.New, client.search) and identifiers
_TOKEN = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z_][A-Za-z0-9_]{2,}")
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1", re.M | re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_COMMENT_LINE = re.compile(r"^\s*(//|#|/\*|\*|\*/|--|;)")
_LICENSE_WORDS = re.compile(r"licen[cs]e|copyright|spdx", re.I)

_HEADING = re.compile(r"^#{1,6}\s")
_NOISE_LINE = re.compile(r"^\s*(\[!\[|!\[|</?(p|div|img|a|br|picture|source|details|summary)\b)", re.I)
_TOC_LINE = re.compile(r"^\s*([-*]|\d+\.)\s*\[[^\]]*\]\(#[^)]*\)\s*$")
_HTML_TAG = re.compile(r"</?(details|summary|p|div|br|img|a|picture|source|sub|sup|kbd)\b[^>]*>", re.I)
_USAGE_HEADING = re.compile(r"install|build|quick|start|usage|example|getting|run|serv|api|endpoint|"
                            r"config|option|param|command|tutorial|how to|docker|deploy|connect", re.I)
_NOISE_HEADING = re.compile(r"contribut|licen[cs]e|changelog|acknowledg|citation|communit|sponsor|"
                            r"star history|hot topic|recent|news|roadmap|table of contents|code of conduct", re.I)

# Error-handling lines, which examples repeat after every call
_ERROR_HANDLING = re.compile(
    r"^\s*(if err != nil \{|log\.Fatal|panic\(|fprintf\(stderr|std::cerr|LOG_ERR|// handle err"
    r"|return (1|-1|err|nil, err);?\s*$|exit\(1\)|sys\.exit\(1\))")

_SCRIPT_KEEP = re.compile(
    r"^\s*(#(?!!)|rem\b|::)"                                   # comments
    r"|^\s*(function\s+)?[\w-]+\s*\(\)\s*\{?\s*$"              # function definitions
    r"|^\s*[\w|*.-]+\)\s*$"                                    # case labels
    r"|usage|please use|--help"                                # usage messages
    r"|^\s*(sudo\s+)?(docker|podman|curl|wget|pip3?|python3?|go|npm|yarn|helm|kubectl|cmake|make|"
    r"apt(-get)?|brew|git|tar|systemctl)\s",                   # commands run
    re.I)


def _strip_license_header(text: str) -> str:
    """Drop a leading comment block that is a license header.

    Only the first block, up to the first blank line, is considered: the
    module or package doc comment that usually follows is worth keeping.  A
    shebang line is kept.  Call before blank lines are collapsed."""
    lines = text.split("\n")
    if lines[0].startswith("#!"):
        return lines[0] + "\n" + _strip_license_header("\n".join(lines[1:]).lstrip("\n"))
    k = 0
    while k < len(lines) and lines[k].strip() and _COMMENT_LINE.match(lines[k]):
        k += 1
    return "\n".join(lines[k:]) if k and _LICENSE_WORDS.search("\n".join(lines[:k])) else text


def _collapse(text: str) -> str:
    """Blank-line runs to one blank line, trailing spaces removed."""
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()


def _clip(text: str, budget: int) -> str:
    """Cut to `budget` characters, at a line end when one is near."""
    if len(text) <= budget:
        return text
    cut = text.rfind("\n", 0, budget)
    return text[:cut if cut > budget // 2 else budget]


def _cut_line(line: str) -> str:
    return line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…"


def _one_line(code: str) -> str:
    return _cut_line(re.sub(r"\s+", " ", code).strip())


def _technical(token: str) -> bool:
    """Flags, dotted calls and identifiers that look like code (snake_case,
    CamelCase, digits) rather than English words."""
    return token.startswith("-") or "." in token or bool(re.search(r"[_A-Z0-9]", token[1:]))


def _split(lines: list[str], starts_group) -> list[list[str]]:
    """Start a new group at each line where starts_group(line) holds, except
    inside fenced code blocks."""
    groups, in_fence = [[]], False
    for line in lines:
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence and starts_group(line) and groups[-1]:
            groups.append([])
        groups[-1].append(line)
    return groups


class Excerpter:
    """Cuts the files of one repo to a character budget (see above)."""

    def __init__(self, repo: Repo):
        self.repo = repo

    @functools.cached_property
    def terms(self) -> Counter:
        """Identifier / flag → number of files using it, counting the code
        blocks and inline code of markdown docs and the full text of example
        code."""
        terms: Counter = Counter()
        for rel in self.repo.files:
            suffix = Path(rel).suffix.lower()
            is_doc = suffix in {".md", ".mdx"}
            if not (is_doc or suffix in CODE_SUFFIXES):
                continue
            feats = extract_features(rel)
            if feats["in_ignored"] or not (is_doc or _is_example(feats)):
                continue
            text = self.repo.read(rel)
            if text and is_doc:
                text = "\n".join([m.group(2) for m in _FENCE.finditer(text)] + _INLINE_CODE.findall(text))
            if text:
                terms.update(set(_TOKEN.findall(text)))
        return terms

    def excerpt(self, rel: str, budget: int) -> str:
        """At most `budget` characters of the file worth showing to the LLM."""
        text = self.repo.read(rel)
        if text is None:
            return "[binary or unreadable file]"
        suffix = Path(rel).suffix.lower()
        if suffix in DOC_SUFFIXES or Path(rel).name.lower().startswith("readme"):
            return self._doc(text, budget)
        text = _collapse(_strip_license_header(text.lstrip()))
        if len(text) <= budget:
            return text
        if suffix in YAML_SUFFIXES:
            return self._yaml(text, budget)
        if suffix in SCRIPT_SUFFIXES:
            return self._script(text, budget)
        if suffix in CODE_SUFFIXES:
            return self._code(text, LANGUAGES.get(suffix, Language()), budget, _is_example(extract_features(rel)))
        return _clip(text, budget)

    # --- docs --------------------------------------------------------------

    def _doc(self, text: str, budget: int) -> str:
        lines = [_HTML_TAG.sub("", l) for l in re.sub(r"<!--.*?-->", "", text, flags=re.S).split("\n")
                 if not _NOISE_LINE.match(l) and not _TOC_LINE.match(l)]
        sections = [_collapse("\n".join(self._shrink_tables(s))) for s in _split(lines, _HEADING.match)]
        sections = [s for s in sections if s]
        if len("\n\n".join(sections)) <= budget:
            return "\n\n".join(sections)

        # Best sections first: the whole section when it fits in half the
        # budget (so one long section cannot crowd out the rest), else its digest
        scores = [self._section_score(s) if i else 100.0 for i, s in enumerate(sections)]
        chosen, size = {}, 0
        for i in sorted(range(len(sections)), key=lambda i: (-scores[i], i)):
            if scores[i] <= 0:
                break
            whole = [sections[i]] if len(sections[i]) <= budget // 2 else []
            for text in whole + [self._digest(sections[i], budget)]:
                if size + len(text) + 2 <= budget:
                    chosen[i] = text
                    size += len(text) + 2
                    break
        return "\n\n".join(chosen[i] for i in sorted(chosen))

    def _section_score(self, section: str) -> float:
        heading = section.split("\n", 1)[0] if _HEADING.match(section) else ""
        if _NOISE_HEADING.search(heading):
            return 0.0
        fences = section.count("```") // 2 + section.count("~~~") // 2
        # A table listing flags or identifiers is reference material, like code
        option_table = any(line.lstrip().startswith("|") and line.count("|") > 1
                           and any(map(_technical, _TOKEN.findall(line.split("|")[1])))
                           for line in section.split("\n"))
        hot = {t for t in _TOKEN.findall(section) if _technical(t) and self.terms[t] >= 2}
        return (3 * min(fences, 3) + 3 * option_table + 2 * bool(_USAGE_HEADING.search(heading))
                + min(len(hot), 10) / 2 + 0.1)

    @staticmethod
    def _digest(section: str, budget: int) -> str:
        """Heading, about 400 characters of prose (paragraphs and lists), the
        first code block (up to a third of the budget) and the first table
        (up to half of it, whole rows) of a section."""
        out, prose, code, table = [], 0, False, False
        for group in _split(section.split("\n"), lambda l: not l.strip()):
            block = "\n".join(group).strip("\n")
            if not block.strip():
                continue
            if _HEADING.match(block):
                out.append(block)
            elif block.lstrip().startswith(("```", "~~~")):
                if not code:
                    limit = max(600, budget // 3)
                    out.append(_clip(block, limit) + ("\n```" if len(block) > limit else ""))
                    code = True
            elif block.lstrip().startswith("|"):
                if not table:
                    out.append(_clip(block, max(800, budget // 2)))
                    table = True
            elif prose < 400:
                out.append(_clip(block, 400 - prose))
                prose += len(block)
        return "\n\n".join(out)

    def _row_score(self, row: str) -> int:
        """Usage of the flags in a table row's first cell when there are any
        (the words in a placeholder such as "<file name>" say nothing), else
        of its identifiers."""
        cell = row.split("|")[1] if row.count("|") > 1 else row
        tokens = [t for t in _TOKEN.findall(cell) if _technical(t)]
        flags = [t for t in tokens if t.startswith("-")]
        return max((self.terms[t] for t in flags or tokens), default=0)

    def _shrink_tables(self, lines: list[str]) -> list[str]:
        """Keep the header and the TABLE_ROWS most-used rows of a long table;
        drop inline env notes."""
        out, run = [], []
        for line in lines + [""]:
            if line.lstrip().startswith("|"):
                run.append(_cut_line(re.sub(r"<br/?>\s*\(env: [^)]*\)", "", line)))
                continue
            if len(run) > TABLE_ROWS + 2:
                rows = run[2:]
                keep = sorted(sorted(range(len(rows)), key=lambda i: (-self._row_score(rows[i]), i))[:TABLE_ROWS])
                run = run[:2] + [rows[i] for i in keep] + [f"| … {len(rows) - len(keep)} more rows |"]
            out += run + [line]
            run = []
        return out[:-1]

    # --- code --------------------------------------------------------------

    def _code(self, text: str, lang: "Language", budget: int, example: bool) -> str:
        text = lang.strip_imports(text)
        if len(text) <= budget:
            return text
        if example:
            return self._example(text, budget)
        decls = lang.declarations(text)
        if not decls:
            return _clip(text, budget)
        # The most widely used names (in examples and docs) that fit, in file order
        chosen, size = [], 0
        for pos, names, rendered in sorted(decls, key=lambda d: (-max(self.terms[n] for n in d[1]), d[0])):
            if size + len(rendered) + 1 <= budget:
                chosen.append((pos, rendered))
                size += len(rendered) + 1
        return "\n".join(rendered for _, rendered in sorted(chosen))

    def _example(self, text: str, budget: int) -> str:
        """Example code reduced to the blocks (runs between blank lines) that
        show the most API calls not shown yet.

        Lines already shown are dropped first: the second example in a file
        repeats the first one's setup, error handling and closing braces.
        Then blocks are picked greedily by new identifiers per character,
        weighting identifiers by how widely the repo's examples and docs use
        them, and printed in file order."""
        seen, lines = set(), []
        for line in text.split("\n"):
            key = line.strip()
            if key and key in seen or _ERROR_HANDLING.search(line):
                continue
            seen.add(key)
            lines.append(_cut_line(line))
        blocks = [b for b in re.split(r"\n\s*\n", "\n".join(lines)) if b.strip()]
        tokens = [{t for t in _TOKEN.findall(b) if _technical(t)} for b in blocks]
        covered, chosen, size = set(), set(), 0

        def gain(i: int) -> float:
            return sum(1 + min(self.terms[t], 10) / 5 for t in tokens[i] - covered) / (len(blocks[i]) + 40)

        while True:
            fits = [i for i in range(len(blocks)) if i not in chosen and size + len(blocks[i]) + 2 <= budget]
            best = max(fits, key=gain, default=None)
            if best is None or gain(best) <= 0:
                break
            chosen.add(best)
            covered |= tokens[best]
            size += len(blocks[best]) + 2
        return "\n\n".join(blocks[i] for i in sorted(chosen))

    # --- config and scripts ------------------------------------------------

    @staticmethod
    def _yaml(text: str, budget: int) -> str:
        """One line per top-level section listing its settings, shortened so
        that every section fits: a map of what can be configured."""
        sections: list[list[str]] = []
        child_indent = None
        for raw in text.split("\n"):
            line = re.sub(r"\s+#.*$", "", raw).rstrip()
            m = re.match(r"( *)([\w.-]+):\s*(.*)$", line)
            if not m:
                continue
            indent, key, value = len(m.group(1)), m.group(2), m.group(3)
            if indent == 0:
                sections.append([line.strip()])
                child_indent = None
            elif sections and indent == (child_indent := child_indent or indent):
                sections[-1].append(f"{key}: {value}" if value else f"{key}: {{…}}")
        if not sections:
            return _clip(text, budget)
        width = max(60, budget // len(sections) - 1)
        lines = [s[0] + (" " + ", ".join(s[1:]) if len(s) > 1 else "") for s in sections]
        return _clip("\n".join(l if len(l) <= width else l[:width - 1] + "…" for l in lines), budget)

    @staticmethod
    def _script(text: str, budget: int) -> str:
        # Joined line continuations; commands keep all their arguments, since
        # image names and ports come last
        keep = dict.fromkeys(re.sub(r"\s+", " ", line.strip())[:600]
                             for line in re.sub(r"\\\n\s*", " ", text).split("\n") if _SCRIPT_KEEP.search(line))
        return _clip("\n".join(keep) if keep else text, budget)


# --- languages -------------------------------------------------------------

# Declarations: (position, names used for ranking, rendered text)
Decl = tuple[int, list[str], str]


class Language:
    """Import lines to drop and public declarations to keep for one language.

    The base class finds declarations by keyword (fn, class, struct, ...);
    subclasses parse their language's signatures, struct fields and options."""
    DECL = re.compile(
        r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:public\s+|static\s+|async\s+)*"
        r"(?:fn|function|class|interface|struct|trait|enum|def|type|module)\s+([A-Za-z_]\w*)[^\n]*", re.M)

    def __init__(self, imports: str | None = None):
        self.imports = re.compile(imports, re.M) if imports else None

    def strip_imports(self, text: str) -> str:
        return _collapse(self.imports.sub("", text)) if self.imports else text

    def declarations(self, text: str) -> list[Decl]:
        return [(m.start(), [m.group(1)], _cut_line(m.group(0).strip().rstrip(" {")))
                for m in self.DECL.finditer(text)]

    @staticmethod
    def _with_doc(text: str, start: int, rendered: str) -> str:
        """Prefix `rendered` with the nearest // comment line above `start`,
        skipping TODOs and links."""
        above = text[:start].rstrip("\n").split("\n")
        while above and above[-1].strip().startswith("//"):
            line = above.pop().strip()
            if not re.search(r"TODO|FIXME|XXX|https?://|ref:", line) and len(line.strip("/ ")) > 3:
                return _cut_line(line) + "\n" + rendered
        return rendered


class Python(Language):
    """Public functions and classes with the first docstring line, and
    argparse options with their help."""

    def declarations(self, text: str) -> list[Decl]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return super().declarations(text)
        decls = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                if isinstance(node, ast.ClassDef):
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                    sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
                else:
                    sig = f"def {node.name}({ast.unparse(node.args)})"
                doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
                decls.append((node.lineno, [node.name], _one_line(sig + (f"  # {doc}" if doc else ""))))
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                help_ = next((k.value.value for k in node.keywords
                              if k.arg == "help" and isinstance(k.value, ast.Constant)), "")
                if flags:
                    decls.append((node.lineno, flags, _one_line(f"{', '.join(flags)}: {help_}")))
        return decls


class Go(Language):
    """Exported functions and types, with the exported fields of structs."""
    FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)\b[^\n]*", re.M)
    STRUCT = re.compile(r"^type\s+([A-Z]\w*)\s+struct\s*\{\n(.*?)^\}", re.M | re.S)
    TYPE = re.compile(r"^type\s+([A-Z]\w*)\s+(?!struct\s*\{)[^\n]*", re.M)

    def declarations(self, text: str) -> list[Decl]:
        decls = [(m.start(), [m.group(1)], self._with_doc(text, m.start(), _cut_line(m.group(0).rstrip(" {"))))
                 for m in self.FUNC.finditer(text)]
        for m in self.STRUCT.finditer(text):
            fields = [_one_line(f) for f in m.group(2).split("\n") if f.strip() and f.strip()[0].isupper()]
            body = f"type {m.group(1)} struct {{\n" + "\n".join("\t" + f for f in fields) + "\n}"
            decls.append((m.start(), [m.group(1)], self._with_doc(text, m.start(), body)))
        decls += [(m.start(), [m.group(1)], _cut_line(m.group(0).rstrip(" {"))) for m in self.TYPE.finditer(text)]
        return decls


class CFamily(Language):
    """Function prototypes and struct fields; deprecated functions and
    function pointers are skipped."""
    PROTO = re.compile(
        r"^[ \t]*((?:[A-Za-z_][\w:<>]*[ \t\*&]+)+)([A-Za-z_]\w*)[ \t]*\(([^;{}]{0,600}?)\)[ \t]*(?:const[ \t]*)?[;{]",
        re.M)
    STRUCT = re.compile(r"^[ \t]*(?:typedef[ \t]+)?struct[ \t]+(\w+)[ \t]*\{\n(.*?)^[ \t]*\}", re.M | re.S)
    NOT_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "else", "do", "case", "defined",
                 "void", "bool", "char", "int", "float", "double", "long", "short", "unsigned", "signed"}

    def declarations(self, text: str) -> list[Decl]:
        decls = []
        for m in self.PROTO.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            if (m.group(2) in self.NOT_NAMES or m.group(3).lstrip().startswith("*")   # function pointer
                    or "DEPRECATED" in text[line_start:m.start() + 20]
                    or m.group(1).strip() in {"return", "else", "new", "delete"}):
                continue
            sig = _one_line(m.group(0).rstrip("{").rstrip() + (";" if m.group(0).endswith("{") else ""))
            decls.append((m.start(), [m.group(2)], self._with_doc(text, line_start, sig)))
        for m in self.STRUCT.finditer(text):
            fields = [re.sub(r"\s+", " ", re.sub(r"//.*", "", f)).strip() for f in m.group(2).split("\n")]
            fields = [f for f in fields if f and not f.startswith(("/*", "*"))]
            decls.append((m.start(), [m.group(1)], _cut_line(f"struct {m.group(1)} {{ " + " ".join(fields) + " }")))
        return decls


LANGUAGES: dict[str, Language] = {
    ".py": Python(r"^from\s+\S+\s+import\s*\([^)]*\)\n?|^(?:from\s+\S+\s+)?import\s[^\n]*\n?"),
    ".go": Go(r"^import\s*\([^)]*\)\n?|^import\s+\S*\s*\"[^\"]+\"\n?"),
    ".rs": Language(r"^use\s[^;]*;\n?"),
    **dict.fromkeys((".java", ".kt"), Language(r"^import\s[^\n]*\n?")),
    **dict.fromkeys(C_SUFFIXES, CFamily(r"^\s*#\s*include\b[^\n]*\n?")),
    **dict.fromkeys(JS_SUFFIXES, Language(r"^import\b[^;]*?(?:from\s+)?['\"][^'\"]+['\"];?\n?")),
}


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def build_context_bundle(repo_url: str, selector: FileSelector | None = None):
    """Main function: clone + select + excerpt into a single text

    Returns (bundle, scanned_files, truncated):
        bundle        -- Packaged context text
        scanned_files -- List of files actually read this time,
                         each item is a dict: {"path": relative path, "category": "priority"|"example", "chars": actual character count}
        truncated     -- bool, whether the packaged content was truncated due to exceeding MAX_TOTAL_CHARS
    """
    selector = selector or FileSelector()
    with Repo.clone(repo_url) as repo:
        excerpter = Excerpter(repo)
        priority = selector.priority(repo)
        groups = (("priority", "File", priority, MAX_FILE_CHARS),
                  ("example", "Example File", selector.secondary(repo, exclude=priority), SECONDARY_FILE_CHARS))

        parts = [f"# Repository: {repo_url}\n", "## Directory Structure (Partial)\n```\n" + repo.tree() + "\n```\n"]
        scanned_files = []
        for category, label, files, budget in groups:
            for path in files:
                content = excerpter.excerpt(path, budget)
                parts.append(f"## {label}: {path}\n```\n{content}\n```\n")
                scanned_files.append({"path": path, "category": category, "chars": len(content)})

    bundle = "\n".join(parts)
    truncated = len(bundle) > MAX_TOTAL_CHARS
    if truncated:
        bundle = bundle[:MAX_TOTAL_CHARS] + "\n\n[Content too long, truncated]"
    return bundle, scanned_files, truncated
