"""
ref_counts.py — Cross-file import reference counts for a repo.

Not an ML feature: collector.py uses the counts to break ties between files
with the same score, and the counts from example files to pick which
public-API files to keep.

Import statements are parsed per language and resolved to real repo paths.
Earlier versions matched only the *stem* of whatever was imported, which
conflated unrelated files (every utils.* got credit for every `import utils`,
and pytest.ini got credit for every `import pytest`). We still use regexes
rather than full parsers — the goal is a cheap, mostly-correct signal.
"""

import posixpath
import re
from pathlib import Path

from model.file_features import features


def build_ref_counts(repo_root: str, file_list: list[str],
                     example_counts: dict[str, int] | None = None) -> dict[str, int]:
    """Scan source files in file_list for import/include statements and return
    rel_path → number of distinct files that reference it (0 if none).

    Specifiers are resolved to concrete repo paths per language (relative
    imports against the importer's directory, absolute ones by path suffix,
    Go imports via the module paths declared in go.mod). Ambiguous or
    external specifiers are dropped instead of being credited to every file
    with a matching name.

    example_counts: optional dict, filled in place with rel_path → number of
    example files referencing it. Unlike the overall count this singles out
    what *users* call: a server repo's internal pkg/ is imported everywhere,
    its SDK mostly by examples.
    """
    root = Path(repo_root)
    counts: dict[str, int] = {f: 0 for f in file_list}
    idx = _RepoIndex(file_list)

    for n, rel in idx.by_norm.items():
        if posixpath.basename(n) == "go.mod":
            try:
                m = _GoResolver.MODULE.search((root / rel).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if m:
                idx.go_modules[m.group(1).lower()] = posixpath.dirname(n)

    if example_counts is not None:
        example_counts.update({f: 0 for f in file_list})

    for rel in file_list:
        resolver = _RESOLVERS.get(Path(rel).suffix.lower())
        if resolver is None:
            continue
        general = _counts_as_referrer(rel)
        example = example_counts is not None and _is_example_referrer(rel)
        if not (general or example):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        importer = _norm(rel)
        targets = set(resolver.references(text, importer, idx))
        targets.discard(importer)           # don't count self-references
        for target in targets:
            orig = idx.by_norm[target]
            if general:
                counts[orig] = counts.get(orig, 0) + 1
            if example:
                example_counts[orig] = example_counts.get(orig, 0) + 1

    return counts


def _counts_as_referrer(rel: str) -> bool:
    """References from tests, CI scripts or vendored code say nothing about
    what a *user* of the project touches, so they are not counted.
    Go *_example_test.go files are usage docs and do count."""
    path = Path(rel)
    under = {p.lower() for p in path.parts}
    if under & (features.dirs["tests"] | features.dirs["ignored"] | features.dirs["devops"]):
        return False
    if path.name.lower().endswith("example_test.go"):
        return True
    return not (features.patterns["test"].search(path.stem.lower())
                or features.ext_group(path) == "test")


def _is_example_referrer(rel: str) -> bool:
    """Files that show users how to call the project: examples/ trees,
    *_example.py style samples and Go *_example_test.go."""
    path = Path(rel)
    under = {p.lower() for p in path.parts}
    if under & features.dirs["ignored"]:
        return False
    return bool(under & features.dirs["examples"]
                or features.patterns["example"].search(path.stem.lower()))


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

def _norm(rel: str) -> str:
    return rel.replace("\\", "/").lower()


class _RepoIndex:
    """Lookup tables for resolving import specifiers to repo files."""

    def __init__(self, file_list: list[str]):
        self.by_norm: dict[str, str] = {}               # normalised path → original rel
        self.by_suffix: dict[str, list[str]] = {}       # "a/b.py", "b.py" → [norm paths]
        self.go_files_in_dir: dict[str, list[str]] = {}
        for rel in file_list:
            n = _norm(rel)
            self.by_norm[n] = rel
            parts = n.split("/")
            for i in range(len(parts)):
                self.by_suffix.setdefault("/".join(parts[i:]), []).append(n)
            if n.endswith(".go") and not n.endswith("_test.go"):
                self.go_files_in_dir.setdefault(posixpath.dirname(n), []).append(n)
        self.go_modules: dict[str, str] = {}             # module path → dir (norm)

    def exists(self, n: str) -> bool:
        return n in self.by_norm

    def relative(self, importer_dir: str, spec: str, exts) -> str | None:
        base = posixpath.normpath(posixpath.join(importer_dir, spec))
        if base.startswith(".."):
            return None
        for ext in exts:
            if self.exists(base + ext):
                return base + ext
        return None

    def by_path_suffix(self, importer_dir: str, spec: str, exts) -> str | None:
        """Find the file whose path ends with spec+ext.  When several match,
        prefer the one sharing the longest directory prefix with the importer;
        give up if that is still ambiguous rather than crediting all of them."""
        spec = spec.strip("/")
        if not spec or spec.startswith("."):
            return None
        for ext in exts:
            cands = self.by_suffix.get(spec + ext)
            if not cands:
                continue
            if len(cands) == 1:
                return cands[0]
            imp = importer_dir.split("/") if importer_dir else []

            def shared(c: str) -> int:
                d = posixpath.dirname(c).split("/")
                k = 0
                while k < min(len(d), len(imp)) and d[k] == imp[k]:
                    k += 1
                return k

            ranked = sorted(cands, key=shared, reverse=True)
            if shared(ranked[0]) > shared(ranked[1]):
                return ranked[0]
            return None
        return None


class _ImportResolver:
    """Resolves one language's import specifiers to repo files.

    By default `pattern` group 1 is the specifier (passed through `clean`,
    which returns None to skip it), tried relative to the importer's directory
    with each suffix in `targets`, then as a path suffix anywhere in the repo
    when `by_suffix`. Subclasses override specs / resolve / references.
    """

    def __init__(self, exts, pattern=None, targets=("",), relative=True, by_suffix=False, clean=None):
        self.exts, self.pattern, self.targets = exts, pattern, targets
        self.relative, self.by_suffix, self.clean = relative, by_suffix, clean

    def specs(self, text: str):
        for m in self.pattern.finditer(text):
            spec = m.group(1).lower()
            spec = self.clean(spec) if self.clean else spec
            if spec is not None:
                yield spec

    def resolve(self, spec: str, importer_dir: str, idx: "_RepoIndex") -> list[str]:
        hit = ((self.relative and idx.relative(importer_dir, spec, self.targets))
               or (self.by_suffix and idx.by_path_suffix(importer_dir, spec, self.targets)))
        return [hit] if hit else []

    def references(self, text: str, importer_rel: str, idx: "_RepoIndex"):
        importer_dir = posixpath.dirname(importer_rel)
        for spec in self.specs(text):
            yield from self.resolve(spec, importer_dir, idx)


class _PythonResolver(_ImportResolver):
    # Names end at the line break unless parenthesised: `from x import (\n a,\n b)`
    FROM_IMPORT = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(\([^)]*\)|[\w \t,*]+)", re.M)
    IMPORT = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)", re.M)

    def __init__(self):
        super().__init__((".py",), targets=(".py", "/__init__.py"), by_suffix=True)

    def specs(self, text):
        for m in self.FROM_IMPORT.finditer(text):
            mod, names = m.groups()
            if mod.strip(".") == "":
                # `from . import a, b` — the names are sibling modules
                yield from (mod + n for n in re.split(r"[\s,()]+", names) if n and n != "*")
            else:
                yield mod
        for m in self.IMPORT.finditer(text):
            yield from (s.strip() for s in m.group(1).split(","))

    def resolve(self, spec, importer_dir, idx):
        dots = len(spec) - len(spec.lstrip("."))
        path = spec.lstrip(".").replace(".", "/").lower()
        if not dots:
            return super().resolve(path, importer_dir, idx)
        up = "/".join([".."] * (dots - 1))
        hit = idx.relative(importer_dir, posixpath.join(up, path) if up else path, self.targets)
        return [hit] if hit else []


class _GoResolver(_ImportResolver):
    MODULE = re.compile(r"^\s*module\s+(\S+)", re.M)
    IMPORT_LINE = re.compile(r"^\s*import\s+(?:[\w.]+\s+)?\"([^\"]+)\"", re.M)
    IMPORT_BLOCK = re.compile(r"^\s*import\s*\(([^)]*)\)", re.M)
    QUOTED = re.compile(r"\"([^\"]+)\"")

    def __init__(self):
        super().__init__((".go",))

    def specs(self, text):
        yield from (m.group(1).lower() for m in self.IMPORT_LINE.finditer(text))
        for block in self.IMPORT_BLOCK.finditer(text):
            yield from (s.lower() for s in self.QUOTED.findall(block.group(1)))

    def resolve(self, spec, importer_dir, idx):
        # Longest module prefix wins (nested modules such as milvus pkg/v2)
        best = max((mod for mod in idx.go_modules if spec == mod or spec.startswith(mod + "/")),
                   key=len, default=None)
        if best is None:
            return []                        # external dependency
        pkg_dir = posixpath.normpath(posixpath.join(idx.go_modules[best], spec[len(best):].lstrip("/")))
        # A Go import names a package (directory), so every file in it is used
        return idx.go_files_in_dir.get("" if pkg_dir == "." else pkg_dir, [])


class _RustResolver(_ImportResolver):
    MOD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M)
    USE_CRATE = re.compile(r"\buse\s+crate::([\w:]+)")

    def __init__(self):
        super().__init__((".rs",), targets=(".rs", "/mod.rs"))

    def _first(self, idx, candidates):
        return next(([hit] for d, spec in candidates
                     if (hit := idx.relative(d, spec, self.targets))), [])

    def references(self, text, importer_rel, idx):
        importer_dir = posixpath.dirname(importer_rel)
        stem = posixpath.splitext(posixpath.basename(importer_rel))[0]
        mod_dirs = [importer_dir] if stem in ("mod", "lib", "main") else [importer_dir, posixpath.join(importer_dir, stem)]
        for m in self.MOD.finditer(text):
            yield from self._first(idx, ((d, m.group(1)) for d in mod_dirs))

        parts = importer_dir.split("/")
        if "src" not in parts:
            return
        crate_src = "/".join(parts[: len(parts) - parts[::-1].index("src")])
        for m in self.USE_CRATE.finditer(text):
            segs = [s for s in m.group(1).split("::") if s]
            # `use crate::a::b::Item` — the longest prefix that is a file wins
            yield from self._first(idx, ((crate_src, "/".join(segs[:k])) for k in range(len(segs), 0, -1)))


# Importer file extension → resolver; files with other extensions are not scanned
_RESOLVERS: dict[str, _ImportResolver] = {ext: r for r in (
    _PythonResolver(), _GoResolver(), _RustResolver(),
    # Bare JS specifiers are npm packages
    _ImportResolver((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
                    re.compile(r"""(?:\bfrom\s+|\bimport\s*\(?\s*|\brequire\s*\(\s*)['"]([^'"]+)['"]"""),
                    targets=("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
                             "/index.ts", "/index.tsx", "/index.js", "/index.jsx"),
                    clean=lambda s: s if s.startswith(".") else None),
    _ImportResolver((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".m", ".mm"),
                    re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.M),
                    by_suffix=True, clean=lambda s: s.replace("\\", "/")),
    # Java / Kotlin imports name packages, not paths: match by path suffix only
    _ImportResolver((".java", ".kt"), re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;?", re.M),
                    targets=(".java", ".kt"), relative=False, by_suffix=True,
                    clean=lambda s: None if s.endswith(".*") else s.replace(".", "/")),
    _ImportResolver((".rb",), re.compile(r"\brequire_relative\s*\(?\s*['\"]([^'\"]+)['\"]"),
                    targets=("", ".rb")),
) for ext in r.exts}
