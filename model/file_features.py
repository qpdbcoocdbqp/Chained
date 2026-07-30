"""
file_features.py — File path feature extraction and rule-based scoring.

Scoring philosophy: USER perspective, not developer perspective.
A high score means "a user of this library would benefit from seeing this file
in a cheatsheet." Low score means "this is an internal implementation detail
that only a maintainer of the library would care about."

High priority (user-facing):
  - Public API surface: __init__.py, shallow package modules, exported interfaces
  - Usage examples: examples/, tutorials/, notebooks/, demos/
  - README and user-facing docs
  - CLI entry points users actually invoke

Low priority (developer-facing, despite being code):
  - Private modules (name starts with _)
  - Internal helpers: utils, helpers, mixins, base classes
  - Build tooling, CI config, internal scripts
  - Deep implementation files (many levels nested)
  - Test code
"""

import re
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Extension groups
# ---------------------------------------------------------------------------

CODE_EXTS = {
    ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs", ".java", ".kt",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".swift", ".rb", ".php", ".scala",
    ".clj", ".ex", ".exs", ".lua", ".r", ".jl",
}

CONFIG_EXTS = {
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".conf",
    ".env", ".properties", ".xml",
}

DOC_EXTS = {".md", ".mdx", ".rst", ".txt", ".adoc", ".wiki"}

DATA_EXTS = {
    ".csv", ".tsv", ".parquet", ".arrow", ".pkl", ".bin", ".so",
    ".dylib", ".dll", ".exe", ".wasm", ".zip", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".pdf",
    ".lock",
}

TEST_EXTS = {".test.ts", ".spec.ts", ".test.js", ".spec.js"}


def ext_group(path: Path) -> str:
    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    for te in TEST_EXTS:
        if name_lower.endswith(te):
            return "test"
    if suffix in CODE_EXTS:
        return "code"
    if suffix in CONFIG_EXTS:
        return "config"
    if suffix in DOC_EXTS:
        return "doc"
    if suffix in DATA_EXTS:
        return "data"
    return "other"


# ---------------------------------------------------------------------------
# Directory signal patterns
# ---------------------------------------------------------------------------

_DOCS_DIRS    = {"docs", "doc", "documentation", "wiki", "guides", "guide"}
_EXAMPLE_DIRS = {"examples", "example", "samples", "sample", "demo", "demos",
                 "tutorials", "tutorial", "cookbook", "notebooks", "quickstart"}

# Source dirs that contain public-facing code
_SRC_DIRS     = {"src", "lib", "pkg", "source", "app", "cmd", "api"}

# Directories that are clearly implementation internals (user doesn't need to
# read these to use the library)
_INTERNAL_DIRS = {"internal", "internals", "_internal", "private",
                  "core", "engine", "backend"}

# Noise: auto-generated, third-party, or build artifacts
_IGNORE_DIRS  = {"node_modules", "vendor", ".git", "__pycache__", ".venv",
                 "venv", ".tox", "dist", "build", "target", ".cache",
                 ".mypy_cache", ".ruff_cache", "coverage", ".pytest_cache",
                 "site-packages", "eggs", ".eggs", "__mocks__", "fixtures",
                 "testdata", "test_data", "mock", "mocks"}

_TEST_DIRS    = {"test", "tests", "spec", "specs", "__tests__", "e2e",
                 "integration", "unit"}

# CI / DevOps / tooling dirs — developer-only, not user-facing
_DEVOPS_DIRS  = {".github", ".gitlab", ".circleci", ".travis", "scripts",
                 "tools", "hack", "contrib", "devtools", ".devcontainer"}


def _parts_lower(path: Path) -> list[str]:
    return [p.lower() for p in path.parts]


def _any_part_in(path: Path, names: set) -> bool:
    return any(p in names for p in _parts_lower(path))


# ---------------------------------------------------------------------------
# Filename signals — user perspective
# ---------------------------------------------------------------------------

# Files a user actually imports or invokes directly
_PUBLIC_API_NAMES = {
    "__init__.py",          # defines the importable package surface
    "mod.rs", "lib.rs",     # Rust crate root
    "index.ts", "index.js", "index.go",  # module entry points
}

# CLI entry points users actually run
_CLI_NAMES = {
    "__main__.py",
    "cli.py", "cli.go", "cli.ts",
    "main.py", "main.go", "main.ts", "main.rs", "main.js",
    "app.py",  "app.go",  "app.ts",
}

# Manifests/configs a USER reads to understand installation and dependencies
_USER_CONFIG_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg",
    "go.mod",
    "cargo.toml",
    "package.json",
    "requirements.txt",   # user needs this to install
    ".env.example",       # user copies this to configure
}

# Developer-only config: build, CI, linting — users don't need these
_DEV_CONFIG_NAMES = {
    "go.sum", "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    "makefile", "cmake", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.yml",
    ".prettierrc", ".prettierrc.js", ".prettierrc.json",
    ".babelrc", "babel.config.js", "babel.config.json",
    "jest.config.js", "jest.config.ts", "vitest.config.ts",
    "webpack.config.js", "vite.config.ts", "rollup.config.js",
    ".flake8", ".pylintrc", "mypy.ini", "ruff.toml",
    ".gitignore", ".gitattributes", ".editorconfig",
    "codecov.yml", "sonar-project.properties",
    "composer.json", "build.gradle", "pom.xml",
    "cmakelists.txt"
}

# Directory-level internal signal: sitting inside utils/, helpers/, base/ etc.
# indicates developer-internal code even when the filename itself looks neutral.
_UTIL_DIRS = {"utils", "util", "helpers", "helper", "base", "bases",
              "abstract", "abstracts", "compat", "shims", "mixins", "mixin",
              "common", "shared", "support", "lib_internal"}

_README_PATTERN  = re.compile(r"readme|getting.?started|quickstart", re.I)
_TEST_PATTERN    = re.compile(r"(^|[_\-])test[s]?([_\-.]|$)|(^|[_\-])spec([_\-.]|$)", re.I)
_EXAMPLE_PATTERN = re.compile(r"(example|sample|demo|tutorial|cookbook|howto|usage)", re.I)

# Filename stem patterns that signal internal / helper code
_INTERNAL_NAME_PATTERN = re.compile(
    r"(^_[^_]|"                    # starts with single _ (Python private convention)
    r"util[s]?$|helper[s]?$|"      # files literally named utils.py, helpers.go
    r"mixin[s]?|"
    r"abstract_|_abstract|^abstract|"  # abstract base classes
    r"^base_|_base$|^base\b|"      # base classes
    r"_impl$|_internal$|"
    r"compat$|shim[s]?$|"
    r"constant[s]?$|const[s]?$|"
    r"exception[s]?$|error[s]?$|"
    r"migration[s]?$|schema[s]?$)",
    re.I,
)

# Filename patterns that signal user-facing API code
_PUBLIC_API_NAME_PATTERN = re.compile(
    r"(client|sdk|api|interface|"
    r"model[s]?|type[s]?|"   # public data models / types
    r"handler[s]?|route[s]?|"
    r"plugin[s]?|extension[s]?|"
    r"command[s]?|action[s]?)",
    re.I,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(rel_path: str, ref_count: int = 0) -> dict:
    """
    Extract a flat feature dict from a relative file path string (user-perspective).

    Parameters
    ----------
    rel_path  : relative file path within the repo
    ref_count : number of other files in the repo that import/require this file
                (computed by build_ref_counts); defaults to 0 when not available

    Features
    --------
    ext_group_*       one-hot: {code, config, doc, data, test, other}
    depth             directory depth (0 = repo root)
    in_docs           under a docs-like directory
    in_examples       under examples/tutorial/demo directory  ← high user value
    in_src            under src/lib/pkg  (public source)
    in_internal       under internal/_internal/private dirs   ← low user value
    in_tests          under test/spec directory
    in_devops         under .github/scripts/tools  ← developer-only
    in_ignored        under vendor/node_modules/build (noise)
    is_public_api     __init__.py, index.ts, mod.rs, lib.rs
    is_cli_entry      main.py, cli.py, __main__.py and equivalents
    is_user_config    pyproject.toml, package.json, requirements.txt, .env.example
    is_dev_config     lockfiles, eslint, jest, dockerfile, CI configs
    is_readme         README, getting-started, quickstart
    is_test_file      filename contains test/spec pattern
    is_example_file   filename contains example/sample/demo/usage
    is_private_module filename starts with _ (Python private convention)
    is_internal_name  name matches util/helper/mixin/impl/compat patterns
    is_public_api_name name matches client/sdk/api/model/handler patterns
    src_depth         depth *within* the src/lib subtree (0 if not in src)
    ref_count         raw import/require reference count from other files
    ref_count_log     log1p(ref_count) — normalised, model-friendly
    rule_score        combined user-perspective score in [0, 1]
    """
    path = Path(rel_path)
    name = path.name.lower()
    stem = path.stem.lower()
    group = ext_group(path)

    # Hard blacklist: filenames that are never useful for a cheatsheet regardless
    # of directory context (build scripts, lock files, editor configs, etc.).
    # Import lazily to avoid a circular dependency with filescore.
    from filescore import SKIP_NAMES
    if name in SKIP_NAMES:
        all_groups = ("code", "config", "doc", "data", "test", "other")
        ext_oh = {f"ext_group_{g}": int(group == g) for g in all_groups}
        return {
            **ext_oh,
            "depth": max(0, len(path.parts) - 1),
            "in_docs": 0, "in_examples": 0, "in_src": 0, "in_internal": 0,
            "in_tests": 0, "in_devops": 0, "in_ignored": 1, "in_util_dir": 0,
            "is_public_api": 0, "is_cli_entry": 0, "is_user_config": 0,
            "is_dev_config": 1, "is_readme": 0, "is_test_file": 0,
            "is_example_file": 0, "is_private_module": 0,
            "is_internal_name": 0, "is_public_api_name": 0,
            "src_depth": 0, "ref_count": ref_count,
            "ref_count_log": math.log1p(ref_count),
            "rule_score": 0.0,
        }

    all_groups = ("code", "config", "doc", "data", "test", "other")
    ext_oh = {f"ext_group_{g}": int(group == g) for g in all_groups}

    depth       = max(0, len(path.parts) - 1)
    in_docs     = int(_any_part_in(path, _DOCS_DIRS))
    in_examples = int(_any_part_in(path, _EXAMPLE_DIRS))
    in_src      = int(_any_part_in(path, _SRC_DIRS))
    in_internal = int(_any_part_in(path, _INTERNAL_DIRS))
    in_tests    = int(_any_part_in(path, _TEST_DIRS))
    in_devops   = int(_any_part_in(path, _DEVOPS_DIRS))
    in_ignored  = int(_any_part_in(path, _IGNORE_DIRS))
    in_util_dir = int(_any_part_in(path, _UTIL_DIRS))  # utils/, helpers/, base/, compat/

    is_public_api   = int(name in _PUBLIC_API_NAMES)
    is_cli_entry    = int(name in _CLI_NAMES)
    is_user_config  = int(name in _USER_CONFIG_NAMES)
    is_dev_config   = int(name in _DEV_CONFIG_NAMES)
    is_readme       = int(bool(_README_PATTERN.search(name)))
    is_test_file    = int(bool(_TEST_PATTERN.search(stem)) or group == "test")
    is_example_file = int(bool(_EXAMPLE_PATTERN.search(stem)))

    # Private module: single leading underscore but not __init__, __main__ etc.
    is_private_module = int(
        stem.startswith("_") and not stem.startswith("__")
    )
    is_internal_name  = int(bool(_INTERNAL_NAME_PATTERN.search(stem)))
    is_public_api_name = int(bool(_PUBLIC_API_NAME_PATTERN.search(stem)))

    # How deep is this file *within* its source subtree?
    # A file at src/client.py has src_depth=1; src/auth/internal/token.py has
    # src_depth=3. Shallower = more likely to be public interface.
    src_depth = _src_relative_depth(path)

    rule_score = _rule_score(
        group=group,
        in_docs=in_docs, in_examples=in_examples,
        in_src=in_src, in_internal=in_internal,
        in_tests=in_tests, in_devops=in_devops, in_ignored=in_ignored,
        in_util_dir=in_util_dir,
        is_public_api=is_public_api, is_cli_entry=is_cli_entry,
        is_user_config=is_user_config, is_dev_config=is_dev_config,
        is_readme=is_readme, is_test_file=is_test_file,
        is_example_file=is_example_file,
        is_private_module=is_private_module,
        is_internal_name=is_internal_name,
        is_public_api_name=is_public_api_name,
        depth=depth, src_depth=src_depth,
    )

    return {
        **ext_oh,
        "depth":              depth,
        "in_docs":            in_docs,
        "in_examples":        in_examples,
        "in_src":             in_src,
        "in_internal":        in_internal,
        "in_tests":           in_tests,
        "in_devops":          in_devops,
        "in_ignored":         in_ignored,
        "in_util_dir":        in_util_dir,
        "is_public_api":      is_public_api,
        "is_cli_entry":       is_cli_entry,
        "is_user_config":     is_user_config,
        "is_dev_config":      is_dev_config,
        "is_readme":          is_readme,
        "is_test_file":       is_test_file,
        "is_example_file":    is_example_file,
        "is_private_module":  is_private_module,
        "is_internal_name":   is_internal_name,
        "is_public_api_name": is_public_api_name,
        "src_depth":          src_depth,
        "ref_count":          ref_count,
        "ref_count_log":      math.log1p(ref_count),
        "rule_score":         rule_score,
    }


def _src_relative_depth(path: Path) -> int:
    """Return depth of the file within its source subtree, or overall depth."""
    parts = [p.lower() for p in path.parts]
    for anchor in ("src", "lib", "pkg", "source", "app", "cmd", "api",
                   "examples", "example", "docs", "doc"):
        if anchor in parts:
            idx = parts.index(anchor)
            # depth within the subtree = parts after anchor, minus 1 for filename
            return max(0, len(parts) - idx - 2)
    return max(0, len(parts) - 1)


# Feature column order — must stay stable so the trained model can be loaded
# and used consistently across versions.
FEATURE_COLUMNS = [
    "ext_group_code", "ext_group_config", "ext_group_doc",
    "ext_group_data", "ext_group_test", "ext_group_other",
    "depth", "src_depth",
    "in_docs", "in_examples", "in_src", "in_internal",
    "in_tests", "in_devops", "in_ignored", "in_util_dir",
    "is_public_api", "is_cli_entry", "is_user_config", "is_dev_config",
    "is_readme", "is_test_file", "is_example_file",
    "is_private_module", "is_internal_name", "is_public_api_name",
    "ref_count", "ref_count_log",
    "rule_score",
]


def features_to_vector(features: dict) -> list:
    """Return feature values as an ordered list matching FEATURE_COLUMNS."""
    return [features[col] for col in FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Rule-based scorer  (user perspective)
# ---------------------------------------------------------------------------

def _rule_score(
    group,
    in_docs, in_examples, in_src, in_internal,
    in_tests, in_devops, in_ignored, in_util_dir,
    is_public_api, is_cli_entry, is_user_config, is_dev_config,
    is_readme, is_test_file, is_example_file,
    is_private_module, is_internal_name, is_public_api_name,
    depth, src_depth,
) -> float:
    """
    Heuristic score in [0, 1] from the USER's perspective.

    Question being answered: "Would a user of this library benefit from
    seeing this file's content in a developer cheatsheet?"
    """
    if in_ignored:
        return 0.0

    score = 0.0

    # --- Base score by content type ---
    # Code is only valuable if it exposes something to the user;
    # doc/example files are valuable by nature.
    group_base = {
        "code":   0.35,   # start neutral — need other signals to go high
        "config": 0.20,   # most configs are dev-facing; user configs get boosts below
        "doc":    0.55,   # docs tend to be user-facing by default
        "data":   0.05,
        "test":   0.05,
        "other":  0.05,
    }
    score += group_base.get(group, 0.05)

    # --- Strong positive signals (user-facing) ---
    if is_readme:          score += 0.40   # highest: overview for new users
    if in_examples:        score += 0.35   # shows users HOW to use the library
    if is_example_file:    score += 0.25   # demo/sample in the filename
    if is_public_api:      score += 0.30   # __init__.py / index.ts = public surface
    if is_cli_entry:       score += 0.25   # users invoke this directly
    if in_docs:            score += 0.25   # documentation directory
    if is_user_config:     score += 0.20   # user needs this to install/configure
    if is_public_api_name: score += 0.15   # "client", "sdk", "api" in name

    # Shallow src files are more likely to be the public interface
    if in_src and src_depth == 0: score += 0.20   # e.g. src/client.py
    if in_src and src_depth == 1: score += 0.10   # e.g. src/auth/token.py

    # --- Penalties (developer-facing / internal) ---
    if in_tests:           score -= 0.35
    if is_test_file:       score -= 0.30
    if in_devops:          score -= 0.30   # .github/, scripts/, tools/
    if is_dev_config:      score -= 0.25   # lockfiles, eslint, jest configs
    if in_internal:        score -= 0.20   # internal/ subdirectory
    if is_private_module:  score -= 0.20   # _private.py convention
    if is_internal_name:   score -= 0.15   # util, helper, mixin, _impl
    if in_util_dir:   score -= 0.15   # util, helper, mixin, _impl
    if group == "config" and not is_user_config: score -= 0.10

    # Depth penalty: very deeply nested files are almost always implementation
    # details. Applied on top of src_depth to doubly penalise buried code.
    score -= depth * 0.04
    if src_depth >= 3:     score -= 0.16   # extra penalty for deeply nested src

    return max(0.0, min(1.0, score))


def score_file(rel_path: str) -> float:
    """Convenience function: return just the rule-based score for a path."""
    return extract_features(rel_path)["rule_score"]


# ---------------------------------------------------------------------------
# Reference count builder
# ---------------------------------------------------------------------------

# Patterns that capture the imported module stem from a source line.
# We intentionally keep these broad (grep-style) rather than full AST parsing —
# the goal is a cheap relative signal, not a perfect dependency graph.
_IMPORT_PATTERNS = [
    # Python:  from .foo import …  |  from foo.bar import …  |  import foo.bar
    re.compile(r"""(?:from\s+([\w./\\-]+)\s+import|import\s+([\w./\\-]+))"""),
    # JS/TS:   import … from './foo'  |  require('./foo')
    re.compile(r"""(?:from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))"""),
    # Go:      import "github.com/foo/bar"
    re.compile(r"""import\s+["'`]([^"'`]+)["'`]"""),
    # Rust:    mod foo;  |  use foo::bar;
    re.compile(r"""(?:^mod\s+(\w+)\s*;|use\s+([\w:]+))""", re.MULTILINE),
]

_SCANNABLE_EXTS = {
    ".py", ".go", ".ts", ".tsx", ".js", ".jsx",
    ".rs", ".java", ".kt", ".rb", ".php", ".swift",
    ".c", ".cpp", ".cc", ".h", ".hpp",
}


def build_ref_counts(repo_root: str, file_list: list[str]) -> dict[str, int]:
    """Scan every scannable file in file_list for import/require statements and
    return a dict mapping relative path → number of files that reference it.

    The matching is intentionally fuzzy: we extract the stem of whatever is
    being imported and check whether any file in the repo has a matching stem.
    This avoids the need to resolve full module paths (which requires language-
    specific tooling) while still capturing the most useful signal.

    Parameters
    ----------
    repo_root : absolute path to the cloned repository root
    file_list : list of relative paths (as returned by list_repo_files or os.walk)

    Returns
    -------
    dict mapping rel_path → int  (0 for files never referenced)
    """
    root = Path(repo_root)
    counts: dict[str, int] = {f: 0 for f in file_list}

    # Build a quick lookup: stem (lower) → [rel_path, ...]
    # We use the stem because import statements rarely include the extension.
    stem_to_paths: dict[str, list[str]] = {}
    for rel in file_list:
        stem = Path(rel).stem.lower()
        stem_to_paths.setdefault(stem, []).append(rel)

    for rel in file_list:
        p = root / rel
        if p.suffix.lower() not in _SCANNABLE_EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        referenced_stems: set[str] = set()
        for pat in _IMPORT_PATTERNS:
            for m in pat.finditer(text):
                for raw in m.groups():
                    if not raw:
                        continue
                    # Normalise separators and take the last component as stem
                    stem = Path(raw.replace("\\", "/")).stem.lower()
                    # Skip obvious non-local references (e.g. "os", "sys", "react")
                    if len(stem) >= 2:
                        referenced_stems.add(stem)

        for stem in referenced_stems:
            for target_rel in stem_to_paths.get(stem, []):
                if target_rel != rel:           # don't count self-references
                    counts[target_rel] = counts.get(target_rel, 0) + 1

    return counts
