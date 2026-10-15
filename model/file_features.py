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
import posixpath
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

# Dependency manifests that use a doc-like extension (.txt) but are config
_MANIFEST_PATTERN = re.compile(r"^(requirements|constraints)([-_.][\w.-]+)?\.(txt|in)$", re.I)


def ext_group(path: Path) -> str:
    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    for te in TEST_EXTS:
        if name_lower.endswith(te):
            return "test"
    if _MANIFEST_PATTERN.match(name_lower):
        return "config"
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

# Source dirs that contain public-facing code.
# include/ holds the public headers of C/C++ projects; client/ and sdk/ hold
# the user-facing SDK of server projects (e.g. milvus client/milvusclient).
_SRC_DIRS     = {"src", "lib", "pkg", "source", "app", "cmd", "api",
                 "include", "client", "clients", "sdk"}

# Directories whose shallow headers form a C/C++ library's public interface
_PUBLIC_HEADER_DIRS = {"include"}
# Directories holding a client SDK; their shallow code is what users call
_SDK_DIRS = {"client", "clients", "sdk", "sdks"}

# Deployment / operations material.  Partly user-facing for server projects
# (install via docker/binary) but mostly upgrade, migration, monitoring and
# log-export runbooks, so it gets a mild penalty rather than exclusion.
_OPS_DIRS = {"deployments", "deployment", "deploy", "deploys", "ops",
             "helm", "charts", "k8s", "kubernetes", "terraform", "ansible"}
_HEADER_EXTS = {".h", ".hpp", ".hh", ".hxx"}

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

# CI / DevOps / tooling dirs — developer-only, not user-facing.
# "tools" is deliberately absent: in many projects it holds the user-facing
# binaries (llama.cpp tools/server, tools/cli), so it is treated as neutral.
_DEVOPS_DIRS  = {".github", ".gitlab", ".circleci", ".travis", "scripts",
                 "hack", "contrib", "devtools", ".devcontainer",
                 "ci", "githooks", ".husky"}

# Sub-directories of docs/ written for maintainers rather than users:
# design records, contributor guides, guides for coding agents.
_MAINTAINER_DOC_DIRS = {"design-docs", "design_docs", "design", "designs",
                        "developer_guides", "developer-guides", "developer",
                        "development", "dev", "contributing", "internals",
                        "rfc", "rfcs", "adr", "adrs", "proposals",
                        "agent_guides", "agents"}


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
    "index.ts", "index.js",  # module entry points
    "doc.go",               # Go package overview (godoc)
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
    in_devops         under .github/scripts/ci  ← developer-only
    in_maintainer_docs under docs/design-docs, docs/development, …  ← contributor docs
    in_ops            under deployments/, helm/, k8s/  ← operator runbooks
    in_ignored        under vendor/node_modules/build (noise)
    is_public_api     __init__.py, index.ts, mod.rs, lib.rs
    is_cli_entry      main.py, cli.py, __main__.py and equivalents
    is_user_config    pyproject.toml, package.json, requirements.txt, .env.example
                      — at the repo root only
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
            "in_maintainer_docs": 0, "in_ops": 0,
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
    in_maintainer_docs = int(_in_maintainer_docs(path))  # docs/design-docs/, docs/development/
    in_ops      = int(_any_part_in(path, _OPS_DIRS))     # deployments/, helm/, k8s/

    # How deep is this file *within* its source subtree?
    # A file at src/client.py has src_depth=1; src/auth/internal/token.py has
    # src_depth=3. Shallower = more likely to be public interface.
    src_depth = _src_relative_depth(path)

    # Public surface: package entry files, plus shallow headers under include/
    # (include/llama.h is the whole public API of llama.cpp).
    is_public_api   = int(
        name in _PUBLIC_API_NAMES
        or (path.suffix.lower() in _HEADER_EXTS
            and _any_part_in(path, _PUBLIC_HEADER_DIRS)
            and src_depth <= 1)
        # A package's namesake file: client/milvusclient/client.go is not,
        # but gguf-py/gguf/gguf.py and tools/server/server.cpp are.
        or (group == "code" and depth > 0 and stem == path.parent.name.lower())
        # client/milvusclient/collection.go — the SDK surface of a server repo
        or (group == "code" and _any_part_in(path, _SDK_DIRS) and src_depth <= 1
            and not _TEST_PATTERN.search(stem))
    )
    is_cli_entry    = int(name in _CLI_NAMES)
    # Only the root manifest tells a user how to install the project; nested
    # ones (cmd/tools/x/requirements.txt, examples/y/package.json) belong to
    # internal tools or individual samples.
    is_user_config  = int(name in _USER_CONFIG_NAMES and depth == 0)
    is_dev_config   = int(name in _DEV_CONFIG_NAMES)
    is_readme       = int(bool(_README_PATTERN.search(name)))
    # Go's *_example_test.go files are godoc runnable examples — usage
    # documentation that merely lives in a test file.
    is_go_example   = name.endswith("example_test.go")
    is_test_file    = int(not is_go_example and (bool(_TEST_PATTERN.search(stem)) or group == "test"))
    is_example_file = int(bool(_EXAMPLE_PATTERN.search(stem)))

    # Private module: single leading underscore but not __init__, __main__ etc.
    is_private_module = int(
        stem.startswith("_") and not stem.startswith("__")
    )
    is_internal_name  = int(bool(_INTERNAL_NAME_PATTERN.search(stem)))
    is_public_api_name = int(bool(_PUBLIC_API_NAME_PATTERN.search(stem)))

    rule_score = _rule_score(
        group=group,
        in_docs=in_docs, in_examples=in_examples,
        in_src=in_src, in_internal=in_internal,
        in_tests=in_tests, in_devops=in_devops, in_ignored=in_ignored,
        in_util_dir=in_util_dir, in_maintainer_docs=in_maintainer_docs, in_ops=in_ops,
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
        "in_maintainer_docs": in_maintainer_docs,
        "in_ops":             in_ops,
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


def _in_maintainer_docs(path: Path) -> bool:
    """True for files under docs/<design|development|...>/ — documentation
    aimed at people working *on* the project rather than *with* it."""
    dirs = _parts_lower(path)[:-1]
    for i, part in enumerate(dirs):
        if part in _DOCS_DIRS:
            return any(p in _MAINTAINER_DOC_DIRS for p in dirs[i + 1:])
    return False


def _src_relative_depth(path: Path) -> int:
    """Return depth of the file within its source subtree, or overall depth."""
    parts = [p.lower() for p in path.parts]
    for anchor in ("src", "lib", "pkg", "source", "app", "cmd", "api",
                   "include", "client", "clients", "sdk",
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
    "in_tests", "in_devops", "in_ignored", "in_util_dir", "in_maintainer_docs", "in_ops",
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
    in_tests, in_devops, in_ignored, in_util_dir, in_maintainer_docs, in_ops,
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

    # --- Positive signals (user-facing) ---
    # The strongest signal counts in full; the rest only contribute a fraction.
    # Plain summation saturated at 1.0 for every examples/*/README.md, leaving
    # no room to rank them against each other or against the root README.
    positives = []
    if is_readme:          positives.append(0.40 if depth == 0 else 0.25)  # root README = project overview
    if in_examples:        positives.append(0.35)   # shows users HOW to use the library
    if is_example_file:    positives.append(0.25)   # demo/sample in the filename
    if is_public_api:      positives.append(0.30)   # __init__.py / index.ts / include/*.h
    if is_cli_entry:       positives.append(0.25)   # users invoke this directly
    if in_docs:            positives.append(0.25)   # documentation directory
    if is_user_config:     positives.append(0.20)   # user needs this to install/configure
    if is_public_api_name: positives.append(0.15)   # "client", "sdk", "api" in name

    # Shallow src files are more likely to be the public interface
    if in_src and src_depth == 0: positives.append(0.20)   # e.g. src/client.py
    if in_src and src_depth == 1: positives.append(0.10)   # e.g. src/auth/token.py

    # Top-level scripts are what users run (e.g. llama.cpp convert_hf_to_gguf.py)
    if group == "code" and depth == 0 and not is_test_file:
        positives.append(0.20)

    if positives:
        positives.sort(reverse=True)
        score += positives[0] + 0.3 * sum(positives[1:])

    # --- Penalties (developer-facing / internal) ---
    if in_tests:           score -= 0.35
    if is_test_file:       score -= 0.30
    if in_devops:          score -= 0.30   # .github/, scripts/, tools/
    if is_dev_config:      score -= 0.25   # lockfiles, eslint, jest configs
    if in_internal:        score -= 0.20   # internal/ subdirectory
    if is_private_module:  score -= 0.20   # _private.py convention
    if is_internal_name:   score -= 0.15   # util, helper, mixin, _impl
    if in_util_dir:   score -= 0.15   # util, helper, mixin, _impl
    if in_maintainer_docs: score -= 0.30   # design records / contributor guides
    if in_ops:             score -= 0.15   # deployment / upgrade / migration runbooks
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


# Import statements are parsed per language and resolved to real repo paths.
# Earlier versions matched only the *stem* of whatever was imported, which
# conflated unrelated files (every utils.* got credit for every `import utils`,
# and pytest.ini got credit for every `import pytest`).  We still use regexes
# rather than full parsers — the goal is a cheap, mostly-correct signal.

_PY_FROM_IMPORT  = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+([\w\s,()*]+)", re.M)
_PY_IMPORT       = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)", re.M)
_JS_IMPORT       = re.compile(
    r"""(?:\bfrom\s+|\bimport\s*\(?\s*|\brequire\s*\(\s*)['"]([^'"]+)['"]""")
_GO_IMPORT_BLOCK = re.compile(r"^\s*import\s*\(([^)]*)\)", re.M)
_GO_IMPORT_LINE  = re.compile(r"^\s*import\s+(?:[\w.]+\s+)?\"([^\"]+)\"", re.M)
_GO_QUOTED       = re.compile(r"\"([^\"]+)\"")
_GO_MODULE       = re.compile(r"^\s*module\s+(\S+)", re.M)
_C_INCLUDE       = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.M)
_RS_MOD          = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M)
_RS_USE_CRATE    = re.compile(r"\buse\s+crate::([\w:]+)")
_JVM_IMPORT      = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;?", re.M)
_RB_REQUIRE_REL  = re.compile(r"\brequire_relative\s*\(?\s*['\"]([^'\"]+)['\"]")

_PY_EXTS = (".py", "/__init__.py")
_JS_EXTS = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
            "/index.ts", "/index.tsx", "/index.js", "/index.jsx")
_C_EXTS  = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".cu", ".m", ".mm"}
_SCANNABLE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                   ".go", ".rs", ".java", ".kt", ".rb"} | _C_EXTS


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


def _resolve_python(text, importer_dir, idx):
    specs = []
    for m in _PY_FROM_IMPORT.finditer(text):
        mod, names = m.group(1), m.group(2)
        if mod.strip(".") == "":
            # `from . import a, b` — the names are sibling modules
            specs += [mod + n.strip() for n in re.split(r"[\s,()]+", names) if n.strip() and n.strip() != "*"]
        else:
            specs.append(mod)
    for m in _PY_IMPORT.finditer(text):
        specs += [s.strip() for s in m.group(1).split(",")]

    for spec in specs:
        dots = len(spec) - len(spec.lstrip("."))
        path = spec.lstrip(".").replace(".", "/").lower()
        if dots:
            up = "/".join([".."] * (dots - 1))
            hit = idx.relative(importer_dir, posixpath.join(up, path) if up else path, _PY_EXTS)
        else:
            hit = idx.relative(importer_dir, path, _PY_EXTS) or idx.by_path_suffix(importer_dir, path, _PY_EXTS)
        if hit:
            yield hit


def _resolve_js(text, importer_dir, idx):
    for m in _JS_IMPORT.finditer(text):
        spec = m.group(1).lower()
        if spec.startswith("."):            # bare specifiers are npm packages
            hit = idx.relative(importer_dir, spec, _JS_EXTS)
            if hit:
                yield hit


def _resolve_c(text, importer_dir, idx):
    for m in _C_INCLUDE.finditer(text):
        spec = m.group(1).replace("\\", "/").lower()
        hit = idx.relative(importer_dir, spec, ("",)) or idx.by_path_suffix(importer_dir, spec, ("",))
        if hit:
            yield hit


def _resolve_go(text, importer_dir, idx):
    specs = [m.group(1) for m in _GO_IMPORT_LINE.finditer(text)]
    for block in _GO_IMPORT_BLOCK.finditer(text):
        specs += _GO_QUOTED.findall(block.group(1))
    for spec in specs:
        spec = spec.lower()
        # Longest module prefix wins (nested modules such as milvus pkg/v2)
        best = max((mod for mod in idx.go_modules if spec == mod or spec.startswith(mod + "/")),
                   key=len, default=None)
        if best is None:
            continue                         # external dependency
        pkg_dir = posixpath.normpath(posixpath.join(idx.go_modules[best], spec[len(best):].lstrip("/")))
        # A Go import names a package (directory), so every file in it is used
        yield from idx.go_files_in_dir.get("" if pkg_dir == "." else pkg_dir, [])


def _resolve_rust(text, importer_rel, idx):
    importer_dir = posixpath.dirname(importer_rel)
    stem = posixpath.splitext(posixpath.basename(importer_rel))[0]
    mod_dirs = [importer_dir] if stem in ("mod", "lib", "main") else [importer_dir, posixpath.join(importer_dir, stem)]
    for m in _RS_MOD.finditer(text):
        for d in mod_dirs:
            hit = idx.relative(d, m.group(1), (".rs", "/mod.rs"))
            if hit:
                yield hit
                break

    parts = importer_dir.split("/")
    if "src" not in parts:
        return
    crate_src = "/".join(parts[: len(parts) - parts[::-1].index("src")])
    for m in _RS_USE_CRATE.finditer(text):
        segs = [s for s in m.group(1).split("::") if s]
        # `use crate::a::b::Item` — the longest prefix that is a file wins
        for k in range(len(segs), 0, -1):
            hit = idx.relative(crate_src, "/".join(segs[:k]), (".rs", "/mod.rs"))
            if hit:
                yield hit
                break


def _resolve_jvm(text, importer_dir, idx):
    for m in _JVM_IMPORT.finditer(text):
        spec = m.group(1)
        if spec.endswith(".*"):
            continue
        hit = idx.by_path_suffix(importer_dir, spec.replace(".", "/").lower(), (".java", ".kt"))
        if hit:
            yield hit


def _resolve_ruby(text, importer_dir, idx):
    for m in _RB_REQUIRE_REL.finditer(text):
        hit = idx.relative(importer_dir, m.group(1).lower(), ("", ".rb"))
        if hit:
            yield hit


def _references(ext: str, text: str, importer_rel: str, idx: _RepoIndex):
    importer_dir = posixpath.dirname(importer_rel)
    if ext == ".py":
        return _resolve_python(text, importer_dir, idx)
    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        return _resolve_js(text, importer_dir, idx)
    if ext in _C_EXTS:
        return _resolve_c(text, importer_dir, idx)
    if ext == ".go":
        return _resolve_go(text, importer_dir, idx)
    if ext == ".rs":
        return _resolve_rust(text, importer_rel, idx)
    if ext in (".java", ".kt"):
        return _resolve_jvm(text, importer_dir, idx)
    if ext == ".rb":
        return _resolve_ruby(text, importer_dir, idx)
    return ()


def _counts_as_referrer(rel: str) -> bool:
    """References from tests, CI scripts or vendored code say nothing about
    what a *user* of the project touches, so they are not counted.
    Go *_example_test.go files are usage docs and do count."""
    path = Path(rel)
    if _any_part_in(path, _TEST_DIRS | _IGNORE_DIRS | _DEVOPS_DIRS):
        return False
    if path.name.lower().endswith("example_test.go"):
        return True
    return not (_TEST_PATTERN.search(path.stem.lower()) or ext_group(path) == "test")


def _is_example_referrer(rel: str) -> bool:
    """Files that show users how to call the project: examples/ trees,
    *_example.py style samples and Go *_example_test.go."""
    path = Path(rel)
    if _any_part_in(path, _IGNORE_DIRS):
        return False
    return _any_part_in(path, _EXAMPLE_DIRS) or bool(_EXAMPLE_PATTERN.search(path.stem.lower()))


def build_ref_counts(repo_root: str, file_list: list[str],
                     example_counts: dict[str, int] | None = None) -> dict[str, int]:
    """Scan source files in file_list for import/include statements and return
    a dict mapping relative path → number of distinct files that reference it.

    Specifiers are resolved to concrete repo paths per language (relative
    imports against the importer's directory, absolute ones by path suffix,
    Go imports via the module paths declared in go.mod).  Ambiguous or
    external specifiers are dropped instead of being credited to every file
    with a matching name.

    Parameters
    ----------
    repo_root : absolute path to the cloned repository root
    file_list : list of relative paths (as returned by list_repo_files or os.walk)
    example_counts : optional dict, filled in place with rel_path → number of
                example files referencing it.  Unlike the overall count this
                singles out what *users* call: a server repo's internal pkg/
                is imported everywhere, its SDK mostly by examples.

    Returns
    -------
    dict mapping rel_path → int  (0 for files never referenced)
    """
    root = Path(repo_root)
    counts: dict[str, int] = {f: 0 for f in file_list}
    idx = _RepoIndex(file_list)

    for n, rel in idx.by_norm.items():
        if posixpath.basename(n) == "go.mod":
            try:
                m = _GO_MODULE.search((root / rel).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if m:
                idx.go_modules[m.group(1).lower()] = posixpath.dirname(n)

    if example_counts is not None:
        example_counts.update({f: 0 for f in file_list})

    for rel in file_list:
        ext = Path(rel).suffix.lower()
        if ext not in _SCANNABLE_EXTS:
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
        targets = set(_references(ext, text, importer, idx))
        targets.discard(importer)           # don't count self-references
        for target in targets:
            orig = idx.by_norm[target]
            if general:
                counts[orig] = counts.get(orig, 0) + 1
            if example:
                example_counts[orig] = example_counts.get(orig, 0) + 1

    return counts
