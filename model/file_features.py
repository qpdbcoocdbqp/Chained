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

All directory / name lists, patterns and score weights live in config.yaml.
"""

import re
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")

EXT_GROUPS = ("code", "config", "doc", "data", "test", "other")
_FLAG_FEATURES = ("is_public_api", "is_cli_entry", "is_user_config", "is_dev_config",
                  "is_readme", "is_test_file", "is_example_file", "is_private_module",
                  "is_internal_name", "is_public_api_name")


class FeatureExtraction:
    """
    Path features and the rule-based score for a repo file, configured by
    config.yaml.

    Features (extract)
    --------
    ext_group_*        one-hot: {code, config, doc, data, test, other}
    depth              directory depth (0 = repo root)
    in_<dir>           under a docs / examples / src / internal / tests /
                       devops / ignored / util / ops directory
    in_maintainer_docs under docs/design-docs, docs/development, …
    is_public_api      __init__.py, index.ts, mod.rs, shallow include/*.h,
                       a package's namesake file, shallow SDK code
    is_cli_entry       main.py, cli.py, __main__.py and equivalents
    is_user_config     pyproject.toml, package.json, … — at the repo root only
    is_dev_config      lockfiles, eslint, jest, dockerfile, CI configs
    is_readme          README, getting-started, quickstart
    is_test_file       file name matches the test pattern (not Go *_example_test.go)
    is_example_file    file name contains example/sample/demo/usage
    is_private_module  file name starts with a single _
    is_internal_name   name matches util/helper/mixin/impl/compat patterns
    is_public_api_name name matches client/sdk/api/model/handler patterns
    src_depth          depth *within* the src/lib subtree
    rule_score         combined user-perspective score in [0, 1]
    """

    def __init__(self, config_path: str | Path = CONFIG_PATH):
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.feature_columns: list[str] = cfg["feature_columns"]
        self.exts = {k: set(v) for k, v in cfg["extensions"].items()}
        # in_dirs keys become in_<key> features; helper_dirs are lookups only
        self.in_dirs: list[str] = list(cfg["in_dirs"])
        self.dirs = {k: set(v) for k, v in {**cfg["in_dirs"], **cfg["helper_dirs"]}.items()}
        self.names = {k: set(v) for k, v in cfg["names"].items()}
        self.skip_names: set[str] = set(cfg["skip"]["names"])
        self.skip_exts: set[str] = set(cfg["skip"]["extensions"])
        self.patterns = {
            k: re.compile("|".join(v) if isinstance(v, list) else v, re.I)
            for k, v in cfg["patterns"].items()
        }
        self.src_anchors: list[str] = cfg["src_depth_anchors"]
        self.weights = cfg["rule_score"]

    # -- features -----------------------------------------------------------

    def ext_group(self, path: Path) -> str:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if any(name.endswith(te) for te in self.exts["test"]):
            return "test"
        if self.patterns["manifest"].match(name):
            return "config"
        for group in ("code", "config", "doc", "data"):
            if suffix in self.exts[group]:
                return group
        return "other"

    def extract(self, rel_path: str) -> dict:
        """Flat feature dict for a relative file path (see the class docstring)."""
        path = Path(rel_path)
        parts = [p.lower() for p in path.parts]
        under = set(parts)
        name, stem, suffix = path.name.lower(), path.stem.lower(), path.suffix.lower()
        group = self.ext_group(path)
        depth = max(0, len(parts) - 1)

        f = {f"ext_group_{g}": int(group == g) for g in EXT_GROUPS}
        f["depth"] = depth

        # Hard blacklist: file names never useful for a cheatsheet regardless of
        # directory (build scripts, lock files, editor configs, …)
        if name in self.skip_names:
            f.update({f"in_{d}": 0 for d in self.in_dirs}, in_maintainer_docs=0)
            f.update(dict.fromkeys(_FLAG_FEATURES, 0))
            return {**f, "in_ignored": 1, "is_dev_config": 1, "src_depth": 0, "rule_score": 0.0}

        f.update({f"in_{d}": int(bool(self.dirs[d] & under)) for d in self.in_dirs})
        f["in_maintainer_docs"] = int(self._in_maintainer_docs(parts))

        # Depth within the source subtree: src/client.py → 0,
        # src/auth/internal/token.py → 2. Shallower = more likely public.
        src_depth = self._src_depth(parts)
        test_name = bool(self.patterns["test"].search(stem))
        f["is_public_api"] = int(bool(
            name in self.names["public_api"]
            # include/llama.h is the whole public API of llama.cpp
            or (suffix in self.exts["header"] and self.dirs["public_header"] & under
                and src_depth <= 1)
            # A package's namesake file: gguf-py/gguf/gguf.py, tools/server/server.cpp
            or (group == "code" and depth > 0 and stem == path.parent.name.lower())
            # client/milvusclient/collection.go — the SDK surface of a server repo
            or (group == "code" and self.dirs["sdk"] & under and src_depth <= 1
                and not test_name)
        ))
        f["is_cli_entry"] = int(name in self.names["cli"])
        # Only the root manifest tells a user how to install the project; nested
        # ones belong to internal tools or individual samples.
        f["is_user_config"] = int(name in self.names["user_config"] and depth == 0)
        f["is_dev_config"] = int(name in self.names["dev_config"])
        f["is_readme"] = int(bool(self.patterns["readme"].search(name)))
        # Go *_example_test.go files are godoc runnable examples — usage docs
        # that merely live in a test file.
        f["is_test_file"] = int(not name.endswith("example_test.go")
                                and (test_name or group == "test"))
        f["is_example_file"] = int(bool(self.patterns["example"].search(stem)))
        f["is_private_module"] = int(stem.startswith("_") and not stem.startswith("__"))
        f["is_internal_name"] = int(bool(self.patterns["internal_name"].search(stem)))
        f["is_public_api_name"] = int(bool(self.patterns["public_api_name"].search(stem)))
        f["src_depth"] = src_depth
        f["rule_score"] = self.rule_score(f, group)
        return f

    def _in_maintainer_docs(self, parts: list[str]) -> bool:
        """True for files under docs/<design|development|...>/ — documentation
        aimed at people working *on* the project rather than *with* it."""
        dirs = parts[:-1]
        for i, part in enumerate(dirs):
            if part in self.dirs["docs"]:
                return any(p in self.dirs["maintainer_docs"] for p in dirs[i + 1:])
        return False

    def _src_depth(self, parts: list[str]) -> int:
        for anchor in self.src_anchors:
            if anchor in parts:
                # parts after the anchor, minus 1 for the file name
                return max(0, len(parts) - parts.index(anchor) - 2)
        return max(0, len(parts) - 1)

    def to_vector(self, features: dict) -> list:
        """Feature values in feature_columns order, as fed to the ML model."""
        return [features[col] for col in self.feature_columns]

    # -- rule-based score ---------------------------------------------------

    def rule_score(self, f: dict, group: str) -> float:
        """Heuristic answer to "would a user of this library benefit from seeing
        this file in a developer cheatsheet?", weighted by config.yaml."""
        if f["in_ignored"]:
            return 0.0
        w = self.weights
        signals = {
            **f,
            "readme_root":     f["is_readme"] and f["depth"] == 0,
            "readme_nested":   f["is_readme"] and f["depth"] > 0,
            "src_depth_0":     f["in_src"] and f["src_depth"] == 0,
            "src_depth_1":     f["in_src"] and f["src_depth"] == 1,
            "root_script":     group == "code" and f["depth"] == 0 and not f["is_test_file"],
            "non_user_config": group == "config" and not f["is_user_config"],
            "deep_src":        f["src_depth"] >= 3,
        }
        score = w["group_base"].get(group, w["group_base"]["other"])

        positives = sorted((v for k, v in w["positive"].items() if signals[k]), reverse=True)
        if positives:
            score += positives[0] + w["secondary_positive_factor"] * sum(positives[1:])
        for k, v in w["penalty"].items():
            if signals[k]:
                score -= v * signals[k]
        return max(0.0, min(1.0, score))

    def score_file(self, rel_path: str) -> float:
        """Just the rule-based score for a path."""
        return self.extract(rel_path)["rule_score"]


# ---------------------------------------------------------------------------
# Module-level API backed by config.yaml
# ---------------------------------------------------------------------------

features = FeatureExtraction()
FEATURE_COLUMNS = features.feature_columns
extract_features = features.extract
features_to_vector = features.to_vector
score_file = features.score_file
