"""
collector.py
负责: git clone 目标仓库, 扫描关键文件, 打包成一份"上下文文本"
供后续丢给 Claude API 分析使用。
"""
import os
import subprocess
import tempfile
import shutil

# 优先级高的文件名(大小写不敏感匹配)
PRIORITY_FILES = [
    "readme.md", "readme", "readme.rst", "readme.txt",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "cargo.toml", "go.mod", "gemfile", "pom.xml", "build.gradle",
    "contributing.md", "usage.md", "getting_started.md", "quickstart.md",
    "docs/readme.md",
]

# 值得整体收录的目录(示例/文档)
INTEREST_DIRS = ["examples", "example", "docs", "sample", "samples"]

MAX_FILE_CHARS = 6000       # 单文件最大截取长度
MAX_TOTAL_CHARS = 60000     # 总打包内容上限,避免超出上下文
IGNORE_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv"}


def clone_repo(repo_url: str) -> str:
    """浅克隆仓库到临时目录,返回本地路径"""
    tmp_dir = tempfile.mkdtemp(prefix="repo_cheatsheet_")
    print(f"Cloning {repo_url} -> {tmp_dir}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, tmp_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {result.stderr}")
    return tmp_dir


def get_dir_tree(root: str, max_depth: int = 2) -> str:
    """生成目录结构(限制深度), 帮助 LLM 了解项目布局"""
    lines = []
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        name = os.path.basename(dirpath) or dirpath
        lines.append(f"{indent}{name}/")
        if depth == max_depth:
            continue
        for f in sorted(filenames):
            if not f.startswith("."):
                lines.append(f"{indent}  {f}")
    return "\n".join(lines[:300])  # 防止超大仓库刷屏


def read_file_safe(path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars)
        return content
    except Exception as e:
        return f"[无法读取: {e}]"


def collect_priority_files(root: str) -> dict:
    """在仓库根目录及一层子目录中查找优先文件"""
    found = {}
    lower_map = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth > 1:
            dirnames[:] = []
            continue
        for f in filenames:
            rel_path = os.path.normpath(os.path.join(rel_dir, f)) if rel_dir != "." else f
            lower_map[rel_path.lower()] = rel_path

    for target in PRIORITY_FILES:
        target_norm = target.lower()
        if target_norm in lower_map:
            rel_path = lower_map[target_norm]
            found[rel_path] = read_file_safe(os.path.join(root, rel_path))
    return found


def collect_interest_dirs(root: str) -> dict:
    """收录 examples/docs 等目录下的少量代表性文件"""
    collected = {}
    total_chars = 0
    for d in INTEREST_DIRS:
        dir_path = os.path.join(root, d)
        if not os.path.isdir(dir_path):
            continue
        count = 0
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirnames[:] = [x for x in dirnames if x not in IGNORE_DIRS and not x.startswith(".")]
            for f in sorted(filenames):
                if count >= 5:  # 每个目录最多取 5 个文件示例
                    break
                if total_chars > MAX_TOTAL_CHARS:
                    return collected
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root)
                content = read_file_safe(full_path, max_chars=2000)
                collected[rel_path] = content
                total_chars += len(content)
                count += 1
    return collected


def build_context_bundle(repo_url: str) -> str:
    """主函数: 克隆 + 扫描 + 打包成一份文本"""
    local_path = clone_repo(repo_url)
    try:
        tree = get_dir_tree(local_path)
        priority_files = collect_priority_files(local_path)
        example_files = collect_interest_dirs(local_path)

        parts = [f"# Repository: {repo_url}\n", "## 目录结构 (部分)\n```\n" + tree + "\n```\n"]

        for path, content in priority_files.items():
            parts.append(f"## 文件: {path}\n```\n{content}\n```\n")

        for path, content in example_files.items():
            parts.append(f"## 示例文件: {path}\n```\n{content}\n```\n")

        bundle = "\n".join(parts)
        if len(bundle) > MAX_TOTAL_CHARS:
            bundle = bundle[:MAX_TOTAL_CHARS] + "\n\n[内容过长,已截断]"
        return bundle
    finally:
        shutil.rmtree(local_path, ignore_errors=True)
