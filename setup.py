"""Setup script for vLLM HCU plugin."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from setuptools import find_packages, setup


# =========================================================
# 基础路径
# =========================================================
ROOT = Path(__file__).parent.resolve()
PWD = str(ROOT)

ADD_GIT_VERSION = os.environ.get("ADD_GIT_VERSION", "0") == "1"


# =========================================================
# Git 信息
# =========================================================
def get_git_sha(root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


# =========================================================
# 生成版本号
# =========================================================
def build_hcu_version(sha: Optional[str] = None) -> str:
    version = "das"

    # Git 版本
    if ADD_GIT_VERSION:
        if sha is None:
            sha = get_git_sha(ROOT)
        if sha != "unknown":
            version = f"das.{sha[:7]}"

    rocm_path = os.getenv("ROCM_PATH")
    if rocm_path:
        rocm_version_file = Path(rocm_path) / ".info" / "rocm_version"
        try:
            with open(rocm_version_file, encoding="utf-8") as f:
                rocm_version = f.readline().strip().replace(".", "")
                version += f".dtk{rocm_version}"
        except Exception:
            pass

    return version


# =========================================================
# 写入 version.py
# =========================================================
def write_version_file(version_suffix: str) -> None:
    version_file = ROOT / "vllm_hcu" / "version.py"

    content = f'''\
try:
    __version__ = "0.18.1"
    __version_tuple__ = (0, 18, 1)
    __hcu_version__ = "0.18.1+{version_suffix}"

    from vllm_hcu.version import __version__, __version_tuple__, __hcu_version__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash: {{e}}", RuntimeWarning)
    __version__ = "dev"
    __version_tuple__ = (0, 0, 0)
'''

    with open(version_file, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================
# 获取最终版本
# =========================================================
def get_version() -> str:
    # 确保 git safe directory（避免 CI 报错）
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", PWD],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    version_suffix = build_hcu_version()
    write_version_file(version_suffix)

    version_file = ROOT / "vllm_hcu" / "version.py"
    scope = {}
    with open(version_file, encoding="utf-8") as f:
        exec(compile(f.read(), str(version_file), "exec"), scope)

    return scope["__hcu_version__"]


# =========================================================
# setup
# =========================================================
setup(
    name="vllm_hcu",
    version=get_version(),
    description="vLLM HCU backend plugin",
    python_requires=">=3.10",
    packages=find_packages(),
    entry_points={
        "vllm.platform_plugins": [
            "hcu = vllm_hcu:hcu_platform_plugin",
        ],
        "vllm.general_plugins": [
            # "hcu_model = vllm_hcu:hcu_platform_register_model",
            "hcu_ops = vllm_hcu:hcu_platform_register_ops",
        ],
    },
)
