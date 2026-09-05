# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Setup script for vLLM HCU plugin."""

import os
import shutil
import subprocess
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch.utils.cpp_extension as torch_cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from setuptools import find_namespace_packages, setup

if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

# =========================================================
# 基础路径
# =========================================================
ROOT = Path(__file__).parent.resolve()
BASE_VERSION = "0.28.0"

ADD_GIT_VERSION = os.environ.get("ADD_GIT_VERSION", "1") == "1"

# PyTorch injects required HIP platform defines into every HCU compiler
# command. Keep the compiler contract intact while preventing those private
# spellings from being copied into package build logs.
_HIP_PLATFORM_DEFINES = tuple(
    flag
    for flag in torch_cpp_extension.COMMON_HIP_FLAGS
    if flag.startswith("-D__HIP_PLATFORM_")
)
_REDACTED_HIP_PLATFORM_DEFINE = "<hcu-platform-define>"


def _sanitize_hcu_build_output(output: str) -> str:
    for platform_define in _HIP_PLATFORM_DEFINES:
        output = output.replace(
            platform_define,
            _REDACTED_HIP_PLATFORM_DEFINE,
        )
    return output


@contextmanager
def _sanitized_ninja_build():
    original_run_ninja_build = torch_cpp_extension._run_ninja_build

    def run_ninja_build(build_directory, _verbose, error_prefix):
        try:
            # PyTorch normally invokes Ninja verbosely and prints complete
            # compiler commands. Quiet successful builds; retain sanitized
            # compiler diagnostics when a build fails.
            original_run_ninja_build(
                build_directory,
                False,
                error_prefix,
            )
        except RuntimeError as error:
            raise RuntimeError(
                _sanitize_hcu_build_output(str(error))
            ) from None

    torch_cpp_extension._run_ninja_build = run_ninja_build
    try:
        yield
    finally:
        torch_cpp_extension._run_ninja_build = original_run_ninja_build


# =========================================================
# Git 信息
# =========================================================
def get_git_sha(root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={Path(root).resolve()}",
                    "rev-parse",
                    "HEAD",
                ],
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
# 获取最终版本
# =========================================================
def get_version() -> str:
    version_suffix = build_hcu_version()
    return f"{BASE_VERSION}+{version_suffix}"

# =========================================================
# --- 2. 定义 C++ 扩展模块 ---
# =========================================================
# 建议将生成的 .so 放到包名路径下，例如 vllm_hcu.cache_ops
ext_modules = [
    CUDAExtension(
        name='vllm_hcu.hcu_ops', 
        sources=['vllm_hcu/csrc/hcu_cache_kernel.cu',
                 'vllm_hcu/csrc/torch_bindings.cpp',
                 'vllm_hcu/csrc/custom_all_reduce.cu',
                 'vllm_hcu/csrc/fused_deepseek_v4_inv_rope_kernel.cu'
                 ], 
        define_macros=[('TORCH_EXTENSION_NAME', 'hcu_ops')], 
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-fno-gpu-rdc']
        }
    )
]

# =========================================================
# --- 3. 自定义并行编译并拷贝的类 ---
# =========================================================
# 这里使用 .with_options(use_ninja=True) 来开启多进程编译支持
class CustomBuildExt(BuildExtension.with_options(use_ninja=True)):
    def run(self):
        # 1. 调用父类的 run()，这会根据 MAX_JOBS 环境并行编译
        with _sanitized_ninja_build():
            super().run()
        
        # 2. 编译完成后，执行你的拷贝逻辑
        for ext in self.extensions:
            # 获取编译出来的 .so 完整路径
            ext_path = self.get_ext_fullpath(ext.name)
            if not os.path.exists(ext_path):
                continue
                
            file_name = os.path.basename(ext_path)
            # 确定目标包路径 vllm_hcu/
            target_dir = os.path.join(ROOT, "vllm_hcu")
            target_path = os.path.join(target_dir, file_name)
            if os.path.abspath(ext_path) == os.path.abspath(target_path):
                continue

            # 如果目标已存在则先删除
            if os.path.exists(target_path):
                os.remove(target_path)
            
            # 拷贝文件
            shutil.copyfile(ext_path, target_path)
            print(f"\n[CustomBuildExt] Success: {file_name} -> {target_path}")

# =========================================================
# setup
# =========================================================
setup(
    name="vllm_hcu",
    version=get_version(),
    description="vLLM HCU backend plugin",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_namespace_packages(include=["vllm_hcu", "vllm_hcu.*"]),
    package_data={"vllm_hcu": ["*.so", "so/*.so", "include/*.h"]},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    entry_points={
        "console_scripts": [
            "vllm-hcu-apply-patches = vllm_hcu.post_install:main",
            "vllm-hcu-doctor = vllm_hcu.doctor:main",
        ],
        "vllm.platform_plugins": [
            "hcu = vllm_hcu:hcu_platform_plugin",
        ],
        "vllm.general_plugins": [
            "hcu_model = vllm_hcu:hcu_platform_register_model",
            "hcu_ops = vllm_hcu:hcu_platform_register_ops",
        ],
    },
)
