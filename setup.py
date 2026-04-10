"""Setup script for vLLM HCU plugin."""

import os
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from typing import Optional, Union
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from setuptools import find_packages, setup

if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

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
# --- 2. 定义 C++ 扩展模块 ---
# =========================================================
# 建议将生成的 .so 放到包名路径下，例如 vllm_hcu.cache_ops
ext_modules = [
    CUDAExtension(
        name='vllm_hcu.hcu_cache_ops', 
        sources=['vllm_hcu/csrc/hcu_cache_kernel.cu'], 
        define_macros=[('TORCH_EXTENSION_NAME', 'hcu_cache_ops')], 
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-D__HIP_PLATFORM_AMD__=1', '-fno-gpu-rdc']
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
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={"vllm_hcu": ["*.so", "so/*.so", "include/*.h"]},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExt
    },
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
