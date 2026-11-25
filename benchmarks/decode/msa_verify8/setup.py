# SPDX-License-Identifier: Apache-2.0
"""Build the isolated Verify8 extension from the deployed vendor source."""
import os
from pathlib import Path
import shutil
import subprocess

from setuptools import setup

os.environ.setdefault("PYTORCH_NVCC", "aicc")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx938")

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).resolve().parent
vendor = Path(os.environ["MSA_VENDOR_SOURCE"]).resolve(strict=True) / "csrc"
sparse = vendor / "gfx93/prefill/sparse"
# Hipify writes beside its input. Keep the deployment source read-only.
stage1 = here / "build" / "vendor" / "msa128_stage1_decode_fp8_gfx938.cu"
stage1.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(sparse / stage1.name, stage1)
with (here / "stage1_variants.patch").open("rb") as patch:
    subprocess.run(["patch", "--batch", "--fuzz=0", str(stage1)], stdin=patch, check=True)

setup(
    name="hcu-minimax-msa-verify8",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="hcu_minimax_msa_verify8",
            sources=[str(here / "bindings.cpp"), str(here / "verify8_msa.cu"), str(stage1)],
            include_dirs=[
                str(path) for path in (
                    here, sparse, vendor, vendor / "api", vendor / "kerutils/include",
                    vendor / "gfx93", vendor / "cutlass/cutlass_3.2.1/include",
                    sparse / "dsa_mls/legacy/include",
                )
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3", "-std=c++20", "-DNDEBUG", "-DDCU_ASM",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "--offload-arch=gfx938",
                    "-mcode-object-version=5", "-mllvm=-support-768-vgprs=true",
                    "-mllvm=-disable-machine-sink", "-mllvm=-disable-code-sink",
                    "-mllvm=-amdgpu-enable-rewrite-partial-reg-uses=false",
                    "-mllvm=-allow-gvn-convergent-call=true",
                    "-mllvm=-disallow-uniform-vmed3-combine=true",
                    "-mllvm=-hcu-pre-emit-load-store-opt=false",
                    "-mllvm=-amdgpu-early-inline-all=true",
                    "-mllvm=-amdgpu-function-calls=false",
                    "-fno-finite-math-only", "--gpu-max-threads-per-block=256",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
