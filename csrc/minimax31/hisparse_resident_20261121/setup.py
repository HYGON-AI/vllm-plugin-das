import os
from setuptools import setup

os.environ.setdefault("PYTORCH_NVCC", "aicc")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx938")
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="minimax31-hisparse-resident-ab",
    ext_modules=[CUDAExtension(
        name="m31_hisparse_resident",
        sources=["bindings.cpp", "planner.cu", "commit.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--offload-arch=gfx938"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
