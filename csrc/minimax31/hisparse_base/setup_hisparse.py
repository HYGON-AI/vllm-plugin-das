import os
from pathlib import Path
os.environ.setdefault('PYTORCH_NVCC','aicc');os.environ.setdefault('PYTORCH_ROCM_ARCH','gfx938')
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
r=Path(__file__).parent.resolve()
setup(name='hcu-minimax-hisparse-experiment',version='0.0.1',
 ext_modules=[CUDAExtension('hcu_minimax_hisparse_native',
 sources=[str(r/'bindings.cpp'),str(r/'ops.cu')],include_dirs=[str(r)],
 extra_compile_args={'cxx':['-O3','-std=c++20'],'nvcc':['-O3','-std=c++20','--offload-arch=gfx938','-mcode-object-version=5']})],
 cmdclass={'build_ext':BuildExtension})
