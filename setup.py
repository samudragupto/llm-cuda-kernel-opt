from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_compile_args = {
    'cxx': ['/O2', '/std:c++17', '/Zc:preprocessor', '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'] if os.name == 'nt' else ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '-arch=sm_89',
        '-lineinfo',
        '--use_fast_math',
        '-Xcompiler', '/Zc:preprocessor',
        '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__'
    ]
}

setup(
    name='fused_ops',
    ext_modules=[
        CUDAExtension(
            name='fused_ops',
            sources=['fused_ops.cu'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
