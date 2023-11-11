import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import subprocess

def get_last_arch_torch():
    arch = torch.cuda.get_arch_list()[-1]
    print(f"Found arch: {arch} from existing torch installation")
    return arch

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

arch = get_last_arch_torch()
# [MP] make install more flexible here
sm_num = arch[-2:]
cc_flag = ['--generate-code=arch=compute_80,code=compute_80']


setup(
    name='monarch_cuda',
    ext_modules=[
        CUDAExtension('monarch_cuda', [
            'monarch.cpp',
            'monarch_cuda/monarch_cuda_interface_fwd.cu',
            'monarch_cuda/monarch_cuda_interface_fwd_complex.cu',
            'monarch_cuda/monarch_cuda_interface_fwd_bf16.cu',
            'monarch_cuda/monarch_cuda_interface_fwd_bf16_complex.cu',
            'monarch_cuda/monarch_cuda_interface_fwd_r2r.cu',
            'monarch_cuda/monarch_cuda_interface_fwd_r2r_bf16.cu',
            'monarch_cuda/monarch_cuda_interface_bwd.cu',
            'monarch_cuda/monarch_cuda_interface_bwd_complex.cu',
            'monarch_cuda/monarch_cuda_interface_bwd_bf16.cu',
            'monarch_cuda/monarch_cuda_interface_bwd_bf16_complex.cu',
            'monarch_cuda/monarch_cuda_interface_bwd_r2r.cu',
            'monarch_cuda/monarch_cuda_interface_bwd_r2r_bf16.cu',
            'butterfly/butterfly_cuda.cu',
            'butterfly/butterfly_padded_cuda.cu',
            'butterfly/butterfly_padded_cuda_bf16.cu',
            'butterfly/butterfly_ifft_cuda.cu',
            'butterfly/butterfly_cuda_bf16.cu',
            'butterfly/butterfly_ifft_cuda_bf16.cu',
            'butterfly/butterfly_padded_ifft_cuda.cu',
            'butterfly/butterfly_padded_ifft_cuda_bf16.cu',
            'conv1d/conv1d_bhl_bf16.cu',
            'conv1d/conv1d_bhl_half.cu',
            'conv1d/conv1d_blh_bf16.cu',
            'conv1d/conv1d_blh_half.cu'
        ],
        extra_compile_args={'cxx': ['-O3'],
                             'nvcc': append_nvcc_threads(['-O3', '-lineinfo', '--use_fast_math', '-std=c++17'] + cc_flag)
                            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='0.0.0',
    description='Fast FFT algorithms for convolutions',
    url='https://github.com/HazyResearch/flash-fft-conv',
    author='Dan Fu, Hermann Kumbong',
    author_email='danfu@cs.stanford.edu',
    license='Apache 2.0')