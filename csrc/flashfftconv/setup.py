from __future__ import annotations

import os
import subprocess

import torch

from functools import cache
from pathlib import Path
from typing import Tuple, List

from packaging.version import parse, Version
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


CUDA_PATH: Path = Path(CUDA_HOME)
TORCH_VERSION: Version = Version(torch.__version__)

TORCH_MAJOR: int = TORCH_VERSION.major
TORCH_MINOR: int = TORCH_VERSION.minor

EXTENSION_NAME: str = 'monarch_cuda'


@cache
def get_cuda_bare_metal_version(cuda_dir: Path) -> Tuple[str, Version]:

    raw = (
        subprocess.run(
            [str(cuda_dir / 'bin' / 'nvcc'), '-V'],
            capture_output=True,
            check=True,
            encoding='utf-8',
        )
        .stdout
    )

    output = raw.split()
    version, _, _ = output[output.index('release') + 1].partition(',')

    return raw, parse(version)


def raise_if_cuda_home_none(global_option: str) -> None:

    if CUDA_HOME is None:

        raise RuntimeError(
            f"{global_option} was requested, but nvcc was not found. Are you sure your "
            "environment has nvcc available? If you're installing within a container from "
            "https://hub.docker.com/r/pytorch/pytorch, only images whose names contain "
            "'devel' will provide nvcc."
        )


def append_nvcc_threads(nvcc_extra_args: List[str]) -> List[str]:

    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_PATH)

    if bare_metal_version >= Version("11.2"):

        nvcc_extra_args.extend(("--threads", "4"))

    return nvcc_extra_args


def build_compiler_flags() -> List[str]:

    flags = ["-gencode", "arch=compute_80,code=sm_80"]

    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_PATH)

    if bare_metal_version < Version("11.0"):

        raise RuntimeError(f"{EXTENSION_NAME} is only supported on CUDA 11 and above")

    elif bare_metal_version >= Version("11.8"):

        flags.extend(
            [
                "-gencode", "arch=compute_89,code=sm_89",
                "-gencode", "arch=compute_90,code=sm_90",
            ]
        )

    return flags


if not torch.cuda.is_available():

    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )

    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:

        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_PATH)

        if bare_metal_version >= Version("11.8"):

            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"

        elif bare_metal_version >= Version("11.1"):

            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"

        elif bare_metal_version == Version("11.0"):

            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

        else:

            raise RuntimeError(f"{EXTENSION_NAME} is only supported on CUDA 11 and above")


# Log PyTorch Version.
print(f"\n\ntorch.__version__ = {TORCH_VERSION}\n\n")


# Verify that CUDA_HOME exists.
raise_if_cuda_home_none(EXTENSION_NAME)


setup(
    name='monarch_cuda',
    ext_modules=[
        CUDAExtension(
            EXTENSION_NAME,
            [
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
                'conv1d/conv1d_bhl.cu',
                'conv1d/conv1d_blh.cu',
                'conv1d/conv1d_bwd_cuda_bhl.cu',
                'conv1d/conv1d_bwd_cuda_blh.cu',
            ],
            extra_compile_args=(
                {
                    'cxx': ['-O3'],
                    'nvcc': append_nvcc_threads(
                        ['-O3', '-lineinfo', '--use_fast_math', '-std=c++17']
                        + build_compiler_flags()
                    ),
                }
            ),
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    version='0.0.0',
    description='Fast FFT algorithms for convolutions',
    url='https://github.com/HazyResearch/flash-fft-conv',
    author='Dan Fu, Hermann Kumbong',
    author_email='danfu@cs.stanford.edu',
    license='Apache 2.0'
)
