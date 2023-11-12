// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>
#include "monarch_cuda/monarch_fwd.h"
#include "monarch_cuda/monarch_fwd_complex.h"
#include "monarch_cuda/monarch_fwd_r2r.h"
#include "monarch_cuda/monarch_bwd.h"
#include "monarch_cuda/monarch_bwd_complex.h"
#include "monarch_cuda/monarch_bwd_r2r.h"
#include "butterfly/butterfly.h"
#include "conv1d/conv1d.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("monarch_conv_forward", &monarch_conv, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_16_16_16", &monarch_conv_16_16_16, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_32_16_16", &monarch_conv_32_16_16, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_16_32_32", &monarch_conv_16_32_32, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_32_32_32", &monarch_conv_32_32_32, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_16_16_16_complex", &monarch_conv_16_16_16_complex, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_32_16_16_complex", &monarch_conv_32_16_16_complex, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_16_32_32_complex", &monarch_conv_16_32_32_complex, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_32_32_32_complex", &monarch_conv_32_32_32_complex, "Monarch forward (CUDA)");
    m.def("monarch_conv_forward_32_32_32_complex_truncated", &monarch_conv_32_32_32_complex_truncated, "Monarch forward (CUDA)");

    m.def("monarch_conv_backward", &monarch_conv_bwd, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_16_16_16", &monarch_conv_bwd_16_16_16, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_32_16_16", &monarch_conv_bwd_32_16_16, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_16_32_32", &monarch_conv_bwd_16_32_32, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_32_32_32", &monarch_conv_bwd_32_32_32, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_16_16_16_complex", &monarch_conv_bwd_16_16_16_complex, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_32_16_16_complex", &monarch_conv_bwd_32_16_16_complex, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_16_32_32_complex", &monarch_conv_bwd_16_32_32_complex, "Monarch backward (CUDA)");
    m.def("monarch_conv_backward_32_32_32_complex", &monarch_conv_bwd_32_32_32_complex, "Monarch backward (CUDA)");

    m.def("monarch_conv_forward_r2r", &monarch_conv_r2r, "Monarch forward (CUDA)");
    m.def("monarch_conv_backward_r2r", &monarch_conv_bwd_r2r, "Monarch backward (CUDA)");

    // butterfly kernels
    m.def("butterfly_forward", &butterfly, "Butterfly forward (CUDA)");
    m.def("butterfly_gated_forward", &butterfly_gated, "Butterfly gated forward (CUDA)");
    m.def("butterfly_bf16_forward", &butterfly_bf16, "Butterfly forward bf16 (CUDA)");
    m.def("butterfly_gated_bf16_forward", &butterfly_gated_bf16, "Butterfly gated forward bf16 (CUDA)");
    m.def("butterfly_padded_forward", &butterfly_padded, "Butterfly padded (CUDA)");
    m.def("butterfly_padded_bf16_forward", &butterfly_padded_bf16, "Butterfly padded (CUDA)");
    m.def("butterfly_padded_gated_forward", &butterfly_padded_gated, "Butterfly padded (CUDA)");
    m.def("butterfly_padded_gated_bf16_forward", &butterfly_padded_gated_bf16, "Butterfly padded (CUDA)");
    m.def("butterfly_ifft_forward", &butterfly_ifft, "Butterfly ifft forard (CUDA)");
    m.def("butterfly_ifft_gated_forward", &butterfly_ifft_gated, "Butterfly ifft gated forard (CUDA)");
    m.def("butterfly_ifft_gated_bf16_forward", &butterfly_ifft_gated_bf16, "Butterfly ifft gated bf16 forard (CUDA)");
    m.def("butterfly_ifft_bf16_forward", &butterfly_ifft_bf16, "Butterfly ifft forward bf16 (CUDA)");
    m.def("butterfly_ifft_padded_forward", &butterfly_ifft_padded, "Butterfly ifft forward padded (CUDA)");
    m.def("butterfly_ifft_padded_gated_forward", &butterfly_ifft_padded_gated, "Butterfly ifft forward padded (CUDA)");
    m.def("butterfly_ifft_padded_bf16_forward", &butterfly_ifft_padded_bf16, "Butterfly ifft forward padded (CUDA)");
    m.def("butterfly_ifft_padded_gated_bf16_forward", &butterfly_ifft_padded_gated_bf16, "Butterfly ifft forward padded (CUDA)");

    m.def("conv1d_forward", &conv1d_fwd, "conv1d forward (CUDA)");
    m.def("conv1d_backward", &conv1d_bwd, "conv1d backward (CUDA)");
    
}