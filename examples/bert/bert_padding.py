# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""
Functions for padding and unpadding 
"""

from typing import Tuple

import torch

class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor,
                first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim,
                             *values.shape[1:],
                             device=values.device,
                             dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        indices, = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply