from __future__ import annotations

from pathlib import Path

from expecttest import TestCase
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path

basic_kernels = import_path(Path(__file__).parent / "data/basic_kernels.py")


class TestClosures(TestCase):
    maxDiff = 16384

    def test_add_global(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, out = code_and_output(basic_kernels.use_globals, args)
        torch.testing.assert_close(
            out,
            torch.sin(args[0] + basic_kernels.global_tensor[None, :])
            + basic_kernels.global_float,
        )
        self.assertExpectedInline(
            code,
            """\
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

import helion._testing.basic_kernels as _source_module

@triton.jit
def _use_globals_kernel(a, _source_module_attr_global_tensor, out, a_size_0, a_size_1, _source_module_attr_global_tensor_stride_0, a_stride_0, a_stride_1, out_stride_0, out_stride_1, _source_module_attr_global_float, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < a_size_1
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(_source_module_attr_global_tensor + indices_1[None, :] * _source_module_attr_global_tensor_stride_0, mask_1[None, :], other=0)
    v_0 = load + load_1
    v_1 = tl_math.sin(v_0)
    v_2 = v_1 + _source_module_attr_global_float
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def use_globals(a):
    out = _source_module.empty_like(a)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _use_globals_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, _source_module.global_tensor, out, a.size(0), a.size(1), _source_module.global_tensor.stride(0), a.stride(0), a.stride(1), out.stride(0), out.stride(1), _source_module.global_float, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )
