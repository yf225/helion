from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel
def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    for tile0, tile1 in hl.tile(out0.size()):
        out0[tile0, tile1] = a[tile0, tile1] + b[tile0, None]
        out1[tile0, tile1] = a[tile0, tile1] + b[None, tile1]
    return out0, out1


def broadcast_fn_ref(a, b):
    out0 = a + b[:, None]
    out1 = a + b[None, :]
    return out0, out1


def _check_broadcast_fn(**config):
    args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
    code, (out0, out1) = code_and_output(broadcast_fn, args, **config)
    ref0, ref1 = broadcast_fn_ref(*args)
    torch.testing.assert_close(out0, ref0)
    torch.testing.assert_close(out1, ref1)
    return code


class TestBroadcasting(TestCase):
    def test_broadcast_no_flatten(self):
        args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
        assert (
            not broadcast_fn.bind(args).config_spec.block_size_specs[0].allow_flattened
        )

    def test_broadcast1(self):
        code = _check_broadcast_fn(
            block_size=[16, 8],
        )
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a_size_0
    block_idx_1 = tl.program_id(1) * BLOCK_SIZE_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
    mask_1 = block_idx_1 < a_size_1
    load = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(b + block_idx_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (block_idx_0[:, None] * out0_stride_0 + block_idx_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(b + block_idx_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (block_idx_0[:, None] * out1_stride_0 + block_idx_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    BLOCK_SIZE_0 = 16
    BLOCK_SIZE_1 = 8
    _broadcast_fn_kernel[triton.cdiv(a.size(0), BLOCK_SIZE_0), triton.cdiv(a.size(1), BLOCK_SIZE_1)](a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), BLOCK_SIZE_0, BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1)""",
        )

    def test_broadcast2(self):
        code = _check_broadcast_fn(block_size=[16, 8], loop_order=(1, 0))
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, BLOCK_SIZE_1: tl.constexpr, BLOCK_SIZE_0: tl.constexpr):
    block_idx_1 = tl.program_id(0) * BLOCK_SIZE_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
    mask_1 = block_idx_1 < a_size_1
    block_idx_0 = tl.program_id(1) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a_size_0
    load = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(b + block_idx_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (block_idx_0[:, None] * out0_stride_0 + block_idx_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(b + block_idx_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (block_idx_0[:, None] * out1_stride_0 + block_idx_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    BLOCK_SIZE_1 = 8
    BLOCK_SIZE_0 = 16
    _broadcast_fn_kernel[triton.cdiv(a.size(1), BLOCK_SIZE_1), triton.cdiv(a.size(0), BLOCK_SIZE_0)](a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), BLOCK_SIZE_1, BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return (out0, out1)""",
        )

    def test_broadcast3(self):
        code = _check_broadcast_fn(
            block_size=[64, 1],
        )
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, BLOCK_SIZE_0: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a_size_0
    block_idx_1 = tl.program_id(1) + tl.zeros([1], tl.int32)
    load = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None], other=0)
    load_1 = tl.load(b + block_idx_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (block_idx_0[:, None] * out0_stride_0 + block_idx_1[None, :] * out0_stride_1), v_0, mask_0[:, None])
    load_2 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None], other=0)
    load_3 = tl.load(b + block_idx_1[None, :] * b_stride_0, None)
    v_1 = load_2 + load_3
    tl.store(out1 + (block_idx_0[:, None] * out1_stride_0 + block_idx_1[None, :] * out1_stride_1), v_1, mask_0[:, None])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    BLOCK_SIZE_0 = 64
    _broadcast_fn_kernel[triton.cdiv(a.size(0), BLOCK_SIZE_0), a.size(1)](a, b, out0, out1, a.size(0), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return (out0, out1)""",
        )

    def test_broadcast4(self):
        code = _check_broadcast_fn(
            block_size=[1, 64],
        )
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, BLOCK_SIZE_1: tl.constexpr):
    block_idx_0 = tl.program_id(0) + tl.zeros([1], tl.int32)
    block_idx_1 = tl.program_id(1) * BLOCK_SIZE_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
    mask_1 = block_idx_1 < a_size_1
    load = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_1[None, :], other=0)
    load_1 = tl.load(b + block_idx_0[:, None] * b_stride_0, None)
    v_0 = load + load_1
    tl.store(out0 + (block_idx_0[:, None] * out0_stride_0 + block_idx_1[None, :] * out0_stride_1), v_0, mask_1[None, :])
    load_2 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_1[None, :], other=0)
    load_3 = tl.load(b + block_idx_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (block_idx_0[:, None] * out1_stride_0 + block_idx_1[None, :] * out1_stride_1), v_1, mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    BLOCK_SIZE_1 = 64
    _broadcast_fn_kernel[a.size(0), triton.cdiv(a.size(1), BLOCK_SIZE_1)](a, b, out0, out1, a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1)""",
        )

    def test_constexpr_index(self):
        @helion.kernel
        def fn(a, idx1):
            out0 = torch.empty_like(a)
            out1 = torch.empty_like(a)
            out2 = torch.empty_like(a)
            idx0 = 11
            for tile0, tile1 in hl.tile(out0.size()):
                out0[tile0, tile1] = a[tile0, tile1] + a[tile0, 3, None]
                out1[tile0, tile1] = a[tile0, tile1] + a[idx0, tile1][None, :]
                out2[tile0, tile1] = a[tile0, tile1] + a[tile0, idx1, None]
            return out0, out1, out2

        args = (torch.randn(512, 512, device=DEVICE), 123)
        code, (out0, out1, out2) = code_and_output(fn, args, block_size=[16, 16])
        torch.testing.assert_close(out0, args[0] + args[0][:, 3, None])
        torch.testing.assert_close(out1, args[0] + args[0][11, None, :])
        torch.testing.assert_close(out2, args[0] + args[0][:, args[1], None])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(a, out0, out1, out2, a_size_0, a_size_1, a_stride_0, a_stride_1, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, out2_stride_0, out2_stride_1, idx1, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a_size_0
    block_idx_1 = tl.program_id(1) * BLOCK_SIZE_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
    mask_1 = block_idx_1 < a_size_1
    load = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + tl.full([1], 3, tl.int32)[None, :] * a_stride_1), mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (block_idx_0[:, None] * out0_stride_0 + block_idx_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(a + (tl.full([1], 11, tl.int32) * a_stride_0 + block_idx_1 * a_stride_1), mask_1, other=0)
    subscript = load_3[None, :]
    v_1 = load_2 + subscript
    tl.store(out1 + (block_idx_0[:, None] * out1_stride_0 + block_idx_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])
    load_4 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + block_idx_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_5 = tl.load(a + (block_idx_0[:, None] * a_stride_0 + tl.full([1], idx1, tl.int32)[None, :] * a_stride_1), mask_0[:, None], other=0)
    v_2 = load_4 + load_5
    tl.store(out2 + (block_idx_0[:, None] * out2_stride_0 + block_idx_1[None, :] * out2_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def fn(a, idx1):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(a)
    idx0 = 11
    BLOCK_SIZE_0 = 16
    BLOCK_SIZE_1 = 16
    _fn_kernel[triton.cdiv(a.size(0), BLOCK_SIZE_0), triton.cdiv(a.size(1), BLOCK_SIZE_1)](a, out0, out1, out2, a.size(0), a.size(1), a.stride(0), a.stride(1), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), out2.stride(0), out2.stride(1), idx1, BLOCK_SIZE_0, BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1, out2)""",
        )


if __name__ == "__main__":
    unittest.main()
