from __future__ import annotations

from pathlib import Path

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


@helion.kernel
def device_loop_3d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    for tile_a in hl.tile(a):
        for tile_b, tile_c, tile_d in hl.tile([b, c, d]):
            out[tile_a, tile_b, tile_c, tile_d] = torch.sin(
                x[tile_a, tile_b, tile_c, tile_d]
            )
    return out


class TestLoops(TestCase):
    def test_pointwise_device_loop(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.pointwise_device_loop,
            args,
            block_sizes=[32, 32],
        )
        torch.testing.assert_close(result, torch.sigmoid(args[0] + 1))
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _pointwise_device_loop_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, n, m, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < n
    for start_1 in range(0, m, BLOCK_SIZE_1):
        block_idx_1 = start_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
        mask_1 = block_idx_1 < m
        load = tl.load(x + (block_idx_0[:, None] * x_stride_0 + block_idx_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = 1.0
        v_1 = load + v_0
        v_2 = tl.sigmoid(v_1)
        tl.store(out + (block_idx_0[:, None] * out_stride_0 + block_idx_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def pointwise_device_loop(x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    BLOCK_SIZE_0 = 32
    BLOCK_SIZE_1 = 32
    _pointwise_device_loop_kernel[triton.cdiv(n, BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, m, BLOCK_SIZE_0, BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_3d_device_loop0(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[[1], [8, 8, 8]],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, d, c, b, BLOCK_SIZE_3: tl.constexpr, BLOCK_SIZE_2: tl.constexpr, BLOCK_SIZE_1: tl.constexpr):
    block_idx_0 = tl.program_id(0) + tl.zeros([1], tl.int32)
    for start_1 in range(0, b, BLOCK_SIZE_1):
        block_idx_1 = start_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
        mask_1 = block_idx_1 < b
        for start_2 in range(0, c, BLOCK_SIZE_2):
            block_idx_2 = start_2 + tl.arange(0, BLOCK_SIZE_2).to(tl.int32)
            mask_2 = block_idx_2 < c
            for start_3 in range(0, d, BLOCK_SIZE_3):
                block_idx_3 = start_3 + tl.arange(0, BLOCK_SIZE_3).to(tl.int32)
                mask_3 = block_idx_3 < d
                load = tl.load(x + (block_idx_0[:, None, None, None] * x_stride_0 + block_idx_1[None, :, None, None] * x_stride_1 + block_idx_2[None, None, :, None] * x_stride_2 + block_idx_3[None, None, None, :] * x_stride_3), mask_1[None, :, None, None] & mask_2[None, None, :, None] & mask_3[None, None, None, :], other=0)
                v_0 = tl_math.sin(load)
                tl.store(out + (block_idx_0[:, None, None, None] * out_stride_0 + block_idx_1[None, :, None, None] * out_stride_1 + block_idx_2[None, None, :, None] * out_stride_2 + block_idx_3[None, None, None, :] * out_stride_3), v_0, mask_1[None, :, None, None] & mask_2[None, None, :, None] & mask_3[None, None, None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    BLOCK_SIZE_3 = 8
    BLOCK_SIZE_2 = 8
    BLOCK_SIZE_1 = 8
    _device_loop_3d_kernel[a,](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), d, c, b, BLOCK_SIZE_3, BLOCK_SIZE_2, BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_3d_device_loop1(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[[2], [8, 4, 1]],
            loop_order=[1, 0, 2],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, d, b, c, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr, BLOCK_SIZE_2: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a
    for start_2 in range(0, c, BLOCK_SIZE_2):
        block_idx_2 = start_2 + tl.arange(0, BLOCK_SIZE_2).to(tl.int32)
        mask_2 = block_idx_2 < c
        for start_1 in range(0, b, BLOCK_SIZE_1):
            block_idx_1 = start_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
            mask_1 = block_idx_1 < b
            for start_3 in range(0, d, 1):
                block_idx_3 = start_3 + tl.arange(0, 1).to(tl.int32)
                load = tl.load(x + (block_idx_0[:, None, None, None] * x_stride_0 + block_idx_1[None, :, None, None] * x_stride_1 + block_idx_2[None, None, :, None] * x_stride_2 + block_idx_3[None, None, None, :] * x_stride_3), mask_0[:, None, None, None] & mask_1[None, :, None, None] & mask_2[None, None, :, None], other=0)
                v_0 = tl_math.sin(load)
                tl.store(out + (block_idx_0[:, None, None, None] * out_stride_0 + block_idx_1[None, :, None, None] * out_stride_1 + block_idx_2[None, None, :, None] * out_stride_2 + block_idx_3[None, None, None, :] * out_stride_3), v_0, mask_0[:, None, None, None] & mask_1[None, :, None, None] & mask_2[None, None, :, None])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    BLOCK_SIZE_0 = 2
    BLOCK_SIZE_1 = 8
    BLOCK_SIZE_2 = 4
    _device_loop_3d_kernel[triton.cdiv(a, BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, d, b, c, BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_3d_device_loop2(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[4, 128],
            loop_order=[2, 0, 1],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, c, b, d, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1_2_3: tl.constexpr):
    block_idx_0 = tl.program_id(0) * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < a
    for lid_1_2_3 in range(tl.cdiv(b * c * d, BLOCK_SIZE_1_2_3)):
        offsets_1_2_3 = lid_1_2_3 * BLOCK_SIZE_1_2_3 + tl.arange(0, BLOCK_SIZE_1_2_3).to(tl.int32)
        block_idx_2 = offsets_1_2_3 % c
        block_idx_1 = offsets_1_2_3 // c % b
        block_idx_3 = offsets_1_2_3 // (b * c)
        mask_1_2_3 = offsets_1_2_3 < b * c * d
        load = tl.load(x + (block_idx_0[:, None] * x_stride_0 + block_idx_1[None, :] * x_stride_1 + block_idx_2[None, :] * x_stride_2 + block_idx_3[None, :] * x_stride_3), mask_0[:, None] & mask_1_2_3[None, :], other=0)
        v_0 = tl_math.sin(load)
        tl.store(out + (block_idx_0[:, None] * out_stride_0 + block_idx_1[None, :] * out_stride_1 + block_idx_2[None, :] * out_stride_2 + block_idx_3[None, :] * out_stride_3), v_0, mask_0[:, None] & mask_1_2_3[None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    BLOCK_SIZE_0 = 4
    BLOCK_SIZE_1_2_3 = 128
    _device_loop_3d_kernel[triton.cdiv(a, BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, c, b, d, BLOCK_SIZE_0, BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)
    return out""",
        )
