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
def _pointwise_device_loop_kernel(_x, _out, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _n, _m, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _n
    for _start_1 in range(0, _m, _BLOCK_SIZE_1):
        _block_idx_1 = _start_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        _mask_1 = _block_idx_1 < _m
        _load = tl.load(_x + (_block_idx_0[:, None] * _x_stride_0 + _block_idx_1[None, :] * _x_stride_1), _mask_0[:, None] & _mask_1[None, :], other=0)
        _v_0 = 1.0
        _v_1 = _load + _v_0
        _v_2 = tl.sigmoid(_v_1)
        tl.store(_out + (_block_idx_0[:, None] * _out_stride_0 + _block_idx_1[None, :] * _out_stride_1), _v_2, _mask_0[:, None] & _mask_1[None, :])

def pointwise_device_loop(x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _pointwise_device_loop_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, m, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
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
def _device_loop_3d_kernel(_x, _out, _out_stride_0, _out_stride_1, _out_stride_2, _out_stride_3, _x_stride_0, _x_stride_1, _x_stride_2, _x_stride_3, _d, _c, _b, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    _block_idx_0 = tl.program_id(0) + tl.zeros([1], tl.int32)
    for _start_1 in range(0, _b, _BLOCK_SIZE_1):
        _block_idx_1 = _start_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        _mask_1 = _block_idx_1 < _b
        for _start_2 in range(0, _c, _BLOCK_SIZE_2):
            _block_idx_2 = _start_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            _mask_2 = _block_idx_2 < _c
            for _start_3 in range(0, _d, _BLOCK_SIZE_3):
                _block_idx_3 = _start_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                _mask_3 = _block_idx_3 < _d
                _load = tl.load(_x + (_block_idx_0[:, None, None, None] * _x_stride_0 + _block_idx_1[None, :, None, None] * _x_stride_1 + _block_idx_2[None, None, :, None] * _x_stride_2 + _block_idx_3[None, None, None, :] * _x_stride_3), _mask_1[None, :, None, None] & _mask_2[None, None, :, None] & _mask_3[None, None, None, :], other=0)
                _v_0 = tl_math.sin(_load)
                tl.store(_out + (_block_idx_0[:, None, None, None] * _out_stride_0 + _block_idx_1[None, :, None, None] * _out_stride_1 + _block_idx_2[None, None, :, None] * _out_stride_2 + _block_idx_3[None, None, None, :] * _out_stride_3), _v_0, _mask_1[None, :, None, None] & _mask_2[None, None, :, None] & _mask_3[None, None, None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_3 = 8
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_1 = 8
    _device_loop_3d_kernel[a,](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), d, c, b, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
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
def _device_loop_3d_kernel(_x, _out, _out_stride_0, _out_stride_1, _out_stride_2, _out_stride_3, _x_stride_0, _x_stride_1, _x_stride_2, _x_stride_3, _a, _d, _b, _c, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _a
    for _start_2 in range(0, _c, _BLOCK_SIZE_2):
        _block_idx_2 = _start_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        _mask_2 = _block_idx_2 < _c
        for _start_1 in range(0, _b, _BLOCK_SIZE_1):
            _block_idx_1 = _start_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
            _mask_1 = _block_idx_1 < _b
            for _start_3 in range(0, _d, 1):
                _block_idx_3 = _start_3 + tl.arange(0, 1).to(tl.int32)
                _load = tl.load(_x + (_block_idx_0[:, None, None, None] * _x_stride_0 + _block_idx_1[None, :, None, None] * _x_stride_1 + _block_idx_2[None, None, :, None] * _x_stride_2 + _block_idx_3[None, None, None, :] * _x_stride_3), _mask_0[:, None, None, None] & _mask_1[None, :, None, None] & _mask_2[None, None, :, None], other=0)
                _v_0 = tl_math.sin(_load)
                tl.store(_out + (_block_idx_0[:, None, None, None] * _out_stride_0 + _block_idx_1[None, :, None, None] * _out_stride_1 + _block_idx_2[None, None, :, None] * _out_stride_2 + _block_idx_3[None, None, None, :] * _out_stride_3), _v_0, _mask_0[:, None, None, None] & _mask_1[None, :, None, None] & _mask_2[None, None, :, None])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 4
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, d, b, c, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
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
def _device_loop_3d_kernel(_x, _out, _out_stride_0, _out_stride_1, _out_stride_2, _out_stride_3, _x_stride_0, _x_stride_1, _x_stride_2, _x_stride_3, _a, _c, _b, _d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1_2_3: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _a
    for _lid_1_2_3 in range(tl.cdiv(_b * _c * _d, _BLOCK_SIZE_1_2_3)):
        _offsets_1_2_3 = _lid_1_2_3 * _BLOCK_SIZE_1_2_3 + tl.arange(0, _BLOCK_SIZE_1_2_3).to(tl.int32)
        _block_idx_2 = _offsets_1_2_3 % _c
        _block_idx_1 = _offsets_1_2_3 // _c % _b
        _block_idx_3 = _offsets_1_2_3 // (_b * _c)
        _mask_1_2_3 = _offsets_1_2_3 < _b * _c * _d
        _load = tl.load(_x + (_block_idx_0[:, None] * _x_stride_0 + _block_idx_1[None, :] * _x_stride_1 + _block_idx_2[None, :] * _x_stride_2 + _block_idx_3[None, :] * _x_stride_3), _mask_0[:, None] & _mask_1_2_3[None, :], other=0)
        _v_0 = tl_math.sin(_load)
        tl.store(_out + (_block_idx_0[:, None] * _out_stride_0 + _block_idx_1[None, :] * _out_stride_1 + _block_idx_2[None, :] * _out_stride_2 + _block_idx_3[None, :] * _out_stride_3), _v_0, _mask_0[:, None] & _mask_1_2_3[None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1_2_3 = 128
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, c, b, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)
    return out""",
        )
