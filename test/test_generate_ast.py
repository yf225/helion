from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


class TestGenerateAst(TestCase):
    maxDiff = 16384

    def test_add1d(self):
        args = (torch.randn([4096], device=DEVICE), torch.randn([4096], device=DEVICE))
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _out_stride_0, _x_stride_0, _y_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _load = tl.load(_x + _block_idx_0 * _x_stride_0, _mask_0, other=0)
    _load_1 = tl.load(_y + _block_idx_0 * _y_stride_0, _mask_0, other=0)
    _v_0 = _load + _load_1
    tl.store(_out + _block_idx_0 * _out_stride_0, _v_0, _mask_0)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 1024
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add2d(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _y_stride_0, _y_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    _offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    _block_idx_1 = _offsets_0_1 % _x_size_1
    _block_idx_0 = _offsets_0_1 // _x_size_1
    _mask_0_1 = _offsets_0_1 < _x_size_0 * _x_size_1
    _load = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1), _mask_0_1, other=0)
    _load_1 = tl.load(_y + (_block_idx_0 * _y_stride_0 + _block_idx_1 * _y_stride_1), _mask_0_1, other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1), _v_0, _mask_0_1)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add2d_loop_order(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=1024, loop_order=(1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _y_stride_0, _y_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    _offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    _block_idx_0 = _offsets_0_1 % _x_size_0
    _block_idx_1 = _offsets_0_1 // _x_size_0
    _mask_0_1 = _offsets_0_1 < _x_size_0 * _x_size_1
    _load = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1), _mask_0_1, other=0)
    _load_1 = tl.load(_y + (_block_idx_0 * _y_stride_0 + _block_idx_1 * _y_stride_1), _mask_0_1, other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1), _v_0, _mask_0_1)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add3d(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_0_1_2: tl.constexpr):
    _offsets_0_1_2 = tl.program_id(0) * _BLOCK_SIZE_0_1_2 + tl.arange(0, _BLOCK_SIZE_0_1_2).to(tl.int32)
    _block_idx_2 = _offsets_0_1_2 % _x_size_2
    _block_idx_1 = _offsets_0_1_2 // _x_size_2 % _x_size_1
    _block_idx_0 = _offsets_0_1_2 // (_x_size_1 * _x_size_2)
    _mask_0_1_2 = _offsets_0_1_2 < _x_size_0 * _x_size_1 * _x_size_2
    _load = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1 + _block_idx_2 * _x_stride_2), _mask_0_1_2, other=0)
    _load_1 = tl.load(_y + (_block_idx_0 * _y_stride_0 + _block_idx_1 * _y_stride_1 + _block_idx_2 * _y_stride_2), _mask_0_1_2, other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1 + _block_idx_2 * _out_stride_2), _v_0, _mask_0_1_2)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1) * x.size(2), _BLOCK_SIZE_0_1_2), 1, 1](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add3d_reorder(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=1024, loop_order=(2, 0, 1)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_0_1_2: tl.constexpr):
    _offsets_0_1_2 = tl.program_id(0) * _BLOCK_SIZE_0_1_2 + tl.arange(0, _BLOCK_SIZE_0_1_2).to(tl.int32)
    _block_idx_1 = _offsets_0_1_2 % _x_size_1
    _block_idx_0 = _offsets_0_1_2 // _x_size_1 % _x_size_0
    _block_idx_2 = _offsets_0_1_2 // (_x_size_0 * _x_size_1)
    _mask_0_1_2 = _offsets_0_1_2 < _x_size_0 * _x_size_1 * _x_size_2
    _load = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1 + _block_idx_2 * _x_stride_2), _mask_0_1_2, other=0)
    _load_1 = tl.load(_y + (_block_idx_0 * _y_stride_0 + _block_idx_1 * _y_stride_1 + _block_idx_2 * _y_stride_2), _mask_0_1_2, other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1 + _block_idx_2 * _out_stride_2), _v_0, _mask_0_1_2)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1) * x.size(2), _BLOCK_SIZE_0_1_2), 1, 1](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add_tilend0(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[8, 16, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _block_idx_1 = tl.program_id(1) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _x_size_1
    _block_idx_2 = tl.program_id(2) * _BLOCK_SIZE_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    _mask_2 = _block_idx_2 < _x_size_2
    _load = tl.load(_x + (_block_idx_0[:, None, None] * _x_stride_0 + _block_idx_1[None, :, None] * _x_stride_1 + _block_idx_2[None, None, :] * _x_stride_2), _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _load_1 = tl.load(_y + (_block_idx_0[:, None, None] * _y_stride_0 + _block_idx_1[None, :, None] * _y_stride_1 + _block_idx_2[None, None, :] * _y_stride_2), _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0[:, None, None] * _out_stride_0 + _block_idx_1[None, :, None] * _out_stride_1 + _block_idx_2[None, None, :] * _out_stride_2), _v_0, _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 32
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0), triton.cdiv(x.size(1), _BLOCK_SIZE_1), triton.cdiv(x.size(2), _BLOCK_SIZE_2)](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add_tilend1(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[8, 16, 32], loop_order=(2, 1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    _block_idx_2 = tl.program_id(0) * _BLOCK_SIZE_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    _mask_2 = _block_idx_2 < _x_size_2
    _block_idx_1 = tl.program_id(1) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _x_size_1
    _block_idx_0 = tl.program_id(2) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _load = tl.load(_x + (_block_idx_0[:, None, None] * _x_stride_0 + _block_idx_1[None, :, None] * _x_stride_1 + _block_idx_2[None, None, :] * _x_stride_2), _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _load_1 = tl.load(_y + (_block_idx_0[:, None, None] * _y_stride_0 + _block_idx_1[None, :, None] * _y_stride_1 + _block_idx_2[None, None, :] * _y_stride_2), _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0[:, None, None] * _out_stride_0 + _block_idx_1[None, :, None] * _out_stride_1 + _block_idx_2[None, None, :] * _out_stride_2), _v_0, _mask_0[:, None, None] | _mask_1[None, :, None] | _mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_2 = 32
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 8
    _add_kernel[triton.cdiv(x.size(2), _BLOCK_SIZE_2), triton.cdiv(x.size(1), _BLOCK_SIZE_1), triton.cdiv(x.size(0), _BLOCK_SIZE_0)](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add_tilend2(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[1, 32, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    _block_idx_0 = tl.program_id(0) + tl.zeros([1], tl.int32)
    _block_idx_1 = tl.program_id(1) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _x_size_1
    _block_idx_2 = tl.program_id(2) * _BLOCK_SIZE_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    _mask_2 = _block_idx_2 < _x_size_2
    _load = tl.load(_x + (_block_idx_0[:, None, None] * _x_stride_0 + _block_idx_1[None, :, None] * _x_stride_1 + _block_idx_2[None, None, :] * _x_stride_2), _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _load_1 = tl.load(_y + (_block_idx_0[:, None, None] * _y_stride_0 + _block_idx_1[None, :, None] * _y_stride_1 + _block_idx_2[None, None, :] * _y_stride_2), _mask_1[None, :, None] | _mask_2[None, None, :], other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0[:, None, None] * _out_stride_0 + _block_idx_1[None, :, None] * _out_stride_1 + _block_idx_2[None, None, :] * _out_stride_2), _v_0, _mask_1[None, :, None] | _mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 32
    _add_kernel[x.size(0), triton.cdiv(x.size(1), _BLOCK_SIZE_1), triton.cdiv(x.size(2), _BLOCK_SIZE_2)](x, y, out, x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_add_tilend3(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_size=[1, 32, 1],
            loop_order=(0, 2, 1),
            num_warps=8,
            num_stages=1,
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_1, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE_1: tl.constexpr):
    _block_idx_0 = tl.program_id(0) + tl.zeros([1], tl.int32)
    _block_idx_2 = tl.program_id(1) + tl.zeros([1], tl.int32)
    _block_idx_1 = tl.program_id(2) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _x_size_1
    _load = tl.load(_x + (_block_idx_0[:, None, None] * _x_stride_0 + _block_idx_1[None, :, None] * _x_stride_1 + _block_idx_2[None, None, :] * _x_stride_2), _mask_1[None, :, None], other=0)
    _load_1 = tl.load(_y + (_block_idx_0[:, None, None] * _y_stride_0 + _block_idx_1[None, :, None] * _y_stride_1 + _block_idx_2[None, None, :] * _y_stride_2), _mask_1[None, :, None], other=0)
    _v_0 = _load + _load_1
    tl.store(_out + (_block_idx_0[:, None, None] * _out_stride_0 + _block_idx_1[None, :, None] * _out_stride_1 + _block_idx_2[None, None, :] * _out_stride_2), _v_0, _mask_1[None, :, None])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    _add_kernel[x.size(0), x.size(2), triton.cdiv(x.size(1), _BLOCK_SIZE_1)](x, y, out, x.size(1), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, num_warps=8, num_stages=1)
    return out""",
        )

    def test_torch_ops_pointwise(self):
        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.torch_ops_pointwise,
            args,
            block_size=128,
        )
        torch.testing.assert_close(
            result, torch.sigmoid(torch.add(torch.sin(args[0]), torch.cos(args[1])))
        )
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _torch_ops_pointwise_kernel(_x, _y, _out, _x_size_0, _out_stride_0, _x_stride_0, _y_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _load = tl.load(_x + _block_idx_0 * _x_stride_0, _mask_0, other=0)
    _v_0 = tl_math.sin(_load)
    _load_1 = tl.load(_y + _block_idx_0 * _y_stride_0, _mask_0, other=0)
    _v_1 = tl_math.cos(_load_1)
    _v_2 = _v_0 + _v_1
    _v_3 = tl.sigmoid(_v_2)
    tl.store(_out + _block_idx_0 * _out_stride_0, _v_3, _mask_0)

def torch_ops_pointwise(x, y):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    _torch_ops_pointwise_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out""",
        )

    def test_hl_zeros_usage(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_size=[32, 32],
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _hl_zeros_usage_kernel(_x, _out, _x_size_0, _x_size_1, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _block_idx_1 = tl.program_id(1) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _x_size_1
    _full = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    _load = tl.load(_x + (_block_idx_0[:, None] * _x_stride_0 + _block_idx_1[None, :] * _x_stride_1), _mask_0[:, None] | _mask_1[None, :], other=0)
    _v_0 = _full + _load
    _load_1 = tl.load(_x + (_block_idx_0[:, None] * _x_stride_0 + _block_idx_1[None, :] * _x_stride_1), _mask_0[:, None] | _mask_1[None, :], other=0)
    _v_1 = _v_0 + _load_1
    tl.store(_out + (_block_idx_0[:, None] * _out_stride_0 + _block_idx_1[None, :] * _out_stride_1), _v_1, _mask_0[:, None] | _mask_1[None, :])

def hl_zeros_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _hl_zeros_usage_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0), triton.cdiv(x.size(1), _BLOCK_SIZE_1)](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_hl_full_usage(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_full_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2 + 1)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _hl_full_usage_kernel(_x, _out, _x_size_0, _out_stride_0, _x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _x_size_0
    _full = tl.full([_BLOCK_SIZE_0], 1, tl.float32)
    _load = tl.load(_x + _block_idx_0 * _x_stride_0, _mask_0, other=0)
    _v_0 = _full + _load
    _load_1 = tl.load(_x + _block_idx_0 * _x_stride_0, _mask_0, other=0)
    _v_1 = _v_0 + _load_1
    tl.store(_out + _block_idx_0 * _out_stride_0, _v_1, _mask_0)

def hl_full_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    _hl_full_usage_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out""",
        )

    def test_hl_zeros_flat(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _hl_zeros_usage_kernel(_x, _out, _x_size_0, _x_size_1, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    _offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    _block_idx_1 = _offsets_0_1 % _x_size_1
    _block_idx_0 = _offsets_0_1 // _x_size_1
    _mask_0_1 = _offsets_0_1 < _x_size_0 * _x_size_1
    _full = tl.full([_BLOCK_SIZE_0_1], 0.0, tl.float32)
    _load = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1), _mask_0_1, other=0)
    _v_0 = _full + _load
    _load_1 = tl.load(_x + (_block_idx_0 * _x_stride_0 + _block_idx_1 * _x_stride_1), _mask_0_1, other=0)
    _v_1 = _v_0 + _load_1
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1), _v_1, _mask_0_1)

def hl_zeros_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 128
    _hl_zeros_usage_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out""",
        )


if __name__ == "__main__":
    unittest.main()
