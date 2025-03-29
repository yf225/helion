from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import unittest

from expecttest import TestCase
import torch

from helion._testing import import_path
from helion.runtime import Config

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


def code_and_output(fn: Kernel, args: tuple[object, ...]) -> tuple[str, torch.Tensor]:
    config = Config()
    code = fn.bind(args).to_triton_code(config)
    compiled_kernel = fn.bind(args).compile_config(config)
    return code, compiled_kernel(*args)


class TestGenerateAst(TestCase):
    maxDiff = 16384

    def test_add1d(self):
        args = (torch.randn([4096], device="cuda"), torch.randn([4096], device="cuda"))
        code, result = code_and_output(basic_kernels.add, args)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _out_stride_0, _x_stride_0, _y_stride_0, _BLOCK_SIZE: tl.constexpr):
    _offsets = tl.program_id(0) * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)
    block_idx_0 = _offsets
    _mask = _offsets < _x_size_0
    _v_0 = tl.load(_x + block_idx_0 * _x_stride_0, _mask)
    _v_1 = tl.load(_y + block_idx_0 * _y_stride_0, _mask)
    _v_2 = _v_0 + _v_1
    tl.store(_out + block_idx_0 * _out_stride_0, _v_2, _mask)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE = 1024
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE), 1, 1](x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE)
    return out""",
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add2d(self):
        args = (
            torch.randn([100, 500], device="cuda"),
            torch.randn([100, 500], device="cuda"),
        )
        code, result = code_and_output(basic_kernels.add, args)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _y_stride_0, _y_stride_1, _BLOCK_SIZE: tl.constexpr):
    _offsets = tl.program_id(0) * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)
    block_idx_1 = _offsets % _x_size_1
    block_idx_0 = _offsets // _x_size_1
    _mask = _offsets < _x_size_0 * _x_size_1
    _v_0 = tl.load(_x + (block_idx_0 * _x_stride_0 + block_idx_1 * _x_stride_1), _mask)
    _v_1 = tl.load(_y + (block_idx_0 * _y_stride_0 + block_idx_1 * _y_stride_1), _mask)
    _v_2 = _v_0 + _v_1
    tl.store(_out + (block_idx_0 * _out_stride_0 + block_idx_1 * _out_stride_1), _v_2, _mask)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE)
    return out""",
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add3d(self):
        args = (
            torch.randn([100, 500, 10], device="cuda"),
            torch.randn([100, 500, 10], device="cuda"),
        )
        code, result = code_and_output(basic_kernels.add, args)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
from triton import language as tl

@triton.jit
def _add_kernel(_x, _y, _out, _x_size_0, _x_size_1, _x_size_2, _out_stride_0, _out_stride_1, _out_stride_2, _x_stride_0, _x_stride_1, _x_stride_2, _y_stride_0, _y_stride_1, _y_stride_2, _BLOCK_SIZE: tl.constexpr):
    _offsets = tl.program_id(0) * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)
    block_idx_2 = _offsets % _x_size_2
    block_idx_1 = _offsets // _x_size_2 % _x_size_1
    block_idx_0 = _offsets // (_x_size_1 * _x_size_2)
    _mask = _offsets < _x_size_0 * _x_size_1 * _x_size_2
    _v_0 = tl.load(_x + (block_idx_0 * _x_stride_0 + block_idx_1 * _x_stride_1 + block_idx_2 * _x_stride_2), _mask)
    _v_1 = tl.load(_y + (block_idx_0 * _y_stride_0 + block_idx_1 * _y_stride_1 + block_idx_2 * _y_stride_2), _mask)
    _v_2 = _v_0 + _v_1
    tl.store(_out + (block_idx_0 * _out_stride_0 + block_idx_1 * _out_stride_1 + block_idx_2 * _out_stride_2), _v_2, _mask)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1) * x.size(2), _BLOCK_SIZE), 1, 1](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE)
    return out""",
        )
        torch.testing.assert_close(result, args[0] + args[1])


if __name__ == "__main__":
    unittest.main()
