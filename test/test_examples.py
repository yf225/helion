from __future__ import annotations

from pathlib import Path

from expecttest import TestCase
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
examples_dir = Path(__file__).parent.parent / "examples"


def run_example(
    name: str,
    args: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
    fn_name: str | None = None,
    skip_accuracy=False,
    **kwargs: object,
) -> str:
    code, result = code_and_output(
        getattr(import_path(examples_dir / f"{name}.py"), fn_name or name),
        args,
        **kwargs,
    )
    skip_accuracy or torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
    return code


class TestLoops(TestCase):
    def test_add(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedInline(
            run_example("add", args, sum(args), block_size=128),
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
    _v_0 = _load_1.to(tl.float32)
    _v_1 = _load + _v_0
    tl.store(_out + (_block_idx_0 * _out_stride_0 + _block_idx_1 * _out_stride_1), _v_1, _mask_0_1)

def add(x: torch.Tensor, y: torch.Tensor):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0_1 = 128
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_matmul(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        self.assertExpectedInline(
            run_example(
                "matmul",
                args,
                args[0] @ args[1],
                skip_accuracy=True,
                block_sizes=[[16, 16], 16],
            ),
            """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _matmul_kernel(_x, _y, _out, _out_stride_0, _out_stride_1, _x_stride_0, _x_stride_1, _y_stride_0, _y_stride_1, _m, _n, _k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    _block_idx_0 = tl.program_id(0) * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    _mask_0 = _block_idx_0 < _m
    _block_idx_1 = tl.program_id(1) * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    _mask_1 = _block_idx_1 < _n
    _full = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for _start_2 in range(0, _k, _BLOCK_SIZE_2):
        _block_idx_2 = _start_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        _mask_2 = _block_idx_2 < _k
        _load = tl.load(_x + (_block_idx_0[:, None] * _x_stride_0 + _block_idx_2[None, :] * _x_stride_1), _mask_0[:, None] & _mask_2[None, :], other=0)
        _load_1 = tl.load(_y + (_block_idx_2[:, None] * _y_stride_0 + _block_idx_1[None, :] * _y_stride_1), _mask_2[:, None] & _mask_1[None, :], other=0)
        _mm = tl.dot(_load, _load_1)
        _full = _full + _mm
    tl.store(_out + (_block_idx_0[:, None] * _out_stride_0 + _block_idx_1[None, :] * _out_stride_1), _full, _mask_0[:, None] & _mask_1[None, :])

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(m, _BLOCK_SIZE_0), triton.cdiv(n, _BLOCK_SIZE_1)](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )
