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
    skip_accuracy or torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
    return code


class TestExamples(TestCase):
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
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    block_idx_1 = offsets_0_1 % x_size_1
    block_idx_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (block_idx_0 * x_stride_0 + block_idx_1 * x_stride_1), mask_0_1, other=0)
    load_1 = tl.load(y + (block_idx_0 * y_stride_0 + block_idx_1 * y_stride_1), mask_0_1, other=0)
    v_0 = load_1.to(tl.float32)
    v_1 = load + v_0
    tl.store(out + (block_idx_0 * out_stride_0 + block_idx_1 * out_stride_1), v_1, mask_0_1)

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
                block_sizes=[[16, 16], 16],
                l2_grouping=4,
            ),
            """\
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 4 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    block_idx_0 = pid_0 * _BLOCK_SIZE_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = block_idx_0 < m
    block_idx_1 = pid_1 * _BLOCK_SIZE_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = block_idx_1 < n
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for start_2 in range(0, k, _BLOCK_SIZE_2):
        block_idx_2 = start_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = block_idx_2 < k
        load = tl.load(x + (block_idx_0[:, None] * x_stride_0 + block_idx_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        load_1 = tl.load(y + (block_idx_2[:, None] * y_stride_0 + block_idx_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    tl.store(out + (block_idx_0[:, None] * out_stride_0 + block_idx_1[None, :] * out_stride_1), acc, mask_0[:, None] & mask_1[None, :])

def matmul(x: torch.Tensor, y: torch.Tensor, acc_dtype: torch.dtype=torch.float32):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )
