from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING
import unittest

from expecttest import TestCase
import torch

from helion._compiler.generate_ast import generate_ast
from helion._testing import import_path
from helion.runtime import Config

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


def generate_report(fn: Kernel, *args):
    bound_kernel = fn.bind(*args)
    with bound_kernel.env:
        return ast.unparse(generate_ast(bound_kernel.host_fn, Config()))


class TestTypePropagation(TestCase):
    def test_add(self):
        output = generate_report(
            basic_kernels.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def _add_kernel():
    tl.store(out + (_block_idx_0 * _out_stride0 + _block_idx_1 * _out_stride1), tl.load(x + (_block_idx_0 * _x_stride0 + _block_idx_1 * _x_stride1), _block_mask_0 | _block_mask_1) + tl.load(y + (_block_idx_0 * _y_stride0 + _block_idx_1 * _y_stride1), _block_mask_0 | _block_mask_1), _block_mask_0 | _block_mask_1)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _add_kernel()
    return out""",
        )


if __name__ == "__main__":
    unittest.main()
