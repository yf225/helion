from __future__ import annotations

import ast
from pathlib import Path
import unittest

from expecttest import TestCase
import torch
from torch._dynamo.source import LocalSource

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.generate_ast import generate_ast
from helion._compiler.host_function import HostFunction
from helion._compiler.type_propagation import propagate_types
from helion._testing import import_path
from helion.runtime import Config

type_prop_inputs = import_path(Path(__file__).parent / "data/type_prop_inputs.py")


def generate_report(fn, *args):
    with CompileEnvironment() as env:
        args = [env.to_fake(arg, LocalSource(f"arg{i}")) for i, arg in enumerate(args)]
        host_fn = HostFunction(fn, env)
        propagate_types(host_fn, args, {})
        return ast.unparse(generate_ast(host_fn, Config()))


class TestTypePropagation(TestCase):
    def test_add(self):
        output = generate_report(
            type_prop_inputs.add,
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
