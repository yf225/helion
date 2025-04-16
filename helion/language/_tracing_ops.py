from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from .._compiler.ast_extension import expr_from_string
from .._compiler.host_function import HostFunction
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""


@_decorators.api()
def _get_symnode(debug_name: str) -> int:
    """FX requires a torch.SymInt to come from an op. This is a fake op is added lazily to work around this."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_get_symnode)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_get_symnode")  # should be unused


@_decorators.api()
def _host_tensor(debug_name: str) -> torch.Tensor:
    """Source of a tensor that was allocated on the host and must be passed to the kernel as an arg."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")  # should be unused


@_decorators.api()
def _for_loop(graph_id: int, args: list[object]) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop)
def _(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)


@_decorators.api()
def _phi(lhs: object, rhs: object) -> object:
    """Combine values from different branches of a control flow."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_phi)
def _(lhs: object, rhs: object) -> object:
    assert isinstance(lhs, torch.Tensor), lhs
    assert isinstance(rhs, torch.Tensor), rhs
    assert lhs.size() == rhs.size()
    assert lhs.dtype == rhs.dtype
    assert lhs.device == rhs.device
    return torch.empty_like(lhs)


@_decorators.codegen(_phi)
def _(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs
