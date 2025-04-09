from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..language import _decorators
from .ast_extension import create
from .ast_extension import expr_from_string

if TYPE_CHECKING:
    from .inductor_lowering import CodegenState


@_decorators.api_custom_op()
def _get_symnode(debug_name: str) -> int:
    raise AssertionError("this should never be called")


@_decorators.codegen(_get_symnode)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_get_symnode")  # should be unused


@_decorators.api_custom_op()
def _host_tensor(debug_name: str) -> torch.Tensor:
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor)
def _(state: CodegenState) -> ast.AST:
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    name = state.device_function.tensor_arg(
        fake_value, prefer_name=state.fx_node.name
    ).name
    return create(ast.Name, id=name, ctx=ast.Load())
