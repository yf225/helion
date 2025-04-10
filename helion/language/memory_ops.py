from __future__ import annotations

import ast
import collections
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.host_function import HostFunction
from .._compiler.variable_origin import BlockSizeOrigin
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["load", "store"]


@_decorators.api(tiles_as_sizes=True)
def store(tensor: torch.Tensor, index: list[object], value: torch.Tensor) -> None:
    raise exc.NotInsideKernel


@_decorators.register_fake(store)
def _(tensor: torch.Tensor, index: list[object], value: torch.Tensor) -> None:
    return None


@_decorators.codegen(store)
def _(state: CodegenState) -> ast.AST:
    indexing = _SubscriptIndexing.create(
        state,
        state.proxy_arg(0),
        state.proxy_arg(1),
    )
    return expr_from_string(
        "tl.store(name + offset, value, mask)",
        value=state.ast_arg(2),
        name=state.ast_arg(0),
        offset=indexing.index_expr,
        mask=indexing.mask_expr,
    )


@_decorators.api(tiles_as_sizes=True)
def load(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    raise exc.NotInsideKernel


@_decorators.register_fake(load)
def _(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    return tensor.new_empty(_SubscriptIndexing.compute_shape(tensor, index))


@_decorators.codegen(load)
def _(state: CodegenState) -> ast.AST:
    indexing = _SubscriptIndexing.create(state, state.proxy_arg(0), state.proxy_arg(1))
    extra = ", other=0" if indexing.has_mask() else ""
    return expr_from_string(
        # TODO(jansel): optimize away mask/other?
        f"tl.load(name + offset, mask{extra})",
        name=state.ast_arg(0),
        offset=indexing.index_expr,
        mask=indexing.mask_expr,
    )


class _SubscriptIndexing(NamedTuple):
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )

    @staticmethod
    def compute_shape(
        tensor: torch.Tensor, index: list[object]
    ) -> list[int | torch.SymInt]:
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(index, (list, tuple))
        input_size = collections.deque(tensor.size())
        output_size = []
        for k in index:
            if k is None:
                output_size.append(1)
            elif isinstance(k, int):
                input_size.popleft()
            elif isinstance(k, torch.SymInt):
                input_size.popleft()
                symbol = k._sympy_()
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().symbol_to_origin.get(symbol.name)
                    if origin and isinstance(origin.origin, BlockSizeOrigin):
                        output_size.append(k)
            else:
                raise exc.InvalidIndexingType(k)
        assert len(input_size) == 0, "invalid subscript"
        return output_size

    @staticmethod
    def create(
        state: CodegenState, fake_value: object, index: object
    ) -> _SubscriptIndexing:
        assert isinstance(fake_value, torch.Tensor)
        assert isinstance(index, (list, tuple))
        tile_strategy = state.tile_strategy
        output_idx = 0
        index_values = []
        mask_values = {}
        output_size = _SubscriptIndexing.compute_shape(fake_value, [*index])
        dtype = CompileEnvironment.current().triton_index_type()
        for k in index:
            if k is None:
                output_idx += 1
            elif isinstance(k, int):
                expand = tile_strategy.expand_str(output_size, output_idx)
                index_values.append(f"tl.full([1], {k!r}, {dtype}){expand}")
            elif isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().symbol_to_origin.get(symbol.name)
                expand = tile_strategy.expand_str(output_size, output_idx)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    index = tile_strategy.index_var(origin.origin.block_size_idx)
                    index_values.append(f"({index}){expand}")
                    if mask := tile_strategy.mask_var(origin.origin.block_size_idx):
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                else:
                    val = state.device_function.literal_expr(k)
                    index_values.append(f"tl.full([1], {val}, {dtype}){expand}")
            else:
                raise exc.InvalidIndexingType(k)
        assert len(output_size) == output_idx
        assert len(index_values) == fake_value.ndim

        index_expr = []
        for i, idx in enumerate(index_values):
            if fake_value.size(i) != 1:
                stride = state.device_function.tensor_stride(fake_value, i).name
                index_expr.append(f"{idx} * {stride}")
        if not index_expr:
            shape_str = tile_strategy.shape_str(output_size)
            index_expr.append(f"tl.zeros({shape_str}, {dtype})")

        return _SubscriptIndexing(
            expr_from_string("+".join(index_expr)),
            expr_from_string("|".join(mask_values) or "None"),
        )
