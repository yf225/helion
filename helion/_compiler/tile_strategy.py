from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING
import weakref

import sympy

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.generate_ast import CodegenState


@dataclasses.dataclass
class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    needed_blocks: dict[int, bool] = dataclasses.field(default_factory=dict)

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def index_var(self, dim: int) -> str:
        raise NotImplementedError

    def mask_var(self, dim: int | None = None) -> str | None:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        raise NotImplementedError


class FlattenedTileStrategy(TileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    def __init__(
        self,
        fn: DeviceFunction,
        orders: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(weakref.ref(fn))
        self.orders = orders
        self.orders_index: itertools.count[int] = itertools.count()

    def _reorder(self, block_indices: list[int]) -> list[int]:
        if self.orders is None:
            return [*reversed(block_indices)]
        order = self.orders[next(self.orders_index)]
        assert len(order) == len(block_indices), (
            f"Invalid order length: {len(order)} != {len(block_indices)}"
        )
        assert set(order) == set(range(len(order))), f"Invalid permutation: {order}"
        return [block_indices[i] for i in order]

    def index_var(self, dim: int) -> str:
        self.needed_blocks[dim] = True
        return f"block_idx_{dim}"

    def mask_var(self, dim: int | None = None) -> str | None:
        # TODO(jansel): optimize away unneeded masks and return None
        # TODO(jansel): support multidimensional blocks
        return self.fn.new_var("mask")

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        device_fn = state.device_function
        offsets_var = device_fn.new_var("offsets")
        mask_var = device_fn.tile_strategy.mask_var()

        block_size_var = device_fn.new_var("BLOCK_SIZE")
        block_size = 1024  # TODO(jansel): read from config
        state.codegen.host_statements.append(
            statement_from_string(f"{block_size_var} = {block_size!r}")
        )
        state.device_function.constexpr_arg(block_size_var)
        state.add_statement(
            f"{offsets_var} = tl.program_id(0) * ({block_size_var}) + tl.arange(0, {block_size_var})"
        )
        for i in self._reorder(block_indices):
            # need to get the block size
            numel = env.block_sizes[i].numel

            # TODO(jansel): get the block size from the Config
            block_index_var = device_fn.tile_strategy.index_var(i)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({device_fn.sympy_expr(total_numel)})"
            if i != block_indices[0]:
                expr = f"({expr}) % ({device_fn.sympy_expr(numel)})"
            state.add_statement(f"{block_index_var} = {expr}")
            total_numel = total_numel * numel

        state.add_statement(
            f"{mask_var} = {offsets_var} < ({device_fn.sympy_expr(total_numel)})"
        )
        state.device_function.set_grid_expr(
            expr_from_string(
                f"(triton.cdiv({HostFunction.current().sympy_expr(total_numel)}, {block_size_var}), 1, 1)"
            )
        )
