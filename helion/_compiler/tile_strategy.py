from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
import weakref

import sympy

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..runtime.config import Config
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.generate_ast import CodegenState


@dataclasses.dataclass
class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    needed_blocks: dict[int, bool]
    block_sizes: Iterator[list[int] | int]
    loop_orders: Iterator[list[int]]

    def __init__(
        self,
        fn: DeviceFunction,
        config: Config,
    ) -> None:
        self._fn = weakref.ref(fn)
        self.config = config
        self.block_sizes = iter(config.block_sizes)
        self.loop_orders = iter(config.loop_orders)
        self.needed_blocks = {}

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def _reorder(self, block_indices: list[int]) -> list[int]:
        if len(block_indices) <= 1:
            return block_indices
        order = next(self.loop_orders)
        assert len(order) == len(block_indices), (
            f"Invalid order length: {len(order)} != {len(block_indices)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_indices[i] for i in reversed(order)]

    def index_var(self, dim: int) -> str:
        self.needed_blocks[dim] = True
        return f"block_idx_{dim}"

    def mask_var(self, dim: int | None = None) -> str | None:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        raise NotImplementedError


class FlattenedTileStrategy(TileStrategy):
    """Collapse all dimensions into single flat iteration space."""

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
        block_size = next(self.block_sizes)
        assert isinstance(block_size, int)
        state.codegen.host_statements.append(
            statement_from_string(f"{block_size_var} = {block_size!r}")
        )
        state.device_function.constexpr_arg(block_size_var)
        state.add_statement(
            f"{offsets_var} = tl.program_id(0) * ({block_size_var}) + tl.arange(0, {block_size_var})"
        )
        for i, block_idx in enumerate(self._reorder(block_indices)):
            # need to get the block size
            numel = env.block_sizes[block_idx].numel
            block_index_var = device_fn.tile_strategy.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({device_fn.sympy_expr(total_numel)})"
            if i + 1 < len(block_indices):
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


class NDTileStrategy(TileStrategy):
    pass
