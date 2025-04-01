from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref

import sympy

from .. import exc
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..runtime.config import Config
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.generate_ast import CodegenState

    _T = TypeVar("_T")


@dataclasses.dataclass
class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
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

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def _reorder(self, block_indices: list[_T]) -> list[_T]:
        if len(block_indices) <= 1:
            return block_indices
        order = next(self.loop_orders)
        assert len(order) == len(block_indices), (
            f"Invalid order length: {len(order)} != {len(block_indices)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_indices[i] for i in reversed(order)]

    def index_var(self, dim: int) -> str:
        return self.fn.new_var(f"block_idx_{dim}")

    def mask_var(self, dim: int | None = None) -> str | None:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        raise NotImplementedError


class FlattenedTileStrategy(TileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    def mask_var(self, dim: int | None = None) -> str | None:
        # TODO(jansel): optimize away unneeded masks and return None
        return self.fn.new_var("mask")

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        device_fn = state.device_function
        offsets_var = device_fn.new_var("offsets")
        mask_var = self.mask_var()

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
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({device_fn.sympy_expr(total_numel)})"
            if i + 1 < len(block_indices):
                expr = f"({expr}) % ({device_fn.sympy_expr(numel)})"
            state.add_statement(f"{block_index_var} = {expr}")
            total_numel = total_numel * numel

        if mask_var is None:
            state.add_statement(f"{mask_var} : tl.constexpr = None")
        else:
            state.add_statement(
                f"{mask_var} = {offsets_var} < ({device_fn.sympy_expr(total_numel)})"
            )
        state.device_function.set_grid_expr(
            expr_from_string(
                f"(triton.cdiv({HostFunction.current().sympy_expr(total_numel)}, {block_size_var}), 1, 1)"
            )
        )


class NDTileStrategy(TileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        config: Config,
    ) -> None:
        super().__init__(fn=fn, config=config)
        self.nontrivial_block_sizes: int = 0
        for sizes in self.config.block_sizes:
            if isinstance(sizes, list):
                self.nontrivial_block_sizes += sum(s != 1 for s in sizes)
            elif sizes != 1:
                self.nontrivial_block_sizes += 1
        self.mask_vars: dict[int, str | None] = {}

    def mask_var(self, dim: int | None = None) -> str | None:
        assert dim is not None
        return self.mask_vars[dim]

    def _expand(self, dim: int) -> str:
        result = ["None"] * self.nontrivial_block_sizes
        result[dim] = ":"
        return ", ".join(result)

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        env = CompileEnvironment.current()
        device_fn = state.device_function
        block_sizes = next(self.block_sizes)
        assert isinstance(block_sizes, list)
        assert len(block_sizes) == len(block_indices)
        if len(block_sizes) > 3:
            raise exc.MaximumGridRank(len(block_sizes))
        grid = []
        dim = 0
        for i, (block_idx, block_size) in enumerate(
            reversed(self._reorder([*zip(block_indices, block_sizes)]))
        ):
            numel = env.block_sizes[block_idx].numel
            offsets_var = device_fn.block_index_var(block_idx)
            if block_size != 1:
                # TODO(jansel): in static shapes mode we can optimize away masks further
                mask_var = self.fn.new_var(f"mask_{dim}")
                self.mask_vars[i] = mask_var
                block_size_var = device_fn.new_var(f"BLOCK_SIZE_{i}")
                state.codegen.host_statements.append(
                    statement_from_string(f"{block_size_var} = {block_size!r}")
                )
                state.device_function.constexpr_arg(block_size_var)
                state.add_statement(
                    f"{offsets_var} = (tl.program_id({i}) * ({block_size_var}) + tl.arange(0, {block_size_var}))"
                    f"[{self._expand(dim)}]"
                )
                state.add_statement(
                    f"{mask_var} = ({offsets_var} < ({device_fn.sympy_expr(numel)}))"
                )
                dim += 1
                grid.append(
                    f"triton.cdiv({HostFunction.current().sympy_expr(numel)}, {block_size_var})"
                )
            else:
                self.mask_vars[i] = None
                state.add_statement(f"{offsets_var} = tl.program_id({i})")
                grid.append(HostFunction.current().sympy_expr(numel))
        state.device_function.set_grid_expr(expr_from_string(f"({', '.join(grid)},)"))
