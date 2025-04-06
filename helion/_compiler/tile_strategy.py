from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref

import sympy
import torch

from .. import exc
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..autotuner.config_spec import BlockSizeSpec
    from ..runtime.config import Config
    from .device_function import DeviceFunction
    from .generate_ast import CodegenState

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

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        raise NotImplementedError

    def shape_str(self, shape: Sequence[int | torch.SymInt]) -> str:
        raise NotImplementedError

    def expand_str(self, shape: Sequence[int | torch.SymInt], i: int) -> str:
        raise NotImplementedError

    @classmethod
    def _get_block_index(cls, size: int | torch.SymInt | sympy.Expr) -> int | None:
        if isinstance(size, torch.SymInt):
            return cls._get_block_index(size._sympy_())
        if isinstance(size, sympy.Symbol):
            if isinstance(
                origin := HostFunction.current().symbol_to_origin[size.name].origin,
                BlockSizeOrigin,
            ):
                return origin.block_size_idx
        return None


class FlattenedTileStrategy(TileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    def mask_var(self, block_idx: int) -> str:
        # TODO(jansel): optimize away unneeded masks and return None
        # TODO(jansel): mask_0_1_2
        return self.fn.new_var("mask")

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.new_var("BLOCK_SIZE")

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        total_numel = sympy.S.One
        device_fn = state.device_function
        offsets_var = device_fn.new_var("offsets")
        mask_var = self.mask_var(-1)
        block_size_var = self.block_size_var(-1)
        block_size = next(self.block_sizes)
        assert isinstance(block_size, int)
        state.codegen.host_statements.append(
            statement_from_string(f"{block_size_var} = {block_size!r}")
        )
        state.device_function.constexpr_arg(block_size_var)
        state.add_statement(
            f"{offsets_var} = tl.program_id(0) * ({block_size_var}) + tl.arange(0, {block_size_var}).to({dtype})"
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

        assert mask_var is not None
        state.add_statement(
            f"{mask_var} = {offsets_var} < ({device_fn.sympy_expr(total_numel)})"
        )
        state.device_function.set_grid_expr(
            expr_from_string(
                f"(triton.cdiv({HostFunction.current().sympy_expr(total_numel)}, {block_size_var}), 1, 1)"
            )
        )

    @classmethod
    def update_allow_flattened(
        cls, specs: list[BlockSizeSpec], shape: Sequence[sympy.Expr]
    ) -> None:
        block_cnt = itertools.count()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = cls._get_block_index(x)
            if block_idx is not None:
                if block_idx in used_indices:
                    # multiple usages of the same block size??? bail out
                    for spec in specs:
                        spec.allow_flattened = False
                    return
                used_indices[block_idx] = i
        for spec in specs:
            block_indices = [next(block_cnt) for _ in range(len(spec))]
            if len(block_indices) == 1 or not spec.allow_flattened:
                continue
            if not (
                all(x in used_indices for x in block_indices)
                or all(x not in used_indices for x in block_indices)
            ):
                # A shape must use all or none of the block indices in the group
                spec.allow_flattened = False
                continue
            for i, j in zip(block_indices, block_indices[1:]):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    spec.allow_flattened = False
                    break

    def _compact_shape(
        self, shape: Sequence[int | torch.SymInt]
    ) -> tuple[int, list[int]]:
        # TODO(jansel): support multiple block size groups here (mirror above behavior)
        num_block_sizes = len(CompileEnvironment.current().block_sizes)
        seen_block_size: int = -1
        output_rank = 0
        output = []
        for s in shape:
            block_size_idx = self._get_block_index(s)
            if block_size_idx is None:
                assert seen_block_size == -1
                output.append(output_rank)
                output_rank += 1
            else:
                if seen_block_size == -1:
                    seen_block_size = block_size_idx
                    output.append(output_rank)
                    output_rank += 1
                else:
                    assert seen_block_size + 1 == block_size_idx
                    seen_block_size = block_size_idx
                    output.append(output[-1])
                if seen_block_size == num_block_sizes - 1:
                    seen_block_size = -1
        assert seen_block_size == -1
        return output_rank, output

    def shape_str(self, shape: Sequence[int | torch.SymInt]) -> str:
        rank, compacted = self._compact_shape(shape)
        output: list[str | None] = [None for _ in range(rank)]
        assert len(compacted) == len(shape)
        for i, s in zip(compacted, shape):
            block_size_idx = self._get_block_index(s)
            if block_size_idx is None:
                assert output[i] is None
                output[i] = self.fn.literal_expr(s)
            else:
                output[i] = self.block_size_var(-1)
        return f"[{', '.join(map(str, output))}]"

    def expand_str(self, shape: Sequence[int | torch.SymInt], i: int) -> str:
        rank, compacted = self._compact_shape(shape)
        output = ["None"] * rank
        output[compacted[i]] = ":"
        if output == [":"]:
            return ""
        return f"[{', '.join(output)}]"


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
        self.block_size_vars: dict[int, str | None] = {}

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def block_size_var(self, block_idx: int) -> str | None:
        return self.block_size_vars[block_idx]

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        env = CompileEnvironment.current()
        device_fn = state.device_function
        dtype = env.triton_index_type()
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
            offsets_var = self.index_var(block_idx)
            if block_size != 1:
                # TODO(jansel): in static shapes mode we can optimize away masks further
                mask_var = self.fn.new_var(f"mask_{block_idx}")
                self.mask_vars[block_idx] = mask_var
                self.block_size_vars[block_idx] = block_size_var = device_fn.new_var(
                    f"BLOCK_SIZE_{block_idx}"
                )
                state.codegen.host_statements.append(
                    statement_from_string(f"{block_size_var} = {block_size!r}")
                )
                state.device_function.constexpr_arg(block_size_var)
                state.add_statement(
                    f"{offsets_var} = tl.program_id({i}) * ({block_size_var}) + tl.arange(0, ({block_size_var})).to({dtype})"
                )
                state.add_statement(
                    f"{mask_var} = ({offsets_var} < ({device_fn.sympy_expr(numel)}))"
                )
                dim += 1
                grid.append(
                    f"triton.cdiv({HostFunction.current().sympy_expr(numel)}, {block_size_var})"
                )
            else:
                self.mask_vars[block_idx] = None
                self.block_size_vars[block_idx] = None
                dtype = CompileEnvironment.current().triton_index_type()
                state.add_statement(
                    f"{offsets_var} = tl.program_id({i}) + tl.zeros([1], {dtype})"
                )
                grid.append(HostFunction.current().sympy_expr(numel))
        state.device_function.set_grid_expr(expr_from_string(f"({', '.join(grid)},)"))

    def shape_str(self, shape: Sequence[int | torch.SymInt]) -> str:
        result = []
        for s in shape:
            if self.is_dropped_size(s):
                continue  # drop size=1 dimensions
            result.append(self.fn.literal_expr(s))
        return f"[{', '.join(result)}]"

    def expand_str(self, shape: Sequence[int | torch.SymInt], i: int) -> str:
        assert 0 <= i < len(shape)
        result = []
        nextval = "None"
        for j, s in enumerate(shape):
            if not self.is_dropped_size(s):
                result.append(nextval)
                nextval = "None"
            if i == j:
                if result:
                    result[-1] = ":"
                else:
                    nextval = ":"
        if result == [":"]:
            return ""
        return f"[{', '.join(result)}]"

    def is_dropped_size(self, size: int | torch.SymInt) -> bool:
        # TODO(jansel): enable this optimization
        """
        if isinstance(size, torch.SymInt):
            if isinstance(sym := size._sympy_(), sympy.Symbol):
                if isinstance(
                    origin := HostFunction.current().symbol_to_origin[sym.name].origin,
                    BlockSizeOrigin,
                ):
                    return self.block_size_vars[origin.block_size_idx] is None
        return False
        """
        return False
