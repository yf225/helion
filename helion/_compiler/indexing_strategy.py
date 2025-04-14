from __future__ import annotations

import ast
import collections
from typing import TYPE_CHECKING
from typing import NamedTuple
import weakref

import sympy
import torch

from .. import exc
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState


class IndexingStrategy:
    def __init__(self, device_fn: DeviceFunction) -> None:
        self._fn: weakref.ReferenceType[DeviceFunction] = weakref.ref(device_fn)

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def codegen_load(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError

    def codegen_store(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError

    @staticmethod
    def select(device_fn: DeviceFunction, config: Config) -> IndexingStrategy:
        indexing = config.indexing
        if indexing == "pointer":
            return PointerIndexingStrategy(device_fn)
        if indexing == "tensor_descriptor":
            return TensorDescriptorIndexingStrategy(device_fn)
        if indexing == "block_ptr":
            return BlockPtrIndexingStrategy(device_fn)
        raise RuntimeError(
            f"Invalid indexing strategy: {indexing!r}, "
            "must be one of 'pointer', 'tensor_descriptor', 'block_ptr'"
        )


class PointerIndexingStrategy(IndexingStrategy):
    """Generate the original pointer math to load/store from tensors"""

    def codegen_load(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript)
        extra = ", other=0" if indexing.has_mask() else ""
        return expr_from_string(
            # TODO(jansel): optimize away mask/other?
            f"tl.load(name + offset, mask{extra})",
            name=state.ast_arg(0),
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )

    def codegen_store(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(
            state,
            fake_tensor,
            subscript,
        )
        return expr_from_string(
            "tl.store(name + offset, value, mask)",
            value=state.ast_arg(2),
            name=state.ast_arg(0),
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )


class TensorDescriptorIndexingStrategy(IndexingStrategy):
    """Use TensorDescriptor to load/store from tensors"""

    def codegen_load(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError

    def codegen_store(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError


class BlockPtrIndexingStrategy(IndexingStrategy):
    """Use block_ptr to load/store from tensors"""

    def codegen_load(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError

    def codegen_store(
        self, state: CodegenState, fake_tensor: torch.Tensor, subscript: list[object]
    ) -> ast.AST:
        raise NotImplementedError


class SubscriptIndexing(NamedTuple):
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
        state: CodegenState, fake_value: torch.Tensor, index: list[object]
    ) -> SubscriptIndexing:
        tile_strategy = state.tile_strategy
        output_idx = 0
        index_values = []
        mask_values = {}
        output_size = SubscriptIndexing.compute_shape(fake_value, index)
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
                    index_var = tile_strategy.index_var(origin.origin.block_size_idx)
                    index_values.append(f"({index_var}){expand}")
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

        return SubscriptIndexing(
            expr_from_string("+".join(index_expr)),
            expr_from_string("&".join(mask_values) or "None"),
        )
