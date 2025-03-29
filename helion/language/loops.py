from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Protocol
from typing import overload

import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.type_propagation import IterType
from .._compiler.type_propagation import Origin
from .._compiler.type_propagation import SequenceType
from .._compiler.type_propagation import TileIndexType
from .._compiler.type_propagation import TypeInfo
from .._compiler.type_propagation import UnknownType
from . import _decorators

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.generate_ast import CodegenState


__all__ = ["TileIndexProtocol", "tile"]


class TileIndexProtocol(Protocol):
    """
    Opaque type for tile() indices.  Should only be used to index tensors using the
    `tensor[tile]` or `tensor[tile0, tile1]` operator.
    """


@overload
@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: int) -> TileIndexProtocol: ...


@overload
@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: Sequence[int]) -> Sequence[TileIndexProtocol]: ...


@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: int | Sequence[int]) -> TileIndexProtocol | Sequence[TileIndexProtocol]:
    """
    Break up an iteration space defined by a size or sequence of sizes into tiles.
    The generated tiles can flatten the iteration space into the product of the sizes,
    perform multidimensional tiling, swizzle the indices for cache locality, reorder
    dimensions, etc.  The only invariant is that every index in the range of the given
    sizes is covered exactly once.

    The exact tiling strategy is determined by a Config object, typically created
    through autotuning.

    If used at the top level of a function, this becomes the grid of the kernel.
    Otherwise, it becomes a loop in the output kernel.

    Examples:

        for tile in hl.tile(1000):
            ...

        for tile0, tile1 in hl.tile([1000, 1000]):
            ...

    :param sizes: An integer or a sequence of integers representing the sizes for tiling.
    :return: A TileIndexProtocol object if a single size is provided, or a sequence of TileIndexProtocol objects if a sequence of sizes is provided.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(tile)
def _tile_type_prop(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    try:
        proxy_sizes = sizes.proxy()
        if not (
            isinstance(proxy_sizes, int | torch.SymInt)
            or isinstance(proxy_sizes, (tuple, list))
            and all(isinstance(x, (int, torch.SymInt)) for x in proxy_sizes)
        ):
            raise NotImplementedError
    except NotImplementedError:
        return UnknownType(
            origin,
            f"tile() expected int or list[int], got {sizes!s}",
            chained_from=sizes,
        )

    parent = ExtendedAST.current()[-2]
    if not isinstance(parent, ast.For):
        raise exc.LoopFunctionNotInFor("tile")
    if isinstance(proxy_sizes, (int, torch.SymInt)):
        result = TileIndexType.allocate(proxy_sizes, origin)
    else:
        result = SequenceType(
            origin, [TileIndexType.allocate(x, origin) for x in proxy_sizes]
        )
    return IterType(origin, result)


@_decorators.codegen(tile)
def _tile_codegen(state: CodegenState) -> ast.AST:
    for_loop = ExtendedAST.current()[-2]
    loop_type = for_loop._loop_type
    type_info = state.type_info
    assert isinstance(for_loop, ast.For)
    assert for_loop.iter is state.node
    assert isinstance(type_info, IterType)
    if isinstance(type_info.inner, SequenceType):
        tile_indices = type_info.inner.unpack()
    else:
        tile_indices = [type_info.inner]
    assert all(isinstance(t, TileIndexType) for t in tile_indices)
    if loop_type == LoopType.GRID:
        # TODO(jansel): implement 2D tiling, swizzling, etc
        state.tile_strategy.codegen_grid(
            state, [t.block_size_idx for t in tile_indices]
        )
        return state.node
    if loop_type == LoopType.DEVICE:
        raise NotImplementedError("TODO: implement tile() for device loops")
    raise AssertionError(f"Expected loop type: {loop_type}")
