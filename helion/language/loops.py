from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Protocol
from typing import overload

import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
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


class TileIndexProtocol(Protocol):
    pass


@overload
@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: int) -> TileIndexProtocol: ...


@overload
@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: Sequence[int]) -> Sequence[TileIndexProtocol]: ...


@_decorators.api(is_device_loop=True, is_device_only=False)
def tile(sizes: int | Sequence[int]) -> TileIndexProtocol | Sequence[TileIndexProtocol]:
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
    # TODO(jansel): implement this
    return state.node
