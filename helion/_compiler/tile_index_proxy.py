from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import TypeVar

import torch
from torch.utils._pytree import tree_map_only

from .. import exc
from .compile_environment import CompileEnvironment

OpOverload = torch._ops.OpOverload

if TYPE_CHECKING:
    _T = TypeVar("_T")


class TileIndexProxy:
    def __init__(self, block_size_index: int) -> None:
        super().__init__()
        self.block_size_index = block_size_index

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is torch.Tensor.__getitem__:
            raise NotImplementedError  # TODO(jansel): implement this
        if func is torch.Tensor.__setitem__:
            raise NotImplementedError  # TODO(jansel): implement this
        raise exc.IncorrectTileUsage(func)

    @classmethod
    def tiles_to_sizes(cls, it: _T) -> _T:
        return tree_map_only(TileIndexProxy, cls._tile_to_size, it)

    @staticmethod
    def _tile_to_size(x: TileIndexProxy) -> torch.SymInt:
        return CompileEnvironment.current().block_sizes[x.block_size_index].var
