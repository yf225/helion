from __future__ import annotations

from typing import Callable

import torch
from torch.utils._pytree import tree_map

from .. import exc
from ..language._decorators import is_api_func
from .compile_environment import CompileEnvironment

OpOverload = torch._ops.OpOverload


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
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.__getitem__:
            raise NotImplementedError  # TODO(jansel): implement this
        if func is torch.Tensor.__setitem__:
            raise NotImplementedError  # TODO(jansel): implement this
        if is_api_func(func) and func._tiles_as_sizes:
            args, kwargs = tree_map(_tiles_to_sizes, (args, kwargs))
            return func(*args, **kwargs)
        raise exc.IncorrectTileUsage(func)


def _tiles_to_sizes(x: object) -> object:
    if isinstance(x, TileIndexProxy):
        return CompileEnvironment.current().block_sizes[x.block_size_index].var
    return x
