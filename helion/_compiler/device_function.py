from __future__ import annotations

from collections import defaultdict
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

if TYPE_CHECKING:
    import ast

    import torch

    _P = TypeVar("_P", bound="TensorProperty")


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass
class TensorArgument(Argument):
    fake_value: torch.Tensor

    def host_str(self) -> str:
        return self.name  # name is same host/device


@dataclasses.dataclass
class TensorProperty(Argument):
    tensor_arg: TensorArgument
    dim: int


class TensorSize(TensorProperty):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStride(TensorProperty):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


class DeviceFunction:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.arguments: list[Argument] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArgument] = {}
        self._tensor_properties: dict[
            tuple[type[TensorProperty], torch.Tensor, int], TensorProperty
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )

    def unique_name(self, prefix: str) -> str:
        return f"_{prefix}_{next(self._unique_counter[prefix])}"

    def tensor_arg(
        self, fake_value: torch.Tensor, host_name: str | None = None
    ) -> TensorArgument:
        if fake_value not in self._tensor_args:
            arg = TensorArgument(host_name or self.unique_name("tensor"), fake_value)
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    def _tensor_property(
        self, prop_cls: type[_P], fake_value: torch.Tensor, dim: int, prefix: str
    ) -> _P:
        # TODO(jansel): dedupe based on sympy expressions
        key = (prop_cls, fake_value, dim)
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"_{arg.name}_{prefix}{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> TensorSize:
        return self._tensor_property(TensorSize, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> TensorStride:
        return self._tensor_property(TensorStride, fake_value, dim, "stride")
