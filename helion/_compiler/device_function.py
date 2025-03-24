from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast


@dataclasses.dataclass
class Argument:
    name: str


class DeviceFunction:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.arguments: list[Argument] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[str, Argument] = {}

    def add_tensor_arg(self, host_name: str) -> str:
        if host_name not in self._tensor_args:
            arg = Argument(host_name)
            self.arguments.append(arg)
            self._tensor_args[host_name] = arg
        return self._tensor_args[host_name].name
