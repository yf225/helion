from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_function import HostFunction
    from .source_location import SourceLocation


@dataclasses.dataclass
class Origin:
    """Keeps track of where a variable came from."""

    def is_host(self) -> bool:
        return False


@dataclasses.dataclass
class HostOrigin(Origin):
    def is_host(self) -> bool:
        return True


@dataclasses.dataclass
class NameOrigin(HostOrigin):
    """A variable that came from an ast.Name node."""

    name: str
    function: HostFunction


@dataclasses.dataclass
class BuiltinOrigin(NameOrigin):
    pass


@dataclasses.dataclass
class GlobalOrigin(NameOrigin):
    pass


@dataclasses.dataclass
class ClosureOrigin(NameOrigin):
    pass


@dataclasses.dataclass
class ArgumentOrigin(NameOrigin):
    pass


@dataclasses.dataclass
class WrappedOrigin(Origin):
    """Keeps track of where a variable came from."""

    value: Origin
    key: int | str

    def is_host(self) -> bool:
        return self.value.is_host()


@dataclasses.dataclass
class AttributeOrigin(WrappedOrigin):
    """Keeps track of where a variable came from."""

    key: str


@dataclasses.dataclass
class GetItemOrigin(WrappedOrigin):
    """Keeps track of where a variable came from."""


@dataclasses.dataclass
class SourceOrigin(HostOrigin):
    location: SourceLocation


@dataclasses.dataclass
class DeviceOrigin(Origin):
    location: SourceLocation
