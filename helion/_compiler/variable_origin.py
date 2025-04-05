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
        """
        Check if the origin is a host.
        """
        return False

    def is_device(self) -> bool:
        """
        Check if the origin is a device.

        :return: True if the origin is a device, False otherwise.
        """
        return not self.is_host()

    def depth(self) -> int:
        """
        Get the depth of the origin.

        :return: The depth of the origin, which is 1 by default and increases each wrapper.
        """
        return 1

    def host_str(self) -> str:
        """
        Get a string representation of the host origin.

        :raises NotImplementedError: Always raises this error as it should be implemented by subclasses.
        """
        raise NotImplementedError(type(self).__name__)

    def suggest_var_name(self) -> str:
        """
        Suggest a variable name based on the origin.

        :raises NotImplementedError: Always raises this error as it should be implemented by subclasses.
        """
        raise NotImplementedError(type(self).__name__)


@dataclasses.dataclass
class HostOrigin(Origin):
    def is_host(self) -> bool:
        return True


@dataclasses.dataclass
class NameOrigin(HostOrigin):
    """A variable that came from an ast.Name node."""

    name: str

    def __init__(self, name: str, function: HostFunction | None = None) -> None:
        super().__init__()
        self.name = name

    def host_str(self) -> str:
        return self.name

    def suggest_var_name(self) -> str:
        return self.name


class BuiltinOrigin(NameOrigin):
    pass


class GlobalOrigin(NameOrigin):
    pass


class ClosureOrigin(NameOrigin):
    pass


class ArgumentOrigin(NameOrigin):
    pass


@dataclasses.dataclass
class WrappedOrigin(Origin):
    """Keeps track of where a variable came from."""

    value: Origin
    key: int | str

    def is_host(self) -> bool:
        return self.value.is_host()

    def depth(self) -> int:
        return 1 + self.value.depth()


@dataclasses.dataclass
class AttributeOrigin(WrappedOrigin):
    key: str

    def host_str(self) -> str:
        return f"{self.value.host_str()}.{self.key}"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_attr_{self.key}"


@dataclasses.dataclass
class GetItemOrigin(WrappedOrigin):
    def host_str(self) -> str:
        return f"{self.value.host_str()}[{self.key!r}]"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_item_{self.key}"


@dataclasses.dataclass
class TensorSizeOrigin(WrappedOrigin):
    key: int

    def host_str(self) -> str:
        return f"{self.value.host_str()}.size({self.key!r})"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_size_{self.key}"


@dataclasses.dataclass
class SourceOrigin(HostOrigin):
    location: SourceLocation


@dataclasses.dataclass
class DeviceOrigin(Origin):
    location: SourceLocation


@dataclasses.dataclass
class BlockSizeOrigin(Origin):
    block_size_idx: int
