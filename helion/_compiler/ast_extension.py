from __future__ import annotations

import ast
import threading
import typing
from typing import TYPE_CHECKING

from .source_location import SourceLocation
from .source_location import current_location

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .type_propagation import TypeInfo

    class _TLS(typing.Protocol):
        active_nodes: list[ExtendedAST]


tls: _TLS = typing.cast("_TLS", threading.local())


class ExtendedAST:
    """
    We add some extra functionality to the AST classes, by dynamically
    subclassing each AST node class and mixing in this one.
    """

    # pyre-ignore[13]
    _fields: tuple[str, ...]

    def __init__(
        self, *args: object, _location: SourceLocation, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self._type_info: TypeInfo | None = None
        self._location = _location

    def __repr__(self) -> str:
        assert isinstance(self, ast.AST)
        return ast.dump(self)

    def update_type_info(self, type_info: TypeInfo) -> TypeInfo:
        if self._type_info is not None and type_info != self._type_info:
            type_info = self._type_info.merge(type_info)
        self._type_info = type_info
        return self._type_info

    def debug_annotation(self) -> str | None:
        if self._type_info:
            return self._type_info.debug_annotation()
        return None

    def __enter__(self) -> None:
        try:
            tls.active_nodes.append(self)
        except AttributeError:
            tls.active_nodes = [self]
        self._location.__enter__()

    def __exit__(self, *args: object) -> None:
        self._location.__exit__(*args)
        tls.active_nodes.pop()

    @staticmethod
    def current() -> Sequence[ExtendedAST]:
        """Stack of nodes currently being processed."""
        try:
            return tls.active_nodes
        except AttributeError:
            tls.active_nodes = rv = []
            return rv


_to_extended: dict[type[ast.AST], type[ast.AST]] = {}


def get_wrapper_cls(cls: type[ast.AST]) -> type[ast.AST]:
    if new_cls := _to_extended.get(cls):
        return new_cls

    class Wrapper(ExtendedAST, cls):
        pass

    Wrapper.__name__ = cls.__name__
    rv = typing.cast("type[ast.AST]", Wrapper)
    _to_extended[cls] = rv
    return rv


def convert(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.AST):
        cls = get_wrapper_cls(type(node))
        if "lineno" in node._attributes:
            location = SourceLocation.from_ast(node)
        else:
            # some nodes like arguments lack location information
            location = current_location()
        with location:
            # pyre-ignore[28]
            return cls(
                **{field: convert(getattr(node, field)) for field in node._fields},
                **{attr: getattr(node, attr) for attr in node._attributes},
                _location=location,
            )
    elif isinstance(node, list):
        return [convert(item) for item in node]
    else:
        return node
