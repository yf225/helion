from __future__ import annotations

import ast
import enum
import threading
import typing
from typing import TYPE_CHECKING
from typing import TypeVar

from .source_location import SourceLocation
from .source_location import current_location

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .type_propagation import TypeInfo

    _T = TypeVar("_T", bound=ast.AST)
    _R = TypeVar("_R")

    class _TLS(typing.Protocol):
        active_nodes: list[ExtendedAST]


tls: _TLS = typing.cast("_TLS", threading.local())


class LoopType(enum.Enum):
    UNSET = enum.auto()
    HOST = enum.auto()
    GRID = enum.auto()
    DEVICE = enum.auto()


class ExtendedAST:
    """
    We add some extra functionality to the AST classes, by dynamically
    subclassing each AST node class and mixing in this one.
    """

    # pyre-ignore[13]
    _fields: tuple[str, ...]

    def __init__(
        self,
        *,
        _location: SourceLocation,
        _type_info: TypeInfo | None = None,
        _loop_type: LoopType = LoopType.UNSET,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._type_info: TypeInfo | None = _type_info
        self._location: SourceLocation = _location
        self._loop_type: LoopType = _loop_type

    def new(self, fields: dict[str, object]) -> ExtendedAST:
        result = self.__class__(
            **fields,
            _location=self._location,
            _type_info=self._type_info,
            _loop_type=self._loop_type,
        )
        return self._location.to_ast(result)

    def __repr__(self) -> str:
        assert isinstance(self, ast.AST)
        return ast.dump(self)

    def update_type_info(self, type_info: TypeInfo) -> TypeInfo:
        if self._type_info is not None and type_info != self._type_info:
            type_info = self._type_info.merge(type_info)
        self._type_info = type_info
        return self._type_info

    def debug_annotations(self) -> list[str]:
        result = []
        if self._type_info:
            result.extend(self._type_info.debug_annotations())
        if self._loop_type != LoopType.UNSET:
            result.append(f"loop_type={self._loop_type.name}")
        return result

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


def create(cls: type[_T], **fields: object) -> _T:
    # pyre-ignore[28]
    result = get_wrapper_cls(cls)(**fields, _location=current_location())
    assert isinstance(result, ExtendedAST)
    result._location.to_ast(result)
    return typing.cast("_T", result)


def expr_from_string(template: str, **placeholders: ast.AST) -> ast.AST:
    (expr,) = ast.parse(template).body
    assert isinstance(expr, ast.Expr)
    location: SourceLocation = current_location()

    def _replace(node: _R) -> _R:
        if isinstance(node, list):
            return [_replace(item) for item in node]
        if not isinstance(node, ast.AST):
            return node
        if isinstance(node, ast.Name) and node.id in placeholders:
            return placeholders[node.id]
        cls = get_wrapper_cls(type(node))
        return location.to_ast(
            cls(
                **{field: _replace(getattr(node, field)) for field in node._fields},
                _location=location,
            )
        )

    return _replace(expr.value)


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
