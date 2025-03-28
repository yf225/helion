from __future__ import annotations

import ast
import inspect
import threading
import typing
from typing import TYPE_CHECKING
from typing import Protocol

from . import ast_extension
from .compile_environment import CompileEnvironment
from .source_location import SourceLocation
from .source_location import UnknownLocation
from .type_printer import print_ast
from .type_propagation import TypeInfo
from .type_propagation import propagate_types

if TYPE_CHECKING:
    import types

    class _TLS(Protocol):
        functions: list[HostFunction]


tls: _TLS = typing.cast("_TLS", threading.local())


class HostFunction:
    def __init__(self, fn: types.FunctionType, fake_args: list[object]) -> None:
        super().__init__()
        self.fn = fn
        self.location: SourceLocation = UnknownLocation()
        self.local_types: dict[str, TypeInfo] | None = None
        with self:
            source = inspect.getsource(fn)
            root = ast.parse(source)
            assert isinstance(root, ast.Module)
            (root,) = root.body
            root = ast_extension.convert(root)
            assert isinstance(root, ast.FunctionDef)
            assert isinstance(root, ast_extension.ExtendedAST)
            self.location = root._location
            self.name: str = root.name
            self.args: ast.arguments = root.args
            self.body: list[ast.stmt] = root.body
            propagate_types(self, fake_args)
            # TODO(jansel): assert we don't have any extra decorators
            # TODO(jansel): check type annotations for hl.constexpr/hl.specialize

    def __repr__(self) -> str:
        return f"<HostFunction {self.name}>"

    def debug_types(self) -> str:
        ast_str = print_ast(
            self.location.to_ast(
                ast.FunctionDef(self.name, self.args, self.body, [], None)
            )
        )
        if error_str := CompileEnvironment.current().errors.report(strip_paths=True):
            return f"{ast_str}\n\n{error_str}"
        return ast_str

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]
        self.location.__enter__()

    def __exit__(self, *args: object) -> None:
        self.location.__exit__(*args)
        tls.functions.pop()

    @staticmethod
    def current() -> HostFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None


class NoCurrentFunction(RuntimeError):
    pass
