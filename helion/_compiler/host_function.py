from __future__ import annotations

import ast
import inspect
import threading
import typing
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol

import sympy
from torch._inductor.codegen.wrapper import pexpr

from . import ast_extension
from .compile_environment import CompileEnvironment
from .source_location import SourceLocation
from .source_location import UnknownLocation
from .type_printer import print_ast
from .variable_origin import NameOrigin
from .variable_origin import Origin

if TYPE_CHECKING:
    import types

    import torch

    from .type_propagation import TypeInfo

    class _TLS(Protocol):
        functions: list[HostFunction]


tls: _TLS = typing.cast("_TLS", threading.local())


class SymbolOrigin(NamedTuple):
    origin: Origin
    fake_value: torch.Tensor | None = None

    def depth(self) -> int:
        return self.origin.depth()


class HostFunction:
    def __init__(self, fn: types.FunctionType, fake_args: list[object]) -> None:
        super().__init__()
        self.fn = fn
        self.location: SourceLocation = UnknownLocation()
        self.local_types: dict[str, TypeInfo] | None = None
        self.symbol_to_origin: dict[str, SymbolOrigin] = {}
        self.tensor_to_origin: dict[torch.Tensor, Origin] = {}
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

            from .type_propagation import propagate_types

            propagate_types(self, fake_args)
            # TODO(jansel): assert we don't have any extra decorators
            # TODO(jansel): check type annotations for hl.constexpr/hl.specialize

    def __repr__(self) -> str:
        return f"<HostFunction {self.name}>"

    def set_local_types(self, local_types: dict[str, TypeInfo]) -> None:
        fn = HostFunction.current()
        self.local_types = local_types
        for name, type_info in local_types.items():
            type_info.populate_symbol_origins(NameOrigin(name, fn))

    def sympy_expr(self, expr: sympy.Expr) -> str:
        expr = CompileEnvironment.current().shape_env.simplify(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            origin = self.symbol_to_origin[sym.name].origin
            replacements[sym] = sympy.Symbol(origin.host_str(), integer=True)
        return pexpr(expr.xreplace(replacements))

    def debug_types(self) -> str:
        ast_str = print_ast(
            self.location.to_ast(
                ast.FunctionDef(self.name, self.args, self.body, [], None)
            )
        )
        if error_str := CompileEnvironment.current().errors.report(strip_paths=True):
            return f"{ast_str}\n\n{error_str}"
        return ast_str

    def codegen_function_def(self, statements: list[ast.AST]) -> ast.FunctionDef:
        return ast_extension.create(
            ast.FunctionDef,
            name=self.name,
            args=self.args,
            body=statements,
            decorator_list=[],
        )

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
