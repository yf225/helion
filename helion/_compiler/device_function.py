from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.codegen.triton import texpr

from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .tile_strategy import FlattenedTileStrategy
from .tile_strategy import NDTileStrategy
from .tile_strategy import TileStrategy
from .variable_origin import BlockSizeOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config

    _P = TypeVar("_P", bound="TensorPropertyArg")


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError

    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name)

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)],)


@dataclasses.dataclass
class TensorArg(Argument):
    fake_value: torch.Tensor
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


@dataclasses.dataclass
class TensorPropertyArg(Argument):
    tensor_arg: TensorArg
    dim: int

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)], self.tensor_arg.name, self.dim)


class TensorSizeArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStrideArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


@dataclasses.dataclass
class NumericArgument(Argument):
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


class ConstExprArg(NumericArgument):
    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name, "tl.constexpr")


@dataclasses.dataclass
class SymbolArgument(NumericArgument):
    pass


_sort_order: dict[type[Argument], int] = {
    TensorArg: 0,
    TensorSizeArg: 1,
    TensorStrideArg: 2,
    SymbolArgument: 3,
    ConstExprArg: 4,
}


class DeviceFunction:
    def __init__(self, name: str, config: Config) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.arguments: list[Argument] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._symbol_args: dict[str, SymbolArgument] = {}
        self._tensor_properties: dict[
            tuple[type[TensorPropertyArg], torch.Tensor, int], TensorPropertyArg
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        if isinstance(config.block_sizes[0], int):
            self.tile_strategy: TileStrategy = FlattenedTileStrategy(self, config)
        else:
            self.tile_strategy: TileStrategy = NDTileStrategy(self, config)
        self.grid_expr: ast.AST | None = None

    def set_grid_expr(self, grid_expr: ast.AST) -> None:
        assert self.grid_expr is None, "grid_expr already set"
        self.grid_expr = grid_expr

    def sympy_expr(self, expr: sympy.Expr) -> str:
        symbol_to_origin = HostFunction.current().symbol_to_origin
        expr = CompileEnvironment.current().shape_env.simplify(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            assert sym.name in symbol_to_origin, f"no origin found for {sym.name}"
            origin = symbol_to_origin[sym.name]
            if isinstance(origin.origin, TensorSizeOrigin):
                assert origin.fake_value is not None
                arg = self.tensor_size(
                    origin.fake_value,
                    origin.origin.key,
                )
                replacements[sym] = sympy.Symbol(arg.name, integer=True)
            elif isinstance(origin.origin, BlockSizeOrigin):
                result = self.tile_strategy.block_size_var(origin.origin.block_size_idx)
                assert result is not None
                replacements[sym] = sympy.Symbol(result, integer=True)
            else:
                replacements[sym] = sympy.Symbol(
                    self.symbol_arg(sym, origin.origin).name, integer=True
                )
        return texpr(expr.xreplace(replacements))

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        return repr(expr)

    def unique_name(self, prefix: str) -> str:
        return self.new_var(f"{prefix}_{next(self._unique_counter[prefix])}")

    def new_var(self, name: str) -> str:
        # TODO(jansel): Check for conflicts with user-defined names and rename if needed
        return f"_{name}"

    def tensor_arg(
        self, fake_value: torch.Tensor, prefer_name: str | None = None
    ) -> TensorArg:
        if fake_value not in self._tensor_args:
            origin = HostFunction.current().tensor_to_origin[fake_value]
            arg = TensorArg(
                self.new_var(prefer_name or origin.suggest_var_name()),
                fake_value,
                origin.host_str(),
            )
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    def symbol_arg(self, sym: sympy.Symbol, origin: Origin) -> SymbolArgument:
        if sym.name not in self._symbol_args:
            arg = SymbolArgument(
                name=self.new_var(origin.suggest_var_name()),
                _host_str=origin.host_str(),
            )
            self.arguments.append(arg)
            self._symbol_args[sym.name] = arg
        return self._symbol_args[sym.name]

    def constexpr_arg(self, name: str, host_str: str | None = None) -> ConstExprArg:
        self.arguments.append(rv := ConstExprArg(name, host_str or name))
        return rv

    def _tensor_property(
        self,
        prop_cls: type[_P],
        fake_value: torch.Tensor,
        dim: int,
        prefix: str,
    ) -> _P:
        # TODO(jansel): dedupe based on sympy expressions
        key = (prop_cls, fake_value, dim)
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"{arg.name}_{prefix}_{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> TensorSizeArg:
        return self._tensor_property(TensorSizeArg, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> TensorStrideArg:
        return self._tensor_property(TensorStrideArg, fake_value, dim, "stride")

    def sorted_args(self) -> list[Argument]:
        self.arguments.sort(key=lambda arg: arg.sort_key())
        return self.arguments

    def codegen_function_def(self) -> ast.FunctionDef:
        return create(
            ast.FunctionDef,
            name=self.name,
            args=create_arguments([arg.arg_def_node() for arg in self.sorted_args()]),
            body=self.body,
            decorator_list=[expr_from_string("triton.jit")],
            type_params=[],
        )

    def codegen_function_call(self) -> ast.AST:
        args = [arg.host_str() for arg in self.sorted_args()]
        args.extend(
            [
                f"num_warps={self.config.num_warps}",
                f"num_stages={self.config.num_stages}",
            ]
        )
        grid_expr = self.grid_expr
        assert grid_expr is not None
        # TODO(jansel): we should run CSE this statement
        return statement_from_string(
            f"{self.name}[__call_grid_expr]({', '.join(args)})",
            __call_grid_expr=grid_expr,
        )
