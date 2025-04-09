from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import Pointwise
from torch._inductor.ir import Reduction
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
from torch.fx.interpreter import Interpreter
from torch.fx.node import Node
from torch.fx.node import map_arg

from ..exc import InductorLoweringError
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from .. import Config
    from .device_function import DeviceFunction
    from .generate_ast import GenerateAST
    from .tile_strategy import TileStrategy


def prepare_graph_lowerings(gm: torch.fx.GraphModule) -> None:
    with compile_lock:
        graph_lowering = GraphLowering(
            gm, shape_env=CompileEnvironment.current().shape_env
        )
        # pyre-ignore[19]
        with V.set_graph_handler(graph_lowering):
            for node in gm.graph.nodes:
                assert node.op in {
                    "call_function",
                    "placeholder",
                    "output",
                }, node.op
                if node.op == "call_function":
                    prior_buffers = len(graph_lowering.buffers)
                    node.meta["lowering"] = prepare_node_lowering(graph_lowering, node)
                    if len(graph_lowering.buffers) > prior_buffers + 1:
                        raise InductorLoweringError(
                            f"Lowering {node.op} resulted in {len(graph_lowering.buffers) - prior_buffers} buffers, expected 1."
                        )


def prepare_node_lowering(
    graph_lowering: GraphLowering,
    node: Node,
) -> Lowering:
    if is_api_func(api := node.target):
        APIFuncLowering.normalize_args_kwargs(api, node)
        return APIFuncLowering(api)

    def convert_arg(arg: Node) -> TensorBox:
        example = arg.meta["val"]
        input_names.append(name := f"{node.name}_input{len(input_names)}")
        return TensorBox.create(
            InputBuffer(
                name=name,
                layout=FixedLayout(
                    example.device,
                    example.dtype,
                    [*map(_unpack_symint, example.size())],
                    [*map(_unpack_symint, example.stride())],
                ),
            )
        )

    input_names: list[str] = []
    result = graph_lowering.call_function(
        # pyre-ignore[6]
        node.target,
        # pyre-ignore[6]
        *map_arg((node.args, node.kwargs), convert_arg),
    )
    result.realize()
    if not isinstance(result, TensorBox) or not isinstance(result.data, StorageBox):
        raise InductorLoweringError(
            f"Lowering {node.target} returned type(result), expected TensorBox(StorageBox(...)): {result}"
        )
    if not isinstance(buffer := result.data.data, ComputedBuffer):
        raise InductorLoweringError(
            f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer: {buffer}"
        )
    if isinstance(buffer.data, Pointwise):
        return PointwiseLowering(buffer, input_names)
    if isinstance(buffer.data, Reduction):
        return ReductionLowering(buffer, input_names)
    raise InductorLoweringError(
        f"Lowering {node.target} returned buffer type {type(buffer.data)}, expected Pointwise or Reduction: {buffer}"
    )


def _unpack_symint(x: torch.SymInt | int) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return x._sympy_()
    if isinstance(x, int):
        return sympy.sympify(x)
    raise TypeError(f"Expected SymInt or int, got {type(x)}")


class Lowering:
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
        raise NotImplementedError


@dataclasses.dataclass
class InductorLowering(Lowering):
    buffer: ComputedBuffer
    input_names: list[str]

    def input_asts(self, ctx: GraphInterpreter, node: torch.fx.Node) -> list[ast.AST]:
        input_asts: list[ast.AST] = []
        map_arg(
            (node.args, node.kwargs),
            lambda arg: input_asts.append(ctx.env[arg]),
        )
        assert len(input_asts) == len(self.input_names)
        return input_asts

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
        raise NotImplementedError(
            f"codegen not implemented for {type(self).__name__}: {self.buffer}"
        )


class PointwiseLowering(InductorLowering):
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
        # pyre-ignore[19]
        with V.set_ops_handler(
            GenerateASTFromInductor(
                ctx.cg, dict(zip(self.input_names, self.input_asts(ctx, node)))
            ),
        ):
            indices = [
                sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
            ]
            output_name = _unpack_opsvalue(self.buffer.data.inner_fn(indices))
            return expr_from_string(output_name)


class ReductionLowering(InductorLowering):
    pass


@dataclasses.dataclass
class APIFuncLowering(Lowering):
    api_func: APIFunc

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
        assert not node.kwargs
        ast_args = [*map_arg(node.args, lambda arg: ctx.env[arg])]
        proxy_args = [*map_arg(node.args, lambda arg: arg.meta["val"])]

        assert self.api_func._codegen is not None
        return self.api_func._codegen(
            CodegenState(
                ctx.cg,
                fx_node=node,
                # pyre-ignore[6]
                proxy_args=proxy_args,
                # pyre-ignore[6]
                ast_args=ast_args,
            ),
        )

    @staticmethod
    def normalize_args_kwargs(
        api_func: APIFunc,
        node: torch.fx.Node,
    ) -> None:
        bound = api_func._signature.bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        node.args = (*bound.arguments.values(),)
        node.kwargs = {}


class GenerateASTFromInductor(DefaultHandler):
    def __init__(self, cg: GenerateAST, input_name_lookup: dict[str, ast.AST]) -> None:
        super().__init__()
        self.parent_handler = TritonOverrides()
        self.cg = cg
        self.input_name_lookup = input_name_lookup

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> str:
        result_str = _unpack_opsvalue(
            getattr(self.parent_handler, name)(*args, **kwargs)
        )
        return self.cg.lift(expr_from_string(result_str)).id

    def load(self, name: str, index: sympy.Expr) -> str:
        # TODO(jansel): assert the index is correct
        return self.cg.lift(self.input_name_lookup[name]).id


def _unpack_opsvalue(value: object) -> str:
    if isinstance(value, OpsValue):
        return str(value)
    assert isinstance(value, str)
    return value


class GraphInterpreter(Interpreter):
    def __init__(self, gm: torch.fx.GraphModule, cg: GenerateAST) -> None:
        super().__init__(gm, garbage_collect_values=False)
        self.cg = cg

    def run_node(self, n: Node) -> object:
        if n.op == "call_function":
            with self._set_current_node(n):
                lowering: Lowering = n.meta["lowering"]
                result = lowering.codegen(self, n)
                if not isinstance(result, ast.AST):
                    assert isinstance(
                        result, (int, torch.SymInt, torch.SymFloat, torch.SymBool)
                    )
                    return result
                assert isinstance(result, ast.expr)
                if len(n.users) > 0:
                    if isinstance(result, (ast.Name, ast.Constant)):
                        return result
                    name = self.cg.device_function.new_var(n.name)
                    self.cg.add_statement(
                        statement_from_string(f"{name} = result", result=result)
                    )
                    return create(ast.Name, id=name, ctx=ast.Load())
                if not isinstance(result, (ast.Name, ast.Constant)):
                    self.cg.add_statement(create(ast.Expr, value=result))
                return None
        return super().run_node(n)


def codegen_call_with_graph(
    cg: GenerateAST, gm: torch.fx.GraphModule, args: list[ast.AST]
) -> None:
    with compile_lock:
        GraphInterpreter(gm, cg).run(*args)


class CodegenState(NamedTuple):
    codegen: GenerateAST
    fx_node: torch.fx.Node
    proxy_args: list[object]
    ast_args: list[object]

    def proxy_arg(self, i: int) -> object:
        return self.proxy_args[i]

    def ast_arg(self, i: int) -> ast.AST:
        rv = self.ast_args[i]
        assert isinstance(rv, ast.AST), "TODO: convert nested/defaults"
        return rv

    @property
    def fake_value(self) -> object:
        return self.fx_node.meta["val"]

    @property
    def device_function(self) -> DeviceFunction:
        return self.codegen.device_function

    @property
    def tile_strategy(self) -> TileStrategy:
        return self.codegen.device_function.tile_strategy

    @property
    def config(self) -> Config:
        return self.codegen.device_function.config

    def add_statement(self, statement: ast.stmt | str) -> None:
        return self.codegen.add_statement(statement)
