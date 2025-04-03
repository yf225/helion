from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

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
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from .generate_ast import GenerateAST


def process_graph_in_type_propagation(gm: torch.fx.GraphModule) -> None:
    with compile_lock:
        graph_lowering = GraphLowering(
            gm, shape_env=CompileEnvironment.current().shape_env
        )
        # pyre-ignore[19]
        with V.set_graph_handler(graph_lowering):
            for node in gm.graph.nodes:
                assert node.op in {"call_function", "placeholder", "output"}, node.op
                if node.op == "call_function":
                    prior_buffers = len(graph_lowering.buffers)
                    node.meta["lowering"] = process_node_in_type_propagation(
                        graph_lowering, node
                    )
                    if len(graph_lowering.buffers) > prior_buffers + 1:
                        raise InductorLoweringError(
                            f"Lowering {node.op} resulted in {len(graph_lowering.buffers) - prior_buffers} buffers, expected 1."
                        )


def process_node_in_type_propagation(
    graph_lowering: GraphLowering,
    node: Node,
) -> InductorLowering:
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


@dataclasses.dataclass
class InductorLowering:
    buffer: ComputedBuffer
    input_names: list[str]

    def codegen(self, cg: GenerateAST, input_asts: list[ast.AST]) -> ast.AST:
        raise NotImplementedError(
            f"codegen not implemented for {type(self).__name__}: {self.buffer}"
        )


class PointwiseLowering(InductorLowering):
    def codegen(self, cg: GenerateAST, input_asts: list[ast.AST]) -> ast.AST:
        # pyre-ignore[19]
        with V.set_ops_handler(
            GenerateASTFromInductor(cg, dict(zip(self.input_names, input_asts)))
        ):
            indices = [
                sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
            ]
            output_name = _unpack_opsvalue(self.buffer.data.inner_fn(indices))
            return expr_from_string(output_name)


class ReductionLowering(InductorLowering):
    pass


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
                input_asts: list[ast.AST] = []
                map_arg(
                    (n.args, n.kwargs),
                    lambda arg: input_asts.append(self.env[arg]),
                )
                lowering: InductorLowering = n.meta["lowering"]
                assert len(input_asts) == len(lowering.input_names)
                return lowering.codegen(self.cg, input_asts)
        return super().run_node(n)


def codegen_call_with_graph(
    cg: GenerateAST, gm: torch.fx.GraphModule, args: list[ast.AST]
) -> ast.AST:
    with compile_lock:
        result = GraphInterpreter(gm, cg).run(*args)
        assert isinstance(result, ast.AST)
        return result
