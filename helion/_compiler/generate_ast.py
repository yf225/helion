from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import NamedTuple

from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .inductor_lowering import codegen_call_with_graph

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..runtime import Config
    from .host_function import HostFunction
    from .type_propagation import TensorType


class GenerateAST(NodeVisitor):
    def __init__(self, func: HostFunction, config: Config) -> None:
        super().__init__()
        self.host_fn = func
        self.host_statements: list[ast.AST] = []
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.device_function = DeviceFunction(f"_{func.name}_kernel", config)

    def add_statement(self, stmt: ast.AST | str) -> None:
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        self.statements_stack[-1].append(stmt)

    def tmpvar(self) -> str:
        return self.device_function.unique_name("v")

    def lift(self, expr: ast.AST) -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST)
        with expr:
            varname = self.tmpvar()
            self.add_statement(statement_from_string(f"{varname} = expr", expr=expr))
            return create(ast.Name, id=varname, ctx=ast.Load())

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
        else:
            self.statements_stack.append(new_statements)
            try:
                yield
            finally:
                self.statements_stack.pop()

    @contextlib.contextmanager
    def set_on_device(self) -> Iterator[None]:
        assert self.on_device is False
        self.on_device = True
        prior = self.host_statements
        self.host_statements = self.statements_stack[-1]
        try:
            yield
        finally:
            self.on_device = False
            self.host_statements = prior

    def generic_visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        fields = {}
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                fields[field] = new_list = []
                with self.set_statements(
                    new_list
                    if old_value and isinstance(old_value[0], ast.stmt)
                    else None
                ):
                    for item in old_value:
                        new_list.append(self.visit(item))  # noqa: PERF401 # mutation in visit
            elif isinstance(old_value, ast.AST):
                fields[field] = self.visit(old_value)
            else:
                fields[field] = old_value
        return node.new(fields)

    def visit_For(self, node: ast.For) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse
            with (
                self.set_on_device(),
                self.set_statements(self.device_function.body),
            ):
                iter_node = node.iter
                assert isinstance(iter_node, ExtendedAST)
                with iter_node:
                    assert isinstance(iter_node, ast.Call)
                    assert not iter_node.keywords
                    assert len(iter_node.args) == 1
                    fn_node = iter_node.func
                    assert isinstance(fn_node, ExtendedAST)
                    arg_node = iter_node.args[0]
                    assert isinstance(arg_node, ExtendedAST)
                    fn = fn_node._type_info.proxy()
                    arg = arg_node._type_info.proxy()
                    assert is_api_func(fn)
                    assert fn._codegen is not None
                    from .inductor_lowering import CodegenState

                    fn._codegen(
                        CodegenState(
                            self,
                            fx_node=None,
                            proxy_args=[arg],
                            ast_args=None,
                        ),
                    )
                codegen_call_with_graph(self, self.host_fn.device_ir.root, [])
            self.device_function.dead_code_elimination()
            return self.device_function.codegen_function_call()
        return self.generic_visit(node)


class TensorReference(NamedTuple):
    node: ast.AST
    name: str
    type_info: TensorType

    @property
    def is_host(self) -> bool:
        return self.type_info.origin.is_host()


class SubscriptIndexing(NamedTuple):
    tensor_ref: TensorReference
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )


def generate_ast(func: HostFunction, config: Config) -> ast.AST:
    with func:
        codegen = GenerateAST(func, config)
        with codegen.device_function:
            for stmt in func.body:
                codegen.add_statement(codegen.visit(stmt))
            CompileEnvironment.current().errors.raise_if_errors()
            return ast.Module(
                [
                    codegen.device_function.codegen_function_def(),
                    func.codegen_function_def(codegen.host_statements),
                ],
                [],
            )
