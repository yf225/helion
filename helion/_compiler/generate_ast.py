from __future__ import annotations

import ast
import contextlib
import functools
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple

from .. import exc
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .inductor_lowering import codegen_call_with_graph
from .type_propagation import CallableType
from .type_propagation import SequenceType
from .type_propagation import TensorType
from .type_propagation import TileIndexType
from .type_propagation import TypeInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

    from ..runtime import Config
    from .host_function import HostFunction
    from .tile_strategy import TileStrategy


OUTPUT_CODE_HEADER = """\
import torch
import triton
from triton import language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

"""


class CodegenState(NamedTuple):
    codegen: GenerateAST
    type_info: TypeInfo | None = None
    ast_node: ast.AST | None = None
    fx_node: torch.fx.Node | None = None
    proxy_args: list[object] | None = None

    def proxy_arg(self, i: int) -> object:
        assert self.proxy_args is not None
        return self.proxy_args[i]

    @property
    def fake_value(self) -> object:
        if self.fx_node is not None:
            return self.fx_node.meta["val"]
        assert self.type_info is not None
        return self.type_info.proxy()

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


def device_only(
    fn: Callable[..., ast.AST],
) -> Callable[..., ast.AST]:
    @functools.wraps(fn)
    def wrapper(self: GenerateAST, node: ast.AST) -> ast.AST:
        if not self.on_device:
            return self.generic_visit(node)
        return fn(self, node)

    return wrapper


@device_only
def _not_supported_on_device(self: GenerateAST, node: ast.AST) -> ast.AST:
    raise exc.NotAllowedOnDevice(type(node).__name__)


class GenerateAST(ast.NodeVisitor):
    def __init__(self, func: HostFunction, config: Config) -> None:
        super().__init__()
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

    def visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        with node:
            try:
                visitor = getattr(
                    self,
                    f"visit_{node.__class__.__name__}",
                    self.generic_visit,
                )
                return visitor(node)
            except exc.Base:
                raise
            except Exception as e:
                raise exc.InternalError(e) from e

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
                self.set_statements(st := self.device_function.body),
            ):
                self.visit(node.iter)
                for stmt in node.body:
                    st.append(self.visit(stmt))
            return self.device_function.codegen_function_call()
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        assert isinstance(node.func, ExtendedAST)
        if isinstance(fn_type := node.func._type_info, CallableType):
            if is_api_func(api := fn_type.value) and api._codegen is not None:
                return api._codegen(
                    CodegenState(
                        self,
                        ast_node=node,
                        type_info=node._type_info,
                        proxy_args=self._get_proxy_args(api, node),
                    )
                )
            if self.on_device and node._graph is not None:
                tensor_args: list[ast.AST] = []
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        with arg:
                            raise exc.StarredArgsNotSupportedOnDevice
                    arg_ast = self.visit(arg)
                    if arg._type_info.contains_tensor():
                        assert isinstance(arg._type_info, TensorType), (
                            "TODO: tensor in collection"
                        )
                        tensor_args.append(self.lift(arg_ast))
                for kwarg in node.keywords:
                    if kwarg.arg is None:
                        with kwarg.value:
                            raise exc.StarredArgsNotSupportedOnDevice
                    arg = kwarg.value
                    arg_ast = self.visit(arg)
                    if arg._type_info.contains_tensor():
                        assert isinstance(arg._type_info, TensorType), (
                            "TODO: tensor in collection"
                        )
                        tensor_args.append(self.lift(arg_ast))
                return codegen_call_with_graph(self, node._graph, tensor_args)
        return self.generic_visit(node)

    def _get_proxy_args(self, api: APIFunc, node: ast.Call) -> list[object] | None:
        args = []
        kwargs = {}
        try:
            for arg in node.args:
                assert isinstance(arg, ExtendedAST)
                if isinstance(arg, ast.Starred):
                    with arg:
                        raise exc.StarredArgsNotSupportedOnDevice
                args.append(arg._type_info.proxy())
            for kwarg in node.keywords:
                assert isinstance(kwarg.value, ExtendedAST)
                if kwarg.arg is None:
                    with kwarg.value:
                        raise exc.StarredArgsNotSupportedOnDevice
                kwargs[kwarg.arg] = kwarg.value._type_info.proxy()
        except NotImplementedError:
            return None
        bound = api._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return [*bound.arguments.values()]

    @device_only
    def visit_Name(self, node: ast.Name) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        if (
            self.on_device
            and isinstance(node.ctx, ast.Load)
            and type_info.origin.is_host()
        ):
            if isinstance(type_info, TensorType):
                return self.tensor_reference(node).node
        return self.generic_visit(node)

    visit_AnnAssign = _not_supported_on_device

    @device_only
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if len(node.targets) != 1:
            raise exc.AssignmentMultipleTargets
        (target,) = node.targets
        if isinstance(target, ast.Name):
            # TODO(jansel): should assert that name is only used on device
            return self.generic_visit(node)
        if not isinstance(target, ast.Subscript):
            raise exc.InvalidAssignment
        assert isinstance(node.value, ExtendedAST)
        rhs_type = node.value._type_info
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info
        if not isinstance(lhs_type, TensorType) or not isinstance(rhs_type, TensorType):
            raise exc.NonTensorSubscriptAssign(lhs_type, rhs_type)
        if rhs_type.fake_value.size() != lhs_type.fake_value.size():
            raise exc.ShapeMismatch(lhs_type, rhs_type)
        value = self.visit(node.value)
        indexing = self.subscript_indexing(target)
        return statement_from_string(
            "tl.store(name + offset, value, mask)",
            value=self.lift(value),
            name=indexing.tensor_ref.node,
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )

    @device_only
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # TODO(jansel): non tensor types?
        # TODO(jansel): de-duplicate redundant loads
        indexing = self.subscript_indexing(node)
        return self.lift(
            expr_from_string(
                # TODO(jansel): optimize away mask/other?
                "tl.load(name + offset, mask, other=0)",
                name=indexing.tensor_ref.node,
                offset=indexing.index_expr,
                mask=indexing.mask_expr,
            )
        )

    def subscript_indexing(self, node: ast.Subscript) -> SubscriptIndexing:
        assert isinstance(node.value, ExtendedAST)
        with node.value:
            tensor_ref = self.tensor_reference(node.value)
        slice_node = node.slice
        assert isinstance(slice_node, ExtendedAST)
        key = slice_node._type_info
        if isinstance(key, SequenceType):
            keys = key.unpack()
        else:
            keys = [key]
        # TODO(jansel): need to rewrite this to traverse the slice
        index_values = []
        mask_values = {}
        for k in keys:
            if isinstance(k, TileIndexType):
                index_values.append(
                    self.device_function.tile_strategy.index_var(k.block_size_idx)
                )
                if mask := self.device_function.tile_strategy.mask_var(
                    k.block_size_idx
                ):
                    mask_values.setdefault(mask)
            else:
                raise exc.InvalidIndexingType(k)

        fake_value = tensor_ref.type_info.fake_value
        if len(index_values) != fake_value.ndim:
            raise exc.RankMismatch(fake_value.ndim, len(index_values))
        index_expr = []
        for i, idx in enumerate(index_values):
            if fake_value.size(i) != 1:
                stride = self.device_function.tensor_stride(fake_value, i).name
                index_expr.append(f"{idx} * {stride}")
        return SubscriptIndexing(
            tensor_ref,
            expr_from_string("+".join(index_expr)),
            expr_from_string("|".join(mask_values) or "None"),
        )

    def tensor_reference(self, node: ast.AST) -> TensorReference:
        assert isinstance(node, ExtendedAST)
        if not isinstance(node, ast.Name):
            raise exc.ExpectedTensorName(type(node.value).__name__)
        if not isinstance(type_info := node._type_info, TensorType):
            raise exc.ExpectedTensorName(node._type_info)
        if node._type_info.origin.is_host():
            name = self.device_function.tensor_arg(
                type_info.fake_value, prefer_name=node.id
            ).name
        else:
            name = node.id
        return TensorReference(
            create(ast.Name, id=name, ctx=ast.Load()), name, node._type_info
        )


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


def generate_ast(func: HostFunction, config: Config) -> ast.AST:
    with func:
        codegen = GenerateAST(func, config)
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
