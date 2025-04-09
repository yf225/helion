from __future__ import annotations

import ast
import re
import textwrap
from typing import Callable
from unittest.mock import patch

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental import proxy_tensor

from .. import exc
from .. import language as hl
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .inductor_lowering import prepare_graph_lowerings
from .tracing_ops import _get_symnode
from .tracing_ops import _host_tensor
from .type_propagation import IterType
from .type_propagation import SequenceType
from .type_propagation import TensorType
from .type_propagation import _eval_binary
from .type_propagation import _eval_unary


def _make_fx(fn: Callable[..., object], *args: object) -> torch.fx.GraphModule:
    """
    We monkey patch get_proxy_slot to support SymInt/SymFloat/SymBool
    in the graph without any origin for them.  We instead insert
    symnode_dummy_origin() in the graph which are ignored in codegen.
    """

    def _get_proxy_slot(
        obj: object,
        tracer: proxy_tensor.PythonKeyTracer,
        default: object = proxy_tensor.no_default,
        transform: Callable[[object], object] = lambda x: x,
    ) -> object:
        if isinstance(obj, torch.Tensor):
            tracker = tracer.tensor_tracker
            if obj not in tracker:
                origin = HostFunction.current().tensor_to_origin[obj]
                assert origin.is_host()
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _host_tensor,
                    (origin.host_str(),),
                    {},
                    name=origin.suggest_var_name(),
                )
                proxy.node.meta["val"] = obj
            return transform(tracker[obj])
        if isinstance(obj, proxy_tensor.py_sym_types):
            tracker = tracer.symnode_tracker
            if obj not in tracker:
                debug_name = CompileEnvironment.current().sympy_debug(obj._sympy_())
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _get_symnode,
                    (debug_name,),
                    {},
                    name=debug_name if debug_name.isidentifier() else "symnode",
                )
                proxy.node.meta["val"] = obj
                proxy.force = lambda: proxy
            return transform(tracker[obj])
        return get_proxy_slot(obj, tracer, default, transform)

    get_proxy_slot: Callable[..., object] = proxy_tensor.get_proxy_slot
    with patch.object(proxy_tensor, "get_proxy_slot", _get_proxy_slot):
        return proxy_tensor.make_fx(fn, decomposition_table=select_decomp_table())(
            *args
        )


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[torch.fx.GraphModule] = []
        self.root: torch.fx.GraphModule | None = None

    def __str__(self) -> str:
        assert len(self.graphs) == 1
        output = self.graphs[0].print_readable(print_output=False).strip()
        return textwrap.dedent(
            re.sub("^[^\n]+\n", "", output).replace("forward(self)", "device_ir()")
        )

    def add_graph(self, graph: torch.fx.GraphModule) -> None:
        self.graphs.append(graph)

    def add_root_graph(self, graph: torch.fx.GraphModule) -> None:
        assert self.root is None
        self.root = graph
        self.add_graph(graph)


class WalkDeviceAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.scope: dict[str, object] = {}

    def generic_visit(self, node: ast.AST) -> None:
        raise exc.StatementNotSupported(type(node).__name__)

    def _assign(self, target: ast.AST, value: object) -> None:
        if isinstance(target, ast.Name):
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, n in enumerate(target.elts):
                if isinstance(n, ast.Starred):
                    raise exc.StarredArgsNotSupportedOnDevice
                # pyre-ignore[16]
                self._assign(n, value[i])
        else:
            raise NotImplementedError(
                f"Unsupported target type {type(target).__name__}"
            )

    def _body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.visit(stmt)

    def _to_proxy(self, node: ast.AST) -> object:
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        if not type_info.contains_tensor():
            return type_info.proxy()
        return self.visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> object:
        return _eval_binary(node.op, self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        return _eval_unary(node.op, self.visit(node.operand))

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse
            assert isinstance(node.iter, ExtendedAST)
            iter_type = node.iter._type_info
            assert isinstance(iter_type, IterType)
            self._assign(node.target, iter_type.inner.proxy())
            self._body(node.body)
        elif node._loop_type == LoopType.DEVICE:
            raise NotImplementedError
        else:
            raise AssertionError(f"Unexpected loop type {node._loop_type}")

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in self.scope:
            return self.scope[node.id]
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        assert type_info.origin.is_host()
        try:
            return type_info.proxy()
        except NotImplementedError:
            raise exc.CantReadOnDevice(type_info) from None

    def _subscript_slice_proxy(self, slice_node: ast.AST) -> list[object]:
        assert isinstance(slice_node, ExtendedAST)
        key = slice_node._type_info
        if isinstance(key, SequenceType):
            keys = key.unpack()
        else:
            keys = [key]
        try:
            return [x.proxy() for x in keys]
        except TypeError:
            raise exc.InvalidSliceType(slice_node._type_info) from None

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise exc.AssignmentMultipleTargets
        (target,) = node.targets
        if isinstance(target, ast.Name):
            # TODO(jansel): should assert that name is only used on device
            self._assign(target, self.visit(node.value))
            return None
        if not isinstance(target, ast.Subscript):
            raise exc.InvalidAssignment
        assert isinstance(node.value, ExtendedAST)
        rhs_type = node.value._type_info
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info
        if not isinstance(lhs_type, TensorType) or not isinstance(rhs_type, TensorType):
            raise exc.NonTensorSubscriptAssign(lhs_type, rhs_type)
        assert isinstance(target.value, ExtendedAST)
        target_origin = target.value._type_info.origin
        assert target_origin.is_host()
        val = self.visit(node.value)
        return hl.store(
            self.visit(target.value), self._subscript_slice_proxy(target.slice), val
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(
                create(
                    ast.Assign,
                    targets=[node.target],
                    value=node.value,
                )
            )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        assert isinstance(node.target, ExtendedAST)
        self._assign(
            node.target,
            _eval_binary(node.op, self.visit(node.target), self.visit(node.value)),
        )

    def visit_Subscript(self, node: ast.Subscript) -> object:
        value = node.value
        assert isinstance(value, ExtendedAST)
        type_info = value._type_info
        if type_info.origin.is_host():
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        return hl.subscript(self.visit(value), self._subscript_slice_proxy(node.slice))

    def visit_Call(self, node: ast.Call) -> object:
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # pyre-ignore[6]
                args.extend(self._to_proxy(arg.value))
            else:
                args.append(self._to_proxy(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                # pyre-ignore[6]
                kwargs.update(self._to_proxy(kwarg.value))
            else:
                kwargs[kwarg.arg] = self._to_proxy(kwarg.value)
        # pyre-ignore[29]
        return self.visit(node.func)(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> object:
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        try:
            return type_info.proxy()
        except NotImplementedError:
            raise exc.CantReadOnDevice(type_info) from None


class WalkHostAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            self.device_ir.add_root_graph(
                _make_fx(lambda: WalkDeviceAST(self.device_ir).visit(node))
            )
        else:
            self.generic_visit(node)


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    with func, compile_lock:
        device_ir = DeviceIR()
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        CompileEnvironment.current().errors.raise_if_errors()
        for graph in device_ir.graphs:
            prepare_graph_lowerings(graph)
        return device_ir
