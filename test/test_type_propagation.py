from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import unittest

from expecttest import TestCase
import torch

from helion._testing import import_path

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_types()


class TestTypePropagation(TestCase):
    maxDiff = 16384

    def test_add(self):
        output = type_propagation_report(
            basic_kernels.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def add(x, y):
    # Call: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32))) SourceOrigin(location=<SourceLocation basic_kernels.py:8>)
    # Attribute: CallableType(broadcast_tensors) AttributeOrigin(value=GlobalOrigin(name='torch'), key='broadcast_tensors')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    # Attribute:     %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    # Attribute:     return (arg0_1, arg1_1)
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    x, y = torch.broadcast_tensors(x, y)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:9>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    # Attribute:     %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    # Attribute:     %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 1), kwargs = {})
    # Attribute:     %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%sym_size_int, %sym_size_int_1],), kwargs = {dtype: torch.int32, layout: torch.strided, device: cpu, pin_memory: False})
    # Attribute:     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 1]), kwargs = {})
    # Attribute:     return permute
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:8>), key=0)
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:9>), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:9>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:11>)
        # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:9>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
        # BinOp: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:11>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:11>)
        # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:8>), key=0)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:11>)
        # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:8>), key=1)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
        out[tile] = x[tile] + y[tile]
    return out""",
        )

    def test_torch_ops_pointwise(self):
        output = type_propagation_report(
            basic_kernels.torch_ops_pointwise,
            torch.ones([1024], dtype=torch.int32),
            torch.ones([1024], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def torch_ops_pointwise(x, y):
    # Call: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:17>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    # Attribute:     %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    # Attribute:     %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%sym_size_int],), kwargs = {dtype: torch.int32, layout: torch.strided, device: cpu, pin_memory: False})
    # Attribute:     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0]), kwargs = {})
    # Attribute:     return permute
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0)])) SourceOrigin(location=<SourceLocation basic_kernels.py:18>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s77), )) SourceOrigin(location=<SourceLocation basic_kernels.py:18>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:17>), key='size')
    # Name: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:17>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Name: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:17>)
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:18>)
        # Call: TensorType([block_size0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Attribute: CallableType(_VariableFunctionsClass.sigmoid) AttributeOrigin(value=GlobalOrigin(name='torch'), key='sigmoid')
        # Attribute: graph():
        # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        # Attribute:     %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg0_1,), kwargs = {})
        # Attribute:     return sigmoid
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Call: TensorType([block_size0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Attribute: CallableType(_VariableFunctionsClass.add) AttributeOrigin(value=GlobalOrigin(name='torch'), key='add')
        # Attribute: graph():
        # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        # Attribute:     %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
        # Attribute:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
        # Attribute:     return add
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Call: TensorType([block_size0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Attribute: CallableType(_VariableFunctionsClass.sin) AttributeOrigin(value=GlobalOrigin(name='torch'), key='sin')
        # Attribute: graph():
        # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        # Attribute:     %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
        # Attribute:     return sin
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Subscript: TensorType([block_size0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Name: TensorType([x_size0], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:18>)
        # Call: TensorType([block_size0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Attribute: CallableType(_VariableFunctionsClass.cos) AttributeOrigin(value=GlobalOrigin(name='torch'), key='cos')
        # Attribute: graph():
        # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        # Attribute:     %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%arg0_1,), kwargs = {})
        # Attribute:     return cos
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Subscript: TensorType([block_size0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Name: TensorType([y_size0], torch.int32) ArgumentOrigin(name='y')
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:18>)
        out[tile] = torch.sigmoid(torch.add(torch.sin(x[tile]), torch.cos(y[tile])))
    return out""",
        )

    def test_all_ast_nodes(self):
        output = type_propagation_report(
            import_path(datadir / "all_ast_nodes.py").all_ast_nodes,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
            ignore=True,
        )
        self.assertExpectedInline(
            output,
            """\
def all_ast_nodes(x, y):
    # Constant: LiteralType(1024) SourceOrigin(location=<SourceLocation all_ast_nodes.py:23>)
    int_literal = 1024
    # JoinedStr: UnsupportedType('str is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:24>)
    formatted_value = f'prefix{int_literal}suffix'
    # Constant: LiteralType('abc') SourceOrigin(location=<SourceLocation all_ast_nodes.py:25>)
    joined_string = 'abc'
    # List: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: LiteralType(1024) SourceOrigin(location=<SourceLocation all_ast_nodes.py:23>)
    list_literal0 = [x, y, int_literal]
    # Tuple: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    tuple_literal0 = (x, y, 1, 2)
    # List: SequenceType([LiteralType(5), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Name: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    list_literal1 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # List: SequenceType([LiteralType(5), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Name: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    tuple_literal2 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # Set: UnsupportedType('set is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    set_literal = {1, 2, 3}
    # Dict: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:31>)
    dict_literal0 = {}
    # Name: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:31>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:32>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:32>)
    dict_literal0[1] = 2
    # Dict: DictType({1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType('y') SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    dict_literal1 = {1: x, 'y': y}
    # Dict: DictType({'foo': LiteralType('bar'), 1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Constant: LiteralType('foo') SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Constant: LiteralType('bar') SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Name: DictType({1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    dict_literal2 = {'foo': 'bar', **dict_literal1}
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:35>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    uadd = +x
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:36>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    usub = -x
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:37>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    invert = ~x
    # UnaryOp: SymBoolType(Eq(u0, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:38>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    not_ = not x
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:39>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add = x + y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:40>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    sub = x - y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:41>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    mul = x * y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:43>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    truediv = x / y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:44>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    floordiv = x // y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:45>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    mod = x % y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:46>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    pow = x ** y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:47>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    lshift = x << y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:48>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    rshift = x >> y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:49>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_and = x & y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:50>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_xor = x ^ y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:51>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_or = x | y
    # BoolOp: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:52>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    and_ = x and y
    # BoolOp: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:53>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    or_ = x and y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:54>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    eq = x == y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:55>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ne = x != y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:56>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    lt = x < y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:57>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    le = x <= y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:58>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    gt = x > y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:59>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ge = x >= y
    # Compare: SymBoolType(Eq(u1, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:60>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    is_ = x is y
    # Compare: SymBoolType(Eq(u2, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:61>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    is_not = x is not y
    # Compare: SymBoolType(Eq(u3, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:62>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    in_ = x in y
    # Compare: SymBoolType(Eq(u4, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:63>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    not_in = x not in y
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:64>)
    # Name: CallableType(func) GlobalOrigin(name='func')
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:64>)
    call0 = func(x, y, 3)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:65>)
    # Name: CallableType(func) GlobalOrigin(name='func')
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:65>)
    call1 = func(x, y, c=3)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:66>)
    # Name: CallableType(func) GlobalOrigin(name='func')
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Tuple: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:66>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    call2 = func(*(x, y, y))
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: CallableType(func) GlobalOrigin(name='func')
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Dict: DictType({'b': TensorType([y_size0, x_size1], torch.int32), 'c': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Constant: LiteralType('b') SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType('c') SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    call3 = func(x, **{'b': y, 'c': y})
    # IfExp: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:53>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ifexp = x if or_ else y
    # Attribute: LiteralType(torch.int32) AttributeOrigin(value=ArgumentOrigin(name='x'), key='dtype')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr0 = x.dtype
    # Attribute: SequenceType((SymIntType(s17), SymIntType(s27))) AttributeOrigin(value=ArgumentOrigin(name='x'), key='shape')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr1 = x.shape
    # Call: SymIntType(s17) SourceOrigin(location=<SourceLocation all_ast_nodes.py:75>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:75>)
    attr2 = x.size(0)
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:76>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr3 = x.size()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:77>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='stride')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr4 = x.stride()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:78>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='stride')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:78>)
    attr5 = x.stride(0)
    # NamedExpr: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    named_expr = (z := (y + 1))
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    zzz = zz = z - 1
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:81>)
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:81>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    q = zzz + zz + z
    # Subscript: UnknownType('Subscript not supported with self=SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) key=LiteralType(0)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:82>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:82>)
    subscript0 = list_literal0[0]
    # Subscript: SequenceType([TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Slice: SliceType(LiteralType(1):LiteralType(None):LiteralType(None)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    subscript1 = list_literal0[1:]
    # Subscript: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Slice: SliceType(LiteralType(None):LiteralType(-1):LiteralType(None)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # UnaryOp: LiteralType(-1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    subscript2 = list_literal0[:-1]
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:39>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add += y
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:39>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add -= y
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    a, b, c = [1, 2, 3]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:88>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Name: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Name: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    tmp0 = [a, b, c]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    a, *bc = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4), LiteralType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:90>)
    # Name: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    tmp1 = [*bc, a]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    a, *ab, c = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(1), LiteralType(4), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:92>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Name: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Name: SequenceType([LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    tmp2 = [a, c, *ab]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Constant: LiteralType(6) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    a, *ab, c = [5, 6]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:94>)
    # Name: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Name: LiteralType(6) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Name: SequenceType([]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    tmp2 = [a, c, *ab]
    try:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:97>)
        e0 = 1
        # Call: UnknownType('Exception is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:98>)
        # Name: CallableType(Exception) BuiltinOrigin(name='Exception')
        raise Exception()
    except Exception as e:
        e1 = 1
    else:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:102>)
        e2 = 1
        pass
    # Compare: SymBoolType(Eq(u5, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:105>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    assert x is not y
    # Compare: SymBoolType(Eq(u6, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:106>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType('msg') SourceOrigin(location=<SourceLocation all_ast_nodes.py:106>)
    assert x is not y, 'msg'
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:39>)
    del add
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:109>)
    import torch
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:110>)
    from torch import Tensor
    # Compare: SymBoolType(Eq(u7, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:112>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    if x is y:
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        join_var0 = x
        # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:114>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        join_var1 = x + y
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:115>)
        join_var2 = 1
    else:
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        join_var0 = y
        # Constant: LiteralType(None) SourceOrigin(location=<SourceLocation all_ast_nodes.py:118>)
        join_var1 = None
        # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:119>)
        join_var2 = 2
    # List: SequenceType([TensorType([y_size0, x_size1], torch.int32), UnknownType("Can't combine types from control flow: TensorType([y_size0, x_size1], torch.int32) and LiteralType(None)"), SymIntType(u8)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:120>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: UnknownType("Can't combine types from control flow: TensorType([y_size0, x_size1], torch.int32) and LiteralType(None)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:118>)
    # Name: SymIntType(u8) SourceOrigin(location=<SourceLocation all_ast_nodes.py:115>)
    combined = [join_var0, join_var1, join_var2]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:122>)
    v = 0
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:123>)
    # For: loop_type=HOST
    z = 0
    # Call: LiteralType(range(0, 3)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
    # Name: CallableType(range) BuiltinOrigin(name='range')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
    for i in range(3):
        # BinOp: SymIntType(u12) SourceOrigin(location=<SourceLocation all_ast_nodes.py:125>)
        # Name: SymIntType(u11) SourceOrigin(location=<SourceLocation all_ast_nodes.py:122>)
        # Name: SymIntType(u10) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
        v = v + i
        # BinOp: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
        # Name: UnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        z = z + x
        break
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:129>)
        t = 0
    # List: SequenceType([SymIntType(u13), ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)")]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:130>)
    # Name: SymIntType(u13) SourceOrigin(location=<SourceLocation all_ast_nodes.py:125>)
    # Name: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
    combined = [v, z]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
    i = 0
    # Compare: SymBoolType(Eq(u16, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:133>)
    # Name: SymIntType(u14) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:133>)
    while i < 3:
        # BinOp: SymIntType(u18) SourceOrigin(location=<SourceLocation all_ast_nodes.py:134>)
        # Name: SymIntType(u17) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:134>)
        i = i + 1
        continue
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:137>)
        t = 0
    with contextlib.nullcontext():
    # Global: UnknownType('ast.Global is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:142>)
        e3 = 1
    global global0
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    # Attribute:     %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    # Attribute:     %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 1), kwargs = {})
    # Attribute:     %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%sym_size_int, %sym_size_int_1],), kwargs = {dtype: torch.int32, layout: torch.strided, device: cpu, pin_memory: False})
    # Attribute:     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 1]), kwargs = {})
    # Attribute:     return permute
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        # BinOp: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        out[tile] = x[tile] + y[tile]
    return out""",
        )


if __name__ == "__main__":
    unittest.main()
