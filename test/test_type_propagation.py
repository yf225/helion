from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
import torch
from torch._dynamo.source import LocalSource

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction
from helion._compiler.type_propagation import propagate_types
from helion._testing import import_path
from helion.exc import TensorOperationInWrapper

type_prop_inputs = import_path(Path(__file__).parent / "data/type_prop_inputs.py")


def type_propagation_report(fn, *args, ignore=False):
    with CompileEnvironment() as env:
        args = [env.to_fake(arg, LocalSource(f"arg{i}")) for i, arg in enumerate(args)]
        host_fn = HostFunction(fn, env)
        if ignore:
            host_fn.env.errors.ignore(TensorOperationInWrapper)
        propagate_types(host_fn, args, {})
        return host_fn.debug_types()


class TestTypePropagation(TestCase):
    def test_add(self):
        output = type_propagation_report(
            type_prop_inputs.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def add(x, y):
    # Call: SequenceType((TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:150>)
    # Attribute: CallableType(broadcast_tensors) AttributeOrigin(value=GlobalOrigin(name='torch', function=<HostFunction add>), key='broadcast_tensors')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    # Attribute:     %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    # Attribute:     return (arg0_1, arg1_1)
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch', function=<HostFunction add>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction add>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction add>)
    x, y = torch.broadcast_tensors(x, y)
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:151>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch', function=<HostFunction add>), key='empty_like')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    # Attribute:     %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    # Attribute:     %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 1), kwargs = {})
    # Attribute:     %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%sym_size_int, %sym_size_int_1],), kwargs = {dtype: torch.int32, layout: torch.strided, device: cpu, pin_memory: False})
    # Attribute:     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 1]), kwargs = {})
    # Attribute:     return permute
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch', function=<HostFunction add>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:150>), key=0)
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType, TileIndexType])) SourceOrigin(location=<SourceLocation type_prop_inputs.py:152>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl', function=<HostFunction add>), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl', function=<HostFunction add>)
    # Call: SequenceType((SymIntType(s0), SymIntType(s1))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:152>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:151>), key='size')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:151>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:153>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:151>)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:152>)
        # BinOp: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:153>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:153>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:150>), key=0)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:152>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:153>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:150>), key=1)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:152>)
        out[tile] = x[tile] + y[tile]
    return out""",
        )

    def test_ast_basics(self):
        output = type_propagation_report(
            type_prop_inputs.basics,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
            ignore=True,
        )
        self.assertExpectedInline(
            output,
            """\
def basics(x, y):
    # Constant: LiteralType(1024) SourceOrigin(location=<SourceLocation type_prop_inputs.py:22>)
    int_literal = 1024
    # JoinedStr: UnsupportedType('str is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:23>)
    formatted_value = f'prefix{int_literal}suffix'
    # Constant: LiteralType('abc') SourceOrigin(location=<SourceLocation type_prop_inputs.py:24>)
    joined_string = 'abc'
    # List: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: LiteralType(1024) SourceOrigin(location=<SourceLocation type_prop_inputs.py:22>)
    list_literal0 = [x, y, int_literal]
    # Tuple: SequenceType((TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    tuple_literal0 = (x, y, 1, 2)
    # List: SequenceType([LiteralType(5), TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Name: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Name: SequenceType((TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    list_literal1 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # List: SequenceType([LiteralType(5), TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Name: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Name: SequenceType((TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    tuple_literal2 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # Set: UnsupportedType('set is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:29>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:29>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:29>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:29>)
    set_literal = {1, 2, 3}
    # Dict: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:30>)
    dict_literal0 = {}
    # Name: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:30>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:31>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:31>)
    dict_literal0[1] = 2
    # Dict: DictType({1: TensorType([arg0_size0, arg0_size1], torch.int32), 'y': TensorType([arg0_size0, arg0_size1], torch.int32)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType('y') SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    dict_literal1 = {1: x, 'y': y}
    # Dict: DictType({'foo': LiteralType('bar'), 1: TensorType([arg0_size0, arg0_size1], torch.int32), 'y': TensorType([arg0_size0, arg0_size1], torch.int32)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Constant: LiteralType('foo') SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Constant: LiteralType('bar') SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Name: DictType({1: TensorType([arg0_size0, arg0_size1], torch.int32), 'y': TensorType([arg0_size0, arg0_size1], torch.int32)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    dict_literal2 = {'foo': 'bar', **dict_literal1}
    # UnaryOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:34>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    uadd = +x
    # UnaryOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:35>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    usub = -x
    # UnaryOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:36>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    invert = ~x
    # UnaryOp: SymBoolType(Eq(u0, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:37>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    not_ = not x
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    add = x + y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:39>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    sub = x - y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:40>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    mul = x * y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:42>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    truediv = x / y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:43>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    floordiv = x // y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:44>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    mod = x % y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:45>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    pow = x ** y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:46>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    lshift = x << y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:47>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    rshift = x >> y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:48>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_and = x & y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:49>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_xor = x ^ y
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:50>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_or = x | y
    # BoolOp: UnknownType('And not supported on TensorType([arg0_size0, arg0_size1], torch.int32) and TensorType([arg0_size0, arg0_size1], torch.int32)') SourceOrigin(location=<SourceLocation type_prop_inputs.py:51>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    and_ = x and y
    # BoolOp: UnknownType('And not supported on TensorType([arg0_size0, arg0_size1], torch.int32) and TensorType([arg0_size0, arg0_size1], torch.int32)') SourceOrigin(location=<SourceLocation type_prop_inputs.py:52>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    or_ = x and y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:53>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    eq = x == y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:54>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    ne = x != y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:55>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    lt = x < y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:56>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    le = x <= y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:57>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    gt = x > y
    # Compare: TensorType([arg0_size0, arg0_size1], torch.bool) SourceOrigin(location=<SourceLocation type_prop_inputs.py:58>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    ge = x >= y
    # Compare: SymBoolType(Eq(u1, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:59>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    is_ = x is y
    # Compare: SymBoolType(Eq(u2, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:60>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    is_not = x is not y
    # Compare: SymBoolType(Eq(u3, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:61>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    in_ = x in y
    # Compare: SymBoolType(Eq(u4, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:62>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    not_in = x not in y
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:63>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:63>)
    call0 = func(x, y, 3)
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:64>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:64>)
    call1 = func(x, y, c=3)
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:65>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Tuple: SequenceType((TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:65>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    call2 = func(*(x, y, y))
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: graph():
    # Name:     %a_1 : [num_users=1] = placeholder[target=a_1]
    # Name:     %b_1 : [num_users=0] = placeholder[target=b_1]
    # Name:     %c_1 : [num_users=2] = placeholder[target=c_1]
    # Name:     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%a_1, %c_1), kwargs = {})
    # Name:     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %c_1), kwargs = {})
    # Name:     return add_1
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Dict: DictType({'b': TensorType([arg0_size0, arg0_size1], torch.int32), 'c': TensorType([arg0_size0, arg0_size1], torch.int32)}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Constant: LiteralType('b') SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType('c') SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    call3 = func(x, **{'b': y, 'c': y})
    # IfExp: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: UnknownType('And not supported on TensorType([arg0_size0, arg0_size1], torch.int32) and TensorType([arg0_size0, arg0_size1], torch.int32)') SourceOrigin(location=<SourceLocation type_prop_inputs.py:52>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    ifexp = x if or_ else y
    # ListComp: UnknownType('ast.ListComp is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # comprehension: UnknownType('ast.comprehension is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # Tuple: SequenceType((LiteralType(1), LiteralType(2), LiteralType(3))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:68>)
    listcomp = [v for v in (1, 2, 3)]
    # DictComp: UnknownType('ast.DictComp is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Name: UnknownType("'k' is undefined") GlobalOrigin(name='k', function=<HostFunction basics>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # comprehension: UnknownType('ast.comprehension is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Tuple: SequenceType((UnknownType("'k' is undefined"), UnknownType("'v' is undefined"))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Name: UnknownType("'k' is undefined") GlobalOrigin(name='k', function=<HostFunction basics>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # List: SequenceType([SequenceType((LiteralType(1), LiteralType(2))), SequenceType((LiteralType(3), LiteralType(4)))]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Tuple: SequenceType((LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Tuple: SequenceType((LiteralType(3), LiteralType(4))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:69>)
    dictcomp = {k: v for k, v in [(1, 2), (3, 4)]}
    # SetComp: UnknownType('ast.SetComp is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # comprehension: UnknownType('ast.comprehension is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # Tuple: SequenceType((LiteralType(1), LiteralType(2), LiteralType(3))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:70>)
    setcomp = {v for v in (1, 2, 3)}
    # GeneratorExp: UnknownType('ast.GeneratorExp is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # comprehension: UnknownType('ast.comprehension is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    # Name: UnknownType("'v' is undefined") GlobalOrigin(name='v', function=<HostFunction basics>)
    # Tuple: SequenceType((LiteralType(1), LiteralType(2), LiteralType(3))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:71>)
    generator = (v for v in (1, 2, 3))
    # Attribute: LiteralType(torch.int32) AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='dtype')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr0 = x.dtype
    # Attribute: SequenceType((SymIntType(s0), SymIntType(s1))) AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='shape')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr1 = x.shape
    # Call: SymIntType(s0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:74>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='size')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:74>)
    attr2 = x.size(0)
    # Call: SequenceType((SymIntType(s0), SymIntType(s1))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:75>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='size')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr3 = x.size()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:76>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='stride')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr4 = x.stride()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:77>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='stride')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:77>)
    attr5 = x.stride(0)
    # NamedExpr: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    named_expr = (z := (y + 1))
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    zzz = zz = z - 1
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:80>)
    # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:80>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    q = zzz + zz + z
    # Subscript: UnknownType('Subscript not supported with self=SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) key=LiteralType(0)') SourceOrigin(location=<SourceLocation type_prop_inputs.py:81>)
    # Name: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:81>)
    subscript0 = list_literal0[0]
    # Subscript: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    # Name: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Slice: SliceType(LiteralType(1):LiteralType(None):LiteralType(None)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    subscript1 = list_literal0[1:]
    # Subscript: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # Name: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), TensorType([arg0_size0, arg0_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Slice: SliceType(LiteralType(None):LiteralType(-1):LiteralType(None)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # UnaryOp: LiteralType(-1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    subscript2 = list_literal0[:-1]
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    add += y
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    add -= y
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    a, b, c = [1, 2, 3]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:87>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    # Name: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    # Name: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:86>)
    tmp0 = [a, b, c]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    a, *bc = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4), LiteralType(1)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:89>)
    # Name: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:88>)
    tmp1 = [*bc, a]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    a, *ab, c = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(1), LiteralType(4), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:91>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Name: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    # Name: SequenceType([LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:90>)
    tmp2 = [a, c, *ab]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    # Constant: LiteralType(6) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    a, *ab, c = [5, 6]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:93>)
    # Name: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    # Name: LiteralType(6) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    # Name: SequenceType([]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:92>)
    tmp2 = [a, c, *ab]
    try:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:96>)
        e0 = 1
        # Call: UnknownType('Exception is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:97>)
        # Name: CallableType(Exception) BuiltinOrigin(name='Exception', function=<HostFunction basics>)
        raise Exception()
    except Exception as e:
        e1 = 1
    else:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:101>)
        e2 = 1
        pass
    # Compare: SymBoolType(Eq(u5, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:104>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    assert x is not y
    # Compare: SymBoolType(Eq(u6, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:105>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType('msg') SourceOrigin(location=<SourceLocation type_prop_inputs.py:105>)
    assert x is not y, 'msg'
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    del add
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:108>)
    import torch
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:109>)
    from torch import Tensor
    # Compare: SymBoolType(Eq(u7, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:111>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    if x is y:
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
        join_var0 = x
        # BinOp: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:113>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
        join_var1 = x + y
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:114>)
        join_var2 = 1
    else:
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
        join_var0 = y
        # Constant: LiteralType(None) SourceOrigin(location=<SourceLocation type_prop_inputs.py:117>)
        join_var1 = None
        # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:118>)
        join_var2 = 2
    # List: SequenceType([TensorType([arg0_size0, arg0_size1], torch.int32), UnknownType("Can't combine types from control flow: TensorType([arg0_size0, arg0_size1], torch.int32) and LiteralType(None)"), SymIntType(u8)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:119>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: UnknownType("Can't combine types from control flow: TensorType([arg0_size0, arg0_size1], torch.int32) and LiteralType(None)") SourceOrigin(location=<SourceLocation type_prop_inputs.py:117>)
    # Name: SymIntType(u8) SourceOrigin(location=<SourceLocation type_prop_inputs.py:114>)
    combined = [join_var0, join_var1, join_var2]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:121>)
    v = 0
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:122>)
    # For: loop_type=HOST
    z = 0
    # Call: LiteralType(range(0, 3)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
    # Name: CallableType(range) BuiltinOrigin(name='range', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
    for i in range(3):
        # BinOp: SymIntType(u12) SourceOrigin(location=<SourceLocation type_prop_inputs.py:124>)
        # Name: SymIntType(u11) SourceOrigin(location=<SourceLocation type_prop_inputs.py:121>)
        # Name: SymIntType(u10) SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
        v = v + i
        # BinOp: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([arg0_size0, arg0_size1], torch.int32)") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
        # Name: UnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([arg0_size0, arg0_size1], torch.int32)") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
        z = z + x
        break
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:128>)
        t = 0
    # List: SequenceType([SymIntType(u13), ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([arg0_size0, arg0_size1], torch.int32)")]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:129>)
    # Name: SymIntType(u13) SourceOrigin(location=<SourceLocation type_prop_inputs.py:124>)
    # Name: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([arg0_size0, arg0_size1], torch.int32)") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
    combined = [v, z]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:131>)
    i = 0
    # Compare: SymBoolType(Eq(u16, 1)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:132>)
    # Name: SymIntType(u14) SourceOrigin(location=<SourceLocation type_prop_inputs.py:131>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:132>)
    while i < 3:
        # BinOp: SymIntType(u18) SourceOrigin(location=<SourceLocation type_prop_inputs.py:133>)
        # Name: SymIntType(u17) SourceOrigin(location=<SourceLocation type_prop_inputs.py:131>)
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:133>)
        i = i + 1
        continue
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:136>)
        t = 0
    with contextlib.nullcontext():
    # Global: UnknownType('ast.Global is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:141>)
        e3 = 1
    global global0
    # Call: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch', function=<HostFunction basics>), key='empty_like')
    # Attribute: graph():
    # Attribute:     %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    # Attribute:     %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    # Attribute:     %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 1), kwargs = {})
    # Attribute:     %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%sym_size_int, %sym_size_int_1],), kwargs = {dtype: torch.int32, layout: torch.strided, device: cpu, pin_memory: False})
    # Attribute:     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 1]), kwargs = {})
    # Attribute:     return permute
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch', function=<HostFunction basics>)
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType, TileIndexType])) SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl', function=<HostFunction basics>), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl', function=<HostFunction basics>)
    # Call: SequenceType((SymIntType(s0), SymIntType(s1))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>), key='size')
    # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        # BinOp: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='x', function=<HostFunction basics>)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        # Subscript: TensorType([block_size0, block_size1], torch.int32) DeviceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Name: TensorType([arg0_size0, arg0_size1], torch.int32) ArgumentOrigin(name='y', function=<HostFunction basics>)
        # Name: SequenceType([TileIndexType, TileIndexType]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        out[tile] = x[tile] + y[tile]
    return out""",
        )


if __name__ == "__main__":
    unittest.main()
