from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from expecttest import TestCase
import torch
from torch._dynamo.source import LocalSource

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction
from helion._compiler.type_propagation import propagate_types
from helion.exc import TensorOperationInWrapper


def import_path(filename: Path):
    spec = importlib.util.spec_from_file_location(
        f"{__name__}.{filename.stem}", filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


type_prop_inputs = import_path(Path(__file__).parent / "data/type_prop_inputs.py")


def type_propagation_report(fn, *args):
    with CompileEnvironment() as env:
        args = [env.to_fake(arg, LocalSource(f"arg{i}")) for i, arg in enumerate(args)]
        host_fn = HostFunction(fn, env)
        host_fn.env.errors.ignore(TensorOperationInWrapper)
        propagate_types(host_fn, args, {})
        return host_fn.debug_types()


class TestTypePropagation(TestCase):
    def test_basics(self):
        output = type_propagation_report(
            type_prop_inputs.basics,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
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
    # List: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: LiteralType(1024) SourceOrigin(location=<SourceLocation type_prop_inputs.py:22>)
    list_literal0 = [x, y, int_literal]
    # Tuple: SequenceType((TensorType, TensorType, LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    tuple_literal0 = (x, y, 1, 2)
    # List: SequenceType([LiteralType(5), TensorType, TensorType, LiteralType(1024), LiteralType(3), TensorType, TensorType, LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Name: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    # Name: SequenceType((TensorType, TensorType, LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation type_prop_inputs.py:27>)
    list_literal1 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # List: SequenceType([LiteralType(5), TensorType, TensorType, LiteralType(1024), LiteralType(3), TensorType, TensorType, LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Name: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:28>)
    # Name: SequenceType((TensorType, TensorType, LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:26>)
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
    # Dict: DictType({1: TensorType, 'y': TensorType}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType('y') SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    dict_literal1 = {1: x, 'y': y}
    # Dict: DictType({'foo': LiteralType('bar'), 1: TensorType, 'y': TensorType}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Constant: LiteralType('foo') SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Constant: LiteralType('bar') SourceOrigin(location=<SourceLocation type_prop_inputs.py:33>)
    # Name: DictType({1: TensorType, 'y': TensorType}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:32>)
    dict_literal2 = {'foo': 'bar', **dict_literal1}
    # UnaryOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:34>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    uadd = +x
    # UnaryOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:35>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    usub = -x
    # UnaryOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:36>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    invert = ~x
    # UnaryOp: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:37>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    not_ = not x
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    add = x + y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:39>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    sub = x - y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:40>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    mul = x * y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:42>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    truediv = x / y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:43>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    floordiv = x // y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:44>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    mod = x % y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:45>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    pow = x ** y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:46>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    lshift = x << y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:47>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    rshift = x >> y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:48>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_and = x & y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:49>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_xor = x ^ y
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:50>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    bitwise_or = x | y
    # BoolOp: UnknownType('And not supported on TensorType and TensorType') SourceOrigin(location=<SourceLocation type_prop_inputs.py:51>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    and_ = x and y
    # BoolOp: UnknownType('And not supported on TensorType and TensorType') SourceOrigin(location=<SourceLocation type_prop_inputs.py:52>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    or_ = x and y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:53>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    eq = x == y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:54>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    ne = x != y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:55>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    lt = x < y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:56>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    le = x <= y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:57>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    gt = x > y
    # Compare: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:58>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    ge = x >= y
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:59>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    is_ = x is y
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:60>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    is_not = x is not y
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:61>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    in_ = x in y
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:62>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    not_in = x not in y
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:63>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:63>)
    call0 = func(x, y, 3)
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:64>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:64>)
    call1 = func(x, y, c=3)
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:65>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Tuple: SequenceType((TensorType, TensorType, TensorType)) SourceOrigin(location=<SourceLocation type_prop_inputs.py:65>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    call2 = func(*(x, y, y))
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: CallableType(func) GlobalOrigin(name='func', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Dict: DictType({'b': TensorType, 'c': TensorType}) SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Constant: LiteralType('b') SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType('c') SourceOrigin(location=<SourceLocation type_prop_inputs.py:66>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    call3 = func(x, **{'b': y, 'c': y})
    # IfExp: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: UnknownType('And not supported on TensorType and TensorType') SourceOrigin(location=<SourceLocation type_prop_inputs.py:52>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
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
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr0 = x.dtype
    # Attribute: SequenceType((SymIntType(s0), SymIntType(s1))) AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='shape')
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr1 = x.shape
    # Call: SymIntType(s0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:74>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='size')
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:74>)
    attr2 = x.size(0)
    # Call: SequenceType((SymIntType(s0), SymIntType(s1))) SourceOrigin(location=<SourceLocation type_prop_inputs.py:75>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='size')
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr3 = x.size()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:76>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='stride')
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    attr4 = x.stride()
    # Call: UnknownType('Tensor.stride() is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:77>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x', function=<HostFunction basics>), key='stride')
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:77>)
    attr5 = x.stride(0)
    # NamedExpr: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    named_expr = (z := (y + 1))
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    zzz = zz = z - 1
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:80>)
    # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:80>)
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:79>)
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:78>)
    q = zzz + zz + z
    # Subscript: UnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:81>)
    # Name: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:81>)
    subscript0 = list_literal0[0]
    # Subscript: UnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    # Name: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Slice: UnknownType('ast.Slice is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:82>)
    subscript1 = list_literal0[1:]
    # Subscript: UnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # Name: SequenceType([TensorType, TensorType, LiteralType(1024)]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:25>)
    # Slice: UnknownType('ast.Slice is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # UnaryOp: LiteralType(-1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:83>)
    subscript2 = list_literal0[:-1]
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    add += y
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
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
        # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:97>)
        # Name: CallableType(Exception) BuiltinOrigin(name='Exception', function=<HostFunction basics>)
        raise Exception()
    except Exception as e:
        e1 = 1
    else:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:101>)
        e2 = 1
        pass
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:104>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    assert x is not y
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:105>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Constant: LiteralType('msg') SourceOrigin(location=<SourceLocation type_prop_inputs.py:105>)
    assert x is not y, 'msg'
    # Name: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:38>)
    del add
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:108>)
    import torch
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:109>)
    from torch import Tensor
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:111>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    if x is y:
        # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
        join_var0 = x
        # BinOp: TensorType SourceOrigin(location=<SourceLocation type_prop_inputs.py:113>)
        # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
        # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
        join_var1 = x + y
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation type_prop_inputs.py:114>)
        join_var2 = 1
    else:
        # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
        join_var0 = y
        # Constant: LiteralType(None) SourceOrigin(location=<SourceLocation type_prop_inputs.py:117>)
        join_var1 = None
        # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation type_prop_inputs.py:118>)
        join_var2 = 2
    # List: SequenceType([TensorType, UnknownType("Can't combine types from control flow: TensorType and LiteralType(None)"), int]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:119>)
    # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
    # Name: UnknownType("Can't combine types from control flow: TensorType and LiteralType(None)") SourceOrigin(location=<SourceLocation type_prop_inputs.py:117>)
    # Name: int SourceOrigin(location=<SourceLocation type_prop_inputs.py:118>)
    combined = [join_var0, join_var1, join_var2]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:121>)
    v = 0
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:122>)
    z = 0
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
    # Name: CallableType(range) BuiltinOrigin(name='range', function=<HostFunction basics>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
    for i in range(3):
        # BinOp: ChainedUnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:124>)
        # Name: ChainedUnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:124>)
        # Name: UnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:123>)
        v = v + i
        # BinOp: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
        # Name: UnknownType("Can't combine types from control flow: LiteralType(0) and TensorType") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
        # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
        z = z + x
        break
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:128>)
        t = 0
    # List: SequenceType([ChainedUnknownType("Iteration over UnknownType('TODO') is not supported"), ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType")]) SourceOrigin(location=<SourceLocation type_prop_inputs.py:129>)
    # Name: ChainedUnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:124>)
    # Name: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType") SourceOrigin(location=<SourceLocation type_prop_inputs.py:125>)
    combined = [v, z]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation type_prop_inputs.py:131>)
    i = 0
    # Compare: bool SourceOrigin(location=<SourceLocation type_prop_inputs.py:132>)
    # Name: int SourceOrigin(location=<SourceLocation type_prop_inputs.py:133>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation type_prop_inputs.py:132>)
    while i < 3:
        # BinOp: int SourceOrigin(location=<SourceLocation type_prop_inputs.py:133>)
        # Name: int SourceOrigin(location=<SourceLocation type_prop_inputs.py:133>)
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
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch', function=<HostFunction basics>), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch', function=<HostFunction basics>)
    # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
    out = torch.empty_like(x)
    # Call: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl', function=<HostFunction basics>), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl', function=<HostFunction basics>)
    # Call: ChainedUnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
    # Attribute: ChainedUnknownType('TODO') AttributeOrigin(value=SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>), key='size')
    # Name: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
    for tile in hl.tile(out.size()):
        # Name: UnknownType('TODO') SourceOrigin(location=<SourceLocation type_prop_inputs.py:143>)
        # Name: UnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        # BinOp: ChainedUnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Subscript: UnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Name: TensorType ArgumentOrigin(name='x', function=<HostFunction basics>)
        # Name: UnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        # Subscript: UnknownType('ast.Subscript is not supported') SourceOrigin(location=<SourceLocation type_prop_inputs.py:145>)
        # Name: TensorType ArgumentOrigin(name='y', function=<HostFunction basics>)
        # Name: UnknownType("Iteration over UnknownType('TODO') is not supported") SourceOrigin(location=<SourceLocation type_prop_inputs.py:144>)
        out[tile] = x[tile] + y[tile]
    return out""",
        )


if __name__ == "__main__":
    unittest.main()
