from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import inspect
import itertools
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import NoReturn
from typing import Protocol

import sympy
import torch

from .. import exc
from .ast_extension import ExtendedAST
from .compile_environment import warning
from .source_location import SourceLocation
from .source_location import current_location
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import BuiltinOrigin
from .variable_origin import ClosureOrigin
from .variable_origin import DeviceOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import Origin
from .variable_origin import SourceOrigin

# pyre-ignore-all-errors[8,15,58]: visit_* overrides
if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from .host_function import HostFunction

    class _VisitMethod(Protocol):
        @staticmethod
        def __call__(self: object, node: ast.AST) -> TypeInfo: ...


class Scope:
    def get(self, name: str) -> TypeInfo:
        raise NotImplementedError

    def set(self, name: str, type_info: TypeInfo) -> NoReturn:
        raise NotImplementedError


@dataclasses.dataclass
class GlobalScope(Scope):
    function: HostFunction
    cache: dict[str, TypeInfo] = dataclasses.field(default_factory=dict)

    def get(self, name: str) -> TypeInfo:
        if name not in self.cache:
            self.cache[name] = self._get(name)
        return self.cache[name]

    def _get(self, name: str) -> TypeInfo:
        origin = GlobalOrigin(name=name, function=self.function)
        try:
            # TODO(jansel): record global access and maybe specialize
            value = self.function.fn.__globals__[name]
        except KeyError:
            if hasattr(builtins, name):
                value = getattr(builtins, name)
                origin = BuiltinOrigin(name=name, function=self.function)
            else:
                return UnknownType(origin, f"{name!r} is undefined")
        return TypeInfo.from_example(value, origin)

    def set(self, name: str, type_info: TypeInfo) -> NoReturn:
        raise AssertionError("Cannot set in global scope")


@dataclasses.dataclass
class LocalScope(Scope):
    parent: Scope
    variables: dict[str, TypeInfo] = dataclasses.field(default_factory=dict)

    def get(self, name: str) -> TypeInfo:
        if name in self.variables:
            return self.variables[name]
        return self.parent.get(name)

    def set(self, name: str, type_info: TypeInfo) -> None:
        self.variables[name] = type_info

    def merge(self, other: LocalScope | dict[str, TypeInfo]) -> LocalScope:
        if isinstance(other, LocalScope):
            other = other.variables
        for k, v in other.items():
            if k in self.variables:
                existing = self.variables[k]
                merged = existing.merge(v)
                if (
                    isinstance(merged, UnknownType)
                    and not isinstance(existing, UnknownType)
                    and not isinstance(v, UnknownType)
                ):
                    # Improve error message
                    merged = UnknownType(
                        merged.origin,
                        "Variable {k!r} has different types in control flow: {existing!s} and {v!s}",
                    )
                self.variables[k] = merged
            else:
                self.variables[k] = v
        return self

    def merge_if_else(
        self, true_scope: LocalScope, false_scope: LocalScope
    ) -> LocalScope:
        true = {**true_scope.variables}
        false = {**false_scope.variables}
        both = {}
        for k in [*false]:
            if k in true:
                both[k] = true.pop(k).merge(false.pop(k))
        self.merge(true)
        self.merge(false)
        # variables defined in both sides of branch overwrite existing values
        self.variables.update(both)
        return self

    def overwrite(self, other: LocalScope) -> None:
        self.variables.update(other.variables)

    def clone(self) -> LocalScope:
        return LocalScope(parent=self.parent, variables=dict(self.variables))


class TypeInfo:
    origin: Origin

    def __init__(self, origin: Origin) -> None:
        assert isinstance(origin, Origin)
        self.origin = origin

    @classmethod
    def from_example(
        cls, value: object, origin: Origin, *, specialize: bool = False
    ) -> TypeInfo:
        if isinstance(value, torch.Tensor):
            # TODO(jansel): need to wrap this in a fake tensor
            # TODO(jansel): tensor subclass support
            return TensorType(origin, fake_value=value)
        if isinstance(value, torch.SymInt):
            if isinstance(value._sympy_(), (int, sympy.Integer)):
                return cls.from_example(
                    int(value._sympy_()), origin, specialize=specialize
                )
            return SymIntType(origin, value)
        if type(value) in (int, float, bool):
            if specialize:
                return LiteralType(origin, value)
            # pyre-ignore[6]
            return NumericType(origin, type(value))
        if type(value) in (str, torch.dtype, torch.device):
            # TODO(jansel): track specializations
            return LiteralType(origin, value)
        if isinstance(value, types.ModuleType):
            return PythonModuleType(origin, value)
        if callable(value):
            # TODO(jansel): track specializations
            return CallableType(origin, value)
        if type(value) is list:
            # TODO(jansel): track specializations
            return SequenceType(origin, cls._unpack_example(enumerate(value), origin))
        if type(value) is tuple:
            # TODO(jansel): track specializations
            return SequenceType(
                origin,
                tuple(cls._unpack_example(enumerate(value), origin)),
            )
        if type(value) is torch.Size:
            return cls.from_example(tuple(value), origin, specialize=specialize)
        if type(value) is dict:
            # TODO(jansel): track specializations
            if not all(type(key) in (str, int) for key in value):
                return UnknownType(
                    debug_msg="Only int/string keys are supported in dict",
                    origin=origin,
                )
            items: list[tuple[int | str, object]] = [*value.items()]
            return DictType(
                origin,
                dict(zip(value.keys(), cls._unpack_example(items, origin))),
            )
        return UnknownType(
            debug_msg=f"{type(value).__name__} is not supported",
            origin=origin,
        )

    @staticmethod
    def _unpack_example(
        values: Sequence[tuple[int | str, object]] | enumerate[object],
        origin: Origin,
    ) -> list[TypeInfo]:
        return [
            TypeInfo.from_example(value, GetItemOrigin(origin, key))
            for key, value in values
        ]

    def __str__(self) -> str:
        return type(self).__name__

    def debug_annotation(self) -> str:
        return f"{self!s} {self.origin!r}"

    def merge(self, other: TypeInfo) -> TypeInfo:
        """Combine two types at a join point in control flow."""
        if isinstance(other, UnknownType):
            return other
        if isinstance(other, NoType):
            return self
        return UnknownType(
            debug_msg=f"Can't combine types from control flow: {self!s} and {other!s}",
            origin=other.origin,
        )

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {self!s}",
            origin=origin,
        )

    def propagate_host_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return self.propagate_call(args, kwargs, origin)

    def propagate_device_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return self.propagate_call(args, kwargs, origin)

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return UnknownType(
            debug_msg=f"Function calls are not supported on {self!s}",
            origin=origin,
        )

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return UnknownType(
            debug_msg=f"Attributes are not supported on {self!s}",
            origin=origin,
        )

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        """Should return updated type of self after running `self[key] = value`"""
        return UnknownType(
            debug_msg=f"SetItem not supported with self={self!s} key={key!s} value={value!s}",
            origin=origin,
        )

    def propagate_iter(self, origin: Origin) -> TypeInfo:
        try:
            values = self.unpack()
        except NotImplementedError:
            pass
        else:
            for val in values:
                if isinstance(val, UnknownType):
                    return val.chained(origin)
            return functools.reduce(lambda x, y: x.merge(y), values)
        return UnknownType(
            debug_msg=f"Iteration over {self!s} is not supported",
            origin=origin,
        )

    def unpack(self) -> list[TypeInfo]:
        raise NotImplementedError

    def example_value(self) -> object:
        raise NotImplementedError

    def truth_value(self) -> bool:
        return len(self.unpack()) > 0

    def as_literal(self) -> object:
        raise NotImplementedError


class TensorType(TypeInfo):
    fake_value: torch.Tensor

    def __init__(self, origin: Origin, fake_value: torch.Tensor) -> None:
        super().__init__(origin)
        self.fake_value = fake_value

    def example_value(self) -> object:
        return self.fake_value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        warning(exc.TensorOperationInWrapper)
        if isinstance(op, ast.Not):
            return NumericType(origin, bool)
        try:
            return TypeInfo.from_example(_eval_unary(op, self.fake_value), origin)
        except Exception as e:
            raise exc.TorchOpTracingError(e) from e

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        assert origin.key == attr
        if attr in {"dtype", "device", "ndim", "shape"}:
            return TypeInfo.from_example(getattr(self.fake_value, attr), origin)
        return TensorAttributeType(origin, self)

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)
        return self

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, TensorType):
            if self.fake_value.device != other.fake_value.device:
                return UnknownType(
                    debug_msg=f"device mismatch in control flow: {self.fake_value.device} != {other.fake_value.device}",
                    origin=other.origin,
                )
            if self.fake_value.dtype != other.fake_value.dtype:
                return UnknownType(
                    debug_msg=f"dtype mismatch in control flow: {self.fake_value.dtype} != {other.fake_value.dtype}",
                    origin=other.origin,
                )
            if self.fake_value.dim() != other.fake_value.dim():
                return UnknownType(
                    debug_msg=f"rank mismatch in control flow: {self.fake_value.dim()} != {other.fake_value.dim()}",
                    origin=other.origin,
                )
            if self.fake_value.size() != other.fake_value.size():
                return UnknownType(
                    debug_msg=f"size mismatch in control flow: {self.fake_value.size()} != {other.fake_value.size()}",
                    origin=other.origin,
                )
            # TODO(jansel): handle symbolic shapes
            # TODO(jansel): stride check?
            return TensorType(other.origin, torch.empty_like(self.fake_value))
        return super().merge(other)


class TensorAttributeType(TypeInfo):
    origin: AttributeOrigin
    tensor: TensorType

    def __init__(self, origin: AttributeOrigin, tensor: TensorType) -> None:
        super().__init__(origin)
        self.tensor = tensor

    def attr(self) -> str:
        return self.origin.key

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        attr = self.attr()
        if attr in {"dim", "ndimension"} and not (args or kwargs):
            return TypeInfo.from_example(self.tensor.fake_value.ndim, origin)
        if attr in {"shape", "size"} and not kwargs:
            fn = getattr(self.tensor.fake_value, attr)
            try:
                return TypeInfo.from_example(
                    fn(*[x.as_literal() for x in args]),
                    origin,
                )
            except NotImplementedError:
                return UnknownType(origin, f"Tensor.{attr}() args must be literals")
        return UnknownType(origin, f"Tensor.{attr}() is not supported")


class LiteralType(TypeInfo):
    value: object

    def __init__(self, origin: Origin, value: object) -> None:
        super().__init__(origin)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    @property
    def python_type(self) -> type[object]:
        return type(self.value)

    def example_value(self) -> object:
        return self.value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return TypeInfo.from_example(
            _eval_unary(op, self.value),
            origin,
            specialize=True,
        )

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return TypeInfo.from_example(getattr(self.value, attr), origin, specialize=True)

    def truth_value(self) -> bool:
        return bool(self.value)

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, LiteralType) and self.value == other.value:
            return self
        if isinstance(other, (LiteralType, NumericType)):
            if self.python_type == other.python_type:
                # pyre-ignore[6]
                return NumericType(other.origin, self.python_type)
        return super().merge(other)

    def unpack(self) -> list[TypeInfo]:
        # pyre-ignore[6]
        it = iter(self.value)
        return [TypeInfo.from_example(x, self.origin, specialize=True) for x in it]

    def as_literal(self) -> object:
        return self.value


# pyre-ignore[36]
_constant_prop_functions: dict[Callable[..., object], None] = dict.fromkeys(
    [abs, all, any, str, int, list, dict, tuple, float],
)


class CallableType(LiteralType):
    value: Callable[..., object]

    def __init__(self, origin: Origin, value: Callable[..., object]) -> None:
        super().__init__(origin, value)
        self.value = value

    def __str__(self) -> str:
        try:
            name = self.value.__qualname__
        except AttributeError:
            try:
                name = self.value.__name__
            except AttributeError:
                name = str(self.value)
        return f"{type(self).__name__}({name})"

    def _propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo | None:
        if self.value in _constant_prop_functions and all(
            isinstance(arg, LiteralType)
            for arg in itertools.chain(args, kwargs.values())
        ):
            return TypeInfo.from_example(
                self.value(
                    # pyre-ignore[16]
                    *[arg.value for arg in args],
                    **{key: arg.value for key, arg in kwargs.items()},
                ),
                origin,
                specialize=True,
            )
        if self.value in (list, tuple) and len(kwargs) == 0:
            if len(args) == 0:
                # pyre-ignore[6]
                return SequenceType(origin, self.value())
            if len(args) == 1 and isinstance(arg0 := args[0], SequenceType):
                # pyre-ignore[6]
                return SequenceType(origin, self.value(arg0.element_types))
        if self.value is dict:
            if len(args) == 0:
                # pyre-ignore[6]
                return DictType(origin, kwargs)
            if not kwargs and len(args) == 1 and isinstance(arg0 := args[0], DictType):
                # pyre-ignore[6]
                return DictType(origin, self.value(arg0.element_types))
            return None
        return None

    def propagate_host_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        rv = self._propagate_call(args, kwargs, origin)
        if rv is not None:
            return rv
        return UnknownType(debug_msg="TODO", origin=origin)

    def propagate_device_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        rv = self._propagate_call(args, kwargs, origin)
        if rv is not None:
            return rv
        return UnknownType(debug_msg="TODO", origin=origin)


class PythonModuleType(LiteralType):
    value: types.ModuleType

    def __init__(self, origin: Origin, value: types.ModuleType) -> None:
        super().__init__(origin, value)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value.__name__})"


class NumericType(TypeInfo):
    python_type: type[float | int | bool]

    def __init__(self, origin: Origin, python_type: type[float | int | bool]) -> None:
        super().__init__(origin)
        self.python_type = python_type

    def __str__(self) -> str:
        return self.python_type.__name__

    def example_value(self) -> object:
        return {int: 0, float: 0.0, bool: False}[self.python_type]

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        # pyre-ignore[6]
        return NumericType(origin, type(_eval_unary(op, self.example_value())))

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, (LiteralType, NumericType)):
            if self.python_type == other.python_type:
                return NumericType(self.origin, self.python_type)
        return super().merge(other)


class SymIntType(NumericType):
    value: torch.SymInt

    def __init__(self, origin: Origin, value: torch.SymInt) -> None:
        super().__init__(origin, int)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value})"

    def merge(self, other: TypeInfo) -> TypeInfo:
        if (
            isinstance(other, SymIntType)
            and self.value._sympy_() == other.value._sympy_()
        ):
            return self
        return super().merge(other)


class NoType(TypeInfo):
    """Used for AST nodes like Store() where a type is not applicable."""

    def merge(self, other: TypeInfo) -> TypeInfo:
        return other

    def debug_annotation(self) -> str:
        return ""


class CollectionType(TypeInfo):
    element_types: list[TypeInfo] | tuple[TypeInfo, ...] | dict[str | int, TypeInfo]

    def __init__(
        self,
        origin: Origin,
        element_types: list[TypeInfo]
        | tuple[TypeInfo, ...]
        | dict[str | int, TypeInfo],
    ) -> None:
        super().__init__(origin)
        self.element_types = element_types

    @property
    def python_type(self) -> type[object]:
        return type(self.element_types)

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        if isinstance(op, ast.Not):
            return LiteralType(origin, not self.element_types)
        return super().propagate_unary(op, origin)

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if isinstance(key, LiteralType):
            if isinstance(elements := self.element_types, (list, dict)) and isinstance(
                k := key.value, (int, str)
            ):
                if k in elements:
                    # pyre-ignore[6]
                    elements[k] = elements[k].merge(value)
                else:
                    # pyre-ignore[6]
                    elements[k] = value
                return self
        return super().propagate_setitem(key, value, origin)

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, CollectionType):
            self_elements = self.element_types
            other_elements = self.element_types
            if (
                isinstance(self_elements, dict)
                and isinstance(other_elements, dict)
                and set(self_elements.keys()) == set(other_elements.keys())
            ):
                return DictType(
                    element_types={
                        key: self_elements[key].merge(other_elements[key])
                        for key in self_elements
                    },
                    origin=other.origin,
                )
            if (
                isinstance(self_elements, (list, tuple))
                and isinstance(other_elements, (list, tuple))
                and len(self_elements) == len(other_elements)
            ):
                element_types = [
                    self_elements[i].merge(other_elements[i])
                    for i in range(len(self_elements))
                ]
                return SequenceType(origin=other.origin, element_types=element_types)
        return super().merge(other)

    def truth_value(self) -> bool:
        return bool(self.element_types)

    def unpack(self) -> list[TypeInfo]:
        if isinstance(self.element_types, dict):
            return [
                TypeInfo.from_example(k, self.origin, specialize=True)
                for k in self.element_types
            ]
        assert isinstance(self.element_types, (list, tuple))
        return [*self.element_types]

    def as_literal(self) -> object:
        if isinstance(self.element_types, dict):
            return {k: v.as_literal() for k, v in self.element_types.items()}
        assert isinstance(self.element_types, (list, tuple))
        # pyre-ignore[19]
        return self.python_type([x.as_literal() for x in self.element_types])


class SequenceType(CollectionType):
    element_types: list[TypeInfo] | tuple[TypeInfo, ...]

    def __str__(self) -> str:
        start, *_, end = repr(self.element_types)
        if len(self.element_types) == 1 and self.python_type is tuple:
            end = ", )"
        items = ", ".join(map(str, self.element_types))
        return f"{type(self).__name__}({start}{items}{end})"


class DictType(CollectionType):
    element_types: dict[str | int, TypeInfo]

    def __str__(self) -> str:
        items = ", ".join(f"{k!r}: {v!s}" for k, v in self.element_types.items())
        return f"{type(self).__name__}({{{items}}})"


class TypeNotAllowedOnDevice(TypeInfo):
    pass


class UnknownType(TypeNotAllowedOnDevice):
    def __init__(self, origin: Origin, debug_msg: str) -> None:
        super().__init__(origin)
        self.locations: list[SourceLocation] = [current_location()]
        self.debug_msg: str = debug_msg

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.debug_msg!r})"

    def chained(self, origin: Origin) -> TypeInfo:
        return ChainedUnknownType(origin, self)

    def merge(self, other: TypeInfo) -> TypeInfo:
        return self

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return self.chained(origin)

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return self.chained(origin)

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return self.chained(origin)


class UnsupportedType(UnknownType):
    def __init__(self, origin: Origin, python_type: type[object]) -> None:
        super().__init__(
            origin=origin,
            debug_msg=f"{python_type.__name__} is not supported",
        )
        self.python_type = python_type


class ChainedUnknownType(UnknownType):
    """Keep track of multiple locations for an operation that is already unknown type."""

    def __init__(self, origin: Origin, prior_type: UnknownType) -> None:
        super().__init__(origin=origin, debug_msg=prior_type.debug_msg)
        self.locations: list[SourceLocation] = [*prior_type.locations, *self.locations]


def _eval_unary(op: ast.unaryop, value: object) -> object:
    if isinstance(op, ast.Not):
        return not value
    if isinstance(op, ast.UAdd):
        # pyre-ignore[16]
        return +value
    if isinstance(op, ast.USub):
        # pyre-ignore[16]
        return -value
    if isinstance(op, ast.Invert):
        # pyre-ignore[16]
        return ~value
    raise AssertionError(f"{type(op).__name__} unknown unary op")


def _eval_binary(op: ast.operator, left: object, right: object) -> object:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        return left**right
    if isinstance(op, ast.LShift):
        return left << right
    if isinstance(op, ast.RShift):
        return left >> right
    if isinstance(op, ast.BitOr):
        return left | right
    if isinstance(op, ast.BitXor):
        return left ^ right
    if isinstance(op, ast.BitAnd):
        return left & right
    if isinstance(op, ast.MatMult):
        return left @ right
    raise AssertionError(f"{type(op).__name__} unknown binary op")


def _eval_compare(op: ast.cmpop, left: object, right: object) -> object:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    raise AssertionError(f"{type(op).__name__} unknown compare op")


CMP_BASIC = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
CMP_IS = (
    ast.Is,
    ast.IsNot,
)
CMP_IN = (
    ast.In,
    ast.NotIn,
)
CMP_ALWAYS_BOOL: tuple[type[ast.AST], ...] = (
    *CMP_IS,
    *CMP_IN,
)


def _unsupported(
    python_type: type[object],
) -> Callable[[TypePropagation, ast.AST], TypeInfo]:
    def visit(self: TypePropagation, node: ast.AST) -> TypeInfo:
        # pyre-ignore[16]
        super(TypePropagation, self).generic_visit(node)
        return UnsupportedType(python_type=python_type, origin=self.origin())

    return visit


class TypePropagation(ast.NodeVisitor):
    def __init__(self, func: HostFunction, scope: LocalScope) -> None:
        super().__init__()
        self.func = func
        self.scope = scope
        self.device_loop_depth = 0

    def push_scope(self) -> None:
        self.scope = LocalScope(parent=self.scope)

    def pop_scope_merge(self) -> None:
        parent = self.scope.parent
        assert isinstance(parent, LocalScope)
        parent.merge(self.scope)
        self.scope = parent

    def pop_scope_overwrite(self) -> None:
        parent = self.scope.parent
        assert isinstance(parent, LocalScope)
        parent.overwrite(self.scope)
        self.scope = parent

    def pop_scope(self) -> LocalScope:
        current = self.scope
        parent = current.parent
        assert isinstance(parent, LocalScope)
        self.scope = parent
        return current

    @contextlib.contextmanager
    def swap_scope(self, new_scope: LocalScope) -> Iterator[None]:
        prior = self.scope
        self.scope = new_scope
        try:
            yield
        finally:
            self.scope = prior

    def visit(self, node: ast.AST) -> TypeInfo:
        assert isinstance(node, ExtendedAST)
        with node:
            try:
                visitor = getattr(
                    self,
                    f"visit_{node.__class__.__name__}",
                    self.generic_visit,
                )
                type_info = visitor(node)
                assert isinstance(type_info, TypeInfo), (
                    f"expected TypeInfo, got {type_info!r} from {visitor!r}"
                )
                return node.update_type_info(type_info)
            except exc.Base:
                raise
            except Exception as e:
                raise exc.InternalError(e) from e

    def origin(self) -> Origin:
        if self.device_loop_depth == 0:
            return SourceOrigin(current_location())
        return DeviceOrigin(current_location())

    def generic_visit(self, node: ast.AST) -> TypeInfo:
        super().generic_visit(node)
        return UnknownType(
            debug_msg=f"ast.{node.__class__.__name__} is not supported",
            origin=self.origin(),
        )

    def _bool_op(self, op: ast.boolop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        try:
            val = left.truth_value()
            if isinstance(op, ast.Or):
                val = not val
            if val:
                return left
            return right
        except NotImplementedError:
            pass
        if (
            isinstance(left, NumericType)
            and isinstance(right, NumericType)
            and left.python_type == right.python_type
        ):
            return NumericType(origin=self.origin(), python_type=left.python_type)

        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())

        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
        )

    def _compare(self, op: ast.cmpop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        if isinstance(left, LiteralType) and isinstance(right, LiteralType):
            return LiteralType(
                origin=self.origin(),
                value=_eval_compare(op, left.value, right.value),
            )
        if (
            isinstance(left, LiteralType)
            and isinstance(right, CollectionType)
            and isinstance(op, CMP_IN)
        ):
            return LiteralType(
                origin=self.origin(),
                value=_eval_compare(op, left.value, right.element_types),
            )
        if isinstance(left, (NumericType, LiteralType)) and isinstance(
            right,
            (NumericType, LiteralType),
        ):
            return NumericType(origin=self.origin(), python_type=bool)
        if isinstance(op, CMP_ALWAYS_BOOL):
            return NumericType(origin=self.origin(), python_type=bool)
        if isinstance(left, TensorType) or isinstance(right, TensorType):
            try:
                left_example = left.example_value()
                right_example = right.example_value()
            except NotImplementedError:
                pass
            else:
                try:
                    return TypeInfo.from_example(
                        _eval_compare(op, left_example, right_example),
                        self.origin(),
                    )
                except Exception as e:
                    raise exc.TorchOpTracingError(e) from e
        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())
        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
        )

    def _assign(self, lhs: ast.AST, rhs: TypeInfo) -> None:
        if isinstance(lhs, ast.Name):
            return self.scope.set(lhs.id, rhs)
        if isinstance(lhs, ast.Starred):
            try:
                unpacked = SequenceType(self.origin(), rhs.unpack())
            except NotImplementedError:
                unpacked = UnknownType(
                    self.origin(),
                    f"Failed to unpack starred assignment: {rhs!s}",
                )
            return self._assign(lhs.value, unpacked)
        if isinstance(lhs, (ast.Tuple, ast.List)):
            lhs = lhs.elts
            elements: list[TypeInfo]
            try:
                elements = rhs.unpack()
            except NotImplementedError:
                elements = [
                    UnknownType(
                        self.origin(),
                        f"Failed to unpack assignment: {rhs!s}",
                    )
                    for _ in lhs
                ]
            used_star = False
            idx = 0
            for elt in lhs:
                if isinstance(elt, ast.Starred):
                    # TODO(jansel): need to test this
                    assert not used_star, "multiple `*` in assignment"
                    used_star = True
                    star_len = len(elements) - len(lhs) + 1
                    assert star_len >= 0, "wrong number of elements to unpack"
                    self._assign(
                        elt.value,
                        SequenceType(self.origin(), elements[idx : idx + star_len]),
                    )
                    idx += star_len
                else:
                    self._assign(elt, elements[idx])
                    idx += 1
            assert idx == len(elements), "wrong number of elements to unpack"
            return None
        if isinstance(lhs, ast.Subscript):
            # TODO(jansel): test different types of subscript
            lhs_base_type = self.visit(lhs.value)
            lhs_base_type = lhs_base_type.propagate_setitem(
                self.visit(lhs.slice), rhs, self.origin()
            )
            # update the stored type for the container
            return self._assign(lhs.value, lhs_base_type)
        raise AssertionError(f"unhandled lhs in assignment {type(lhs).__name__}")

    ################################################################
    # Expressions
    ################################################################

    def visit_Constant(self, node: ast.Constant) -> TypeInfo:
        return LiteralType(value=node.value, origin=self.origin())

    visit_FormattedValue: _VisitMethod = _unsupported(str)
    visit_JoinedStr: _VisitMethod = _unsupported(str)

    def _list_or_tuple(self, node: ast.List | ast.Tuple) -> TypeInfo:
        errors = []
        elements = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                to_unpack = self.visit(elt.value)
                try:
                    elements.extend(to_unpack.unpack())
                except NotImplementedError:
                    errors.append(
                        UnknownType(
                            self.origin(),
                            f"Failed to unpack starred assignment: {to_unpack!s}",
                        )
                    )
            else:
                elements.append(self.visit(elt))
        if errors:
            return errors[0]
        cls = list if isinstance(node, ast.List) else tuple
        return SequenceType(
            self.origin(),
            cls(elements),
        )

    visit_List: _VisitMethod = _list_or_tuple
    visit_Tuple: _VisitMethod = _list_or_tuple
    visit_Set: _VisitMethod = _unsupported(set)

    def visit_Dict(self, node: ast.Dict) -> TypeInfo:
        assert len(node.keys) == len(node.values)
        errors = []
        element_types = {}
        for key_node, value_node in zip(node.keys, node.values):
            value = self.visit(value_node)
            if key_node is not None:
                key = self.visit(key_node)
                if not (
                    isinstance(key, LiteralType) and isinstance(key.value, (str, int))
                ):
                    errors.append(
                        UnknownType(
                            debug_msg=f"Only string/int literals are supported as dict keys, got {key!s}",
                            origin=self.origin(),
                        )
                    )
                    continue
                element_types[key.value] = value
            else:
                if not (isinstance(value, DictType)):
                    errors.append(
                        UnknownType(
                            debug_msg=f"Only collection types are supported as dict** values, got {value!s}",
                            origin=self.origin(),
                        )
                    )
                    continue
                element_types.update(value.element_types)
        if errors:
            return errors[0]
        return DictType(element_types=element_types, origin=self.origin())

    def visit_Name(self, node: ast.Name) -> TypeInfo:
        return self.scope.get(node.id)

    visit_Starred: _VisitMethod = generic_visit

    def visit_Expr(self, node: ast.Expr) -> TypeInfo:
        return self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> TypeInfo:
        return self.visit(node.operand).propagate_unary(node.op, self.origin())

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(left, TensorType) or isinstance(right, TensorType):
            warning(exc.TensorOperationInWrapper)

        try:
            left_example = left.example_value()
            right_example = right.example_value()
        except NotImplementedError:
            pass
        else:
            try:
                return TypeInfo.from_example(
                    _eval_binary(node.op, left_example, right_example),
                    self.origin(),
                )
            except Exception as e:
                raise exc.TorchOpTracingError(e) from e

        if isinstance(left, CollectionType) and isinstance(right, CollectionType):
            result = _eval_binary(
                node.op,
                left.element_types,
                right.element_types,
            )
            if isinstance(result, dict):
                return DictType(
                    element_types=result,
                    origin=self.origin(),
                )
            assert isinstance(result, (list, tuple))
            return SequenceType(
                element_types=result,
                origin=self.origin(),
            )

        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())

        return UnknownType(
            debug_msg=f"{type(node.op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:
        values = [self.visit(node.values[0])]
        for value in node.values[1:]:
            # Everything after first node is conditionally executed
            self.push_scope()
            values.append(self.visit(value))
            self.pop_scope_merge()

        result = values[0]
        for value in values[1:]:
            result = self._bool_op(node.op, result, value)
        return result

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:
        comparators = [
            self.visit(node.left),
            *[self.visit(comparator) for comparator in node.comparators],
        ]
        if any(isinstance(comparator, TensorType) for comparator in comparators):
            warning(exc.TensorOperationInWrapper)
        result = self._compare(node.ops[0], comparators[0], comparators[1])
        for i in range(2, len(comparators)):
            new_result = self._compare(
                node.ops[i - 1],
                comparators[i - 1],
                comparators[i],
            )
            result = self._bool_op(ast.And(), result, new_result)
        return result

    def visit_Call(self, node: ast.Call) -> TypeInfo:
        # TODO(jansel): test handling if *args and **kwargs
        func = self.visit(node.func)
        unhandled = []
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                arg_type = self.visit(arg.value)
                if isinstance(arg_type, SequenceType):
                    args.extend(arg_type.element_types)
                else:
                    unhandled.append(arg_type)
            else:
                args.append(self.visit(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                kwarg_type = self.visit(kwarg.value)
                if isinstance(kwarg_type, DictType):
                    kwargs.update(kwarg_type.element_types)
                else:
                    unhandled.append(kwarg_type)
            else:
                kwargs[kwarg.arg] = self.visit(kwarg.value)
        if unhandled:
            for arg in unhandled:
                if isinstance(arg, UnknownType):
                    return arg.chained(self.origin())
            return UnknownType(
                debug_msg="Failed to unpack */** args to function, got: "
                + ", ".join(map(str, unhandled)),
                origin=self.origin(),
            )
        if self.device_loop_depth:
            return func.propagate_device_call(tuple(args), kwargs, self.origin())
        return func.propagate_host_call(tuple(args), kwargs, self.origin())

    def visit_IfExp(self, node: ast.IfExp) -> TypeInfo:
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        try:
            truth_val = test.truth_value()
            if truth_val:
                return body
            return orelse
        except NotImplementedError:
            pass
        return body.merge(orelse)

    def visit_Attribute(self, node: ast.Attribute) -> TypeInfo:
        value = self.visit(node.value)
        origin = AttributeOrigin(value.origin, node.attr)
        return value.propagate_attribute(node.attr, origin)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> TypeInfo:
        # x := y
        type_info = self.visit(node.value)
        self._assign(node.target, type_info)
        return type_info

    # TODO(jansel): need to implement these
    visit_Subscript: _VisitMethod = generic_visit
    visit_Slice: _VisitMethod = generic_visit
    visit_ListComp: _VisitMethod = generic_visit
    visit_SetComp: _VisitMethod = generic_visit
    visit_GeneratorExp: _VisitMethod = generic_visit
    visit_DictComp: _VisitMethod = generic_visit

    ################################################################
    # Statements
    ################################################################

    def generic_statement(self, node: ast.AST) -> TypeInfo:
        self.generic_visit(node)
        return NoType(origin=self.origin())

    def visit_Assign(self, node: ast.Assign) -> TypeInfo:
        type_info = self.visit(node.value)
        for target in node.targets:
            self._assign(target, type_info)
        return NoType(origin=self.origin())

    def visit_AnnAssign(self, node: ast.AnnAssign) -> TypeInfo:
        # TODO(jansel): handle constexpr in annotation
        if node.value is not None:
            type_info = self.visit(node.value)
            self._assign(node.target, type_info)
        return NoType(origin=self.origin())

    visit_AugAssign: _VisitMethod = generic_statement
    visit_Raise: _VisitMethod = generic_statement
    visit_Assert: _VisitMethod = generic_statement
    visit_Delete: _VisitMethod = generic_statement
    visit_Pass: _VisitMethod = generic_statement
    visit_TypeAlias: _VisitMethod = generic_statement
    visit_Import: _VisitMethod = generic_statement
    visit_ImportFrom: _VisitMethod = generic_statement

    # TODO(jansel): support lambda
    visit_Lambda: _VisitMethod = generic_visit

    ################################################################
    # Control flow
    ################################################################

    def visit_If(self, node: ast.If) -> TypeInfo:
        test = self.visit(node.test)
        body = self._body(node.body)
        orelse = self._body(node.orelse)
        try:
            truth_val = test.truth_value()
            if truth_val:
                self.scope.merge(body)
            else:
                self.scope.merge(orelse)
        except NotImplementedError:
            self.scope.merge_if_else(body, orelse)
        return NoType(origin=self.origin())

    def _body(self, stmts: list[ast.stmt]) -> LocalScope:
        self.push_scope()
        for stmt in stmts:
            self.visit(stmt)
        return self.pop_scope()

    def _loop_body(self, stmts: list[ast.stmt]) -> LocalScope:
        self.push_scope()
        exit_scopes = [self.scope]
        for stmt in stmts:
            self.visit(stmt)
            if isinstance(stmt, (ast.Break, ast.Continue)):
                exit_scopes.append(self.scope.clone())
        self.pop_scope()
        return functools.reduce(lambda x, y: x.merge(y), exit_scopes)

    def visit_For(self, node: ast.For) -> TypeInfo:
        self.push_scope()
        iter_type = self.visit(node.iter)
        self._assign(node.target, iter_type.propagate_iter(self.origin()))
        body = self._loop_body(node.body)
        with self.swap_scope(body):
            # second pass for fixed point
            body.merge(self._loop_body(node.body))
        orelse = self._body(node.orelse)
        self.scope.merge_if_else(body, orelse)
        return NoType(origin=self.origin())

    def visit_While(self, node: ast.While) -> TypeInfo:
        self.visit(node.test)
        body = self._loop_body(node.body)
        with self.swap_scope(body):
            # second pass for fixed point
            self.visit(node.test)
            body.merge(self._loop_body(node.body))
        orelse = self._body(node.orelse)
        self.scope.merge_if_else(body, orelse)
        return NoType(origin=self.origin())

    visit_Break: _VisitMethod = generic_statement
    visit_Continue: _VisitMethod = generic_statement

    def visit_Try(self, node: ast.Try) -> TypeInfo:
        self.scope.merge(self._body(node.body))
        for handler in node.handlers:
            self.push_scope()
            self.visit(handler)
            self.pop_scope_merge()
        self.scope.merge(self._body(node.orelse))
        self.scope.overwrite(self._body(node.finalbody))
        return NoType(origin=self.origin())

    visit_TryStar: _VisitMethod = visit_Try

    def _not_on_device_statement(self, node: ast.AST) -> TypeInfo:
        if self.device_loop_depth:
            raise exc.NotAllowedOnDevice(type(node).__name__)
        return NoType(origin=self.origin())

    visit_ExceptHandler: _VisitMethod = _not_on_device_statement
    visit_With: _VisitMethod = _not_on_device_statement
    visit_Return: _VisitMethod = _not_on_device_statement

    def _not_supported(self, node: ast.AST) -> TypeInfo:
        raise exc.StatementNotSupported(type(node).__name__)

    # TODO(jansel): support closure functions defined on host
    visit_FunctionDef: _VisitMethod = _not_supported

    visit_ClassDef: _VisitMethod = _not_supported
    visit_Yield: _VisitMethod = _not_supported
    visit_YieldFrom: _VisitMethod = _not_supported
    visit_AsyncFunctionDef: _VisitMethod = _not_supported
    visit_AsyncFor: _VisitMethod = _not_supported
    visit_AsyncWith: _VisitMethod = _not_supported
    visit_Await: _VisitMethod = _not_supported
    visit_Match: _VisitMethod = _not_supported
    visit_MatchValue: _VisitMethod = _not_supported
    visit_MatchSingleton: _VisitMethod = _not_supported
    visit_MatchSequence: _VisitMethod = _not_supported
    visit_MatchStar: _VisitMethod = _not_supported
    visit_MatchMapping: _VisitMethod = _not_supported
    visit_MatchClass: _VisitMethod = _not_supported
    visit_MatchAs: _VisitMethod = _not_supported
    visit_MatchOr: _VisitMethod = _not_supported


def propagate_types(
    func: HostFunction, args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    with func:
        global_scope = GlobalScope(function=func)
        local_scope = LocalScope(parent=global_scope)
        params = inspect.signature(func.fn).bind(*args, **kwargs)
        for name, value in params.arguments.items():
            # TODO(jansel): handle specializations/constexpr
            type_info = TypeInfo.from_example(
                value,
                ArgumentOrigin(name=name, function=func),
            )
            local_scope.set(name, type_info)
        closure = func.fn.__closure__ or ()
        assert len(closure) == len(func.fn.__code__.co_freevars)
        for name, cell in zip(func.fn.__code__.co_freevars, closure):
            # TODO(jansel): auto-specialize
            # TODO(jansel): ban closure writes
            value = cell.cell_contents
            type_info = TypeInfo.from_example(
                value,
                ClosureOrigin(name=name, function=func),
            )
            local_scope.set(name, type_info)
        prop = TypePropagation(func, local_scope)
        for stmt in func.body:
            prop.visit(stmt)

    # func.errors.raise_if_errors()
