from __future__ import annotations

import inspect
from types import FunctionType
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generic
from typing import Literal
from typing import Never
from typing import Protocol
from typing import TypeGuard
from typing import TypeVar
from typing import cast

import torch

if TYPE_CHECKING:
    import ast

    from helion._compiler.generate_ast import CodegenState
    from helion._compiler.type_propagation import TypeInfo

    _T = TypeVar("_T")
    _C = TypeVar("_C", bound=Callable[..., object])

    class _Decorator(Protocol):
        def __call__(self, fn: _C) -> _C: ...

    class _NoReturnDecorator(Protocol, Generic[_T]):
        def __call__(self, fn: Callable[..., _T]) -> object: ...


class APIFunc(Protocol):
    __qualname__: str
    _helion_api: Literal[True]
    # a device loop can transition between host and device code
    _is_device_loop: bool
    _is_device_only: bool
    _tiles_as_sizes: bool
    _type_function: Callable[..., TypeInfo] | None
    _codegen: Callable[[CodegenState], ast.AST] | None
    _register_fake: Callable[..., None] | None
    _signature: inspect.Signature

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _no_call(*args: object, **kwargs: object) -> Never:
    raise TypeError("type_prop/codegen functions cannot be called directly")


def is_api_func(fn: object) -> TypeGuard[APIFunc]:
    return getattr(fn, "_helion_api", False)


def api(
    *,
    is_device_loop: bool = False,
    is_device_only: bool = True,
    tiles_as_sizes: bool = False,
    signature: inspect.Signature | None = None,
) -> _Decorator:
    def _impl(fn: _C) -> _C:
        api = cast("APIFunc", fn)
        api._helion_api = True
        api._is_device_loop = is_device_loop
        api._is_device_only = is_device_only
        api._tiles_as_sizes = tiles_as_sizes
        api._type_function = None
        api._codegen = None
        api._register_fake = None
        api._signature = signature or inspect.signature(
            cast("Callable[..., object]", fn)
        )
        return fn

    return _impl


def type_propagation(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[TypeInfo]:
    def _impl(type_fn: Callable[..., TypeInfo]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        assert original_fn._type_function is None, (
            "type_prop can only be used once per function"
        )
        original_fn._type_function = type_fn
        return _no_call

    return _impl


def codegen(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[ast.AST]:
    def _impl(codegen_fn: Callable[[CodegenState], ast.AST]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        assert original_fn._codegen is None, (
            "codegen can only be used once per function"
        )
        original_fn._codegen = codegen_fn
        return _no_call

    return _impl


def api_custom_op(**kwargs: bool) -> _Decorator:
    def _impl(fn: _C) -> _C:
        assert isinstance(fn, FunctionType)
        op_def = torch.library.custom_op(f"helion::{fn.__name__}", mutates_args=())(fn)
        overload = api(**kwargs, signature=inspect.signature(fn))(op_def._opoverload)
        assert is_api_func(overload)
        overload._register_fake = op_def.register_fake
        return overload

    return _impl


def register_fake(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(fake_fn: Callable[..., object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{register_fake.__qualname__} can only be used on API functions"
        )
        assert original_fn._register_fake is not None
        original_fn._register_fake(fake_fn)
        return _no_call

    return _impl
