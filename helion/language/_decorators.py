from __future__ import annotations

import functools
import inspect
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
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map_only

from helion import exc
from helion._compiler.compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.type_propagation import TypeInfo
    from helion._compiler.variable_origin import Origin

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
    _fake_fn: Callable[..., object] | None
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
        api._fake_fn = None
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


def api_custom_op(*, tiles_as_sizes: bool = False, **kwargs: bool) -> _Decorator:
    def _impl(fn: _C) -> _C:
        # pyre-fixme[6]
        @api(**kwargs, tiles_as_sizes=tiles_as_sizes, signature=inspect.signature(fn))
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            """
            We hit type errors if we use the regular overload, here we
            intercept the call and fake the custom op.
            """
            from helion._compiler.tile_index_proxy import TileIndexProxy

            mode = proxy_tensor.get_proxy_mode()
            if mode is None:
                if CompileEnvironment.has_current():
                    if tiles_as_sizes:
                        args, kwargs = TileIndexProxy.tiles_to_sizes((args, kwargs))
                    return wrapper._fake_fn(*args, **kwargs)
                return fn(*args, **kwargs)
            assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
            tracer = mode.tracer
            with proxy_tensor.disable_proxy_modes_tracing():
                if tiles_as_sizes:
                    args, kwargs = TileIndexProxy.tiles_to_sizes((args, kwargs))
                proxy_args, proxy_kwargs = tree_map_only(
                    proxy_tensor._ProxyTensor,
                    lambda x: x.proxy,
                    tree_map_only(
                        (torch.Tensor, torch.SymInt, torch.SymBool, torch.SymFloat),
                        functools.partial(proxy_tensor.get_proxy_slot, tracer=tracer),
                        (args, kwargs),
                    ),
                )
                proxy_out = tracer.create_proxy(
                    "call_function",
                    wrapper,
                    proxy_args,
                    proxy_kwargs,
                )
                assert wrapper._fake_fn is not None
                out = wrapper._fake_fn(*args, **kwargs)
                if out is not None:
                    proxy_tensor.track_tensor_tree(
                        out, proxy_out, constant=None, tracer=tracer
                    )
            return out

        return wrapper

    return _impl


def register_fake(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(fake_fn: Callable[..., object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{register_fake.__qualname__} can only be used on API functions"
        )
        assert original_fn._fake_fn is None
        original_fn._fake_fn = fake_fn
        if original_fn._type_function is None:
            original_fn._type_function = _default_type_function(
                fake_fn, original_fn._tiles_as_sizes
            )
        return _no_call

    return _impl


def _default_type_function(
    fake_fn: Callable[..., object], tiles_as_sizes: bool
) -> Callable[..., TypeInfo]:
    from .._compiler.tile_index_proxy import TileIndexProxy
    from .._compiler.type_propagation import TypeInfo

    def type_prop_with_fake_fn(
        *args: object, origin: Origin, **kwargs: object
    ) -> TypeInfo:
        args, kwargs = tree_map_only(TypeInfo, _to_proxy, (args, kwargs))
        if tiles_as_sizes:
            args, kwargs = TileIndexProxy.tiles_to_sizes((args, kwargs))
        return TypeInfo.from_example(fake_fn(*args, **kwargs), origin)

    return type_prop_with_fake_fn


def _to_proxy(arg: TypeInfo) -> object:
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None
