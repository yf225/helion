from __future__ import annotations

import ast
import functools
import inspect
from typing import TYPE_CHECKING
from typing import Callable
from typing import overload

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import PyCodeCache

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.generate_ast import OUTPUT_CODE_HEADER
from .._compiler.generate_ast import generate_ast
from .._compiler.host_function import HostFunction
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence
    import types

    from .config import Config


class Kernel:
    def __init__(self, fn: types.FunctionType, settings: Settings | None) -> None:
        """
        Initialize the Kernel object.  This is typically called from the `@helion.kernel` decorator.

        :param fn: The function to be compiled as a Helion kernel.
        :param settings: The settings to be used by the Kernel. If None, default settings are used.
        """
        super().__init__()
        self.name: str = fn.__name__
        self.fn = fn
        self.signature: inspect.Signature = inspect.signature(fn)
        self.settings: Settings = settings or Settings.default()
        # pyre-fixme[11]: BoundKernel undefined?
        self.bound_kernels: dict[Hashable, BoundKernel] = {}
        if any(
            param.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for param in self.signature.parameters.values()
        ):
            raise TypeError(
                f"Kernel({self.name}) cannot have *args, **kwargs, or keyword-only arguments"
            )

    def bind(self, args: tuple[object, ...]) -> BoundKernel:
        """
        Bind the given arguments to the Kernel and return a BoundKernel object.

        :param args: The arguments to bind to the Kernel.
        :return: A BoundKernel object with the given arguments bound.
        """
        if not isinstance(args, tuple):
            assert isinstance(args, list), "args must be a tuple or list"
            args = tuple(args)
        signature = _specialization_key(args)
        bound_kernel = self.bound_kernels.get(signature)
        if bound_kernel is None:
            normalized_args: tuple[object, ...] = self.normalize_args(*args)
            if len(normalized_args) != len(args):
                # we had default args that needed to be applied
                bound_kernel = self.bind(normalized_args)
            else:
                bound_kernel = BoundKernel(self, args)
            self.bound_kernels[signature] = bound_kernel
        return bound_kernel

    def normalize_args(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        """
        Normalize the given arguments and keyword arguments according to the function signature.

        :param args: The positional arguments to normalize.
        :param kwargs: The keyword arguments to normalize.
        :return: A tuple of normalized positional arguments.
        """
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return tuple(bound_args.args)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """
        Call the Kernel with the given arguments and keyword arguments.

        :param args: The positional arguments to pass to the Kernel.
        :param kwargs: The keyword arguments to pass to the Kernel.
        :return: The result of the Kernel function call.
        """
        if kwargs:
            args = self.normalize_args(*args, **kwargs)
        return self.bind(args)(*args)


class BoundKernel:
    # pyre-fixme[11]: Kernel undefined?
    def __init__(self, kernel: Kernel, args: tuple[object, ...]) -> None:
        super().__init__()
        self.kernel = kernel
        self.env = CompileEnvironment(self.kernel.settings)
        with self.env:
            assert len(args) == len(self.kernel.signature.parameters)
            self.fake_args: list[object] = [
                # TODO(jansel): support hl.constexpr
                self.env.to_fake(arg, LocalSource(name))
                for name, arg in zip(self.kernel.signature.parameters, args)
            ]
            self.host_fn: HostFunction = HostFunction(self.kernel.fn, self.fake_args)

    def to_triton_code(self, config: Config) -> str:
        with self.env:
            return OUTPUT_CODE_HEADER + ast.unparse(generate_ast(self.host_fn, config))

    def compile_config(self, config: Config) -> Callable[..., object]:
        module = PyCodeCache.load(self.to_triton_code(config))
        return getattr(module, self.kernel.name)

    def _debug_types(self) -> str:
        with self.env:
            return self.host_fn.debug_types()

    def __call__(self, *args: object) -> object:
        raise NotImplementedError


def _specialization_key(obj: object) -> Hashable:
    try:
        extractor = _specialization_extractors[type(obj)]
    except KeyError:
        raise TypeError(f"unsupported argument type: {type(obj).__name__}") from None
    return extractor(obj)


def _tensor_key(obj: torch.Tensor) -> Hashable:
    return (
        obj.dtype,
        obj.device,
        # 0, 1, or >=2 specialization
        tuple([min(s, 2) for s in obj.size()]),
        # TODO(jansel): add a way to disable this one
        obj.is_contiguous(),
    )


def _sequence_key(obj: Sequence) -> Hashable:
    return type(obj), tuple([_specialization_key(item) for item in obj])


_specialization_extractors: dict[type[object], Callable[[object], Hashable]] = {
    torch.Tensor: _tensor_key,
    torch.nn.Parameter: _tensor_key,
    torch.dtype: lambda x: x,
    torch.device: lambda x: x,
    int: lambda x: int,
    float: lambda x: float,
    bool: lambda x: bool,
    str: lambda x: str,
    list: _sequence_key,
    tuple: _sequence_key,
    dict: lambda x: tuple(sorted((k, _specialization_key(v)) for k, v in x.items())),
}


@overload
def kernel(fn: Callable[..., object], **settings: object) -> Kernel: ...


@overload
def kernel(
    fn: None = None, **settings: object
) -> Callable[[Callable[..., object]], Kernel]: ...


def kernel(fn: Callable[..., object] | None = None, **settings: object) -> object:
    """
    Decorator to create a Kernel object from a Python function.

    :param fn: The function to be wrapped by the Kernel. If None, a decorator is returned.
    :param settings: Keyword arguments representing settings for the Kernel.
                    Can also use settings=Settings(...) to pass a Settings object directly.
    :return: A Kernel object or a decorator that returns a Kernel object.
    """
    if fn is None:
        return functools.partial(kernel, **settings)
    if settings_obj := settings.pop("settings", None):
        assert len(settings) == 0, "settings must be the only keyword argument"
        assert isinstance(settings_obj, Settings), "settings must be a Settings object"
    else:
        settings_obj = Settings(**settings)
    return Kernel(fn, settings_obj)
