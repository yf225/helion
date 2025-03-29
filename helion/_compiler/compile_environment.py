from __future__ import annotations

import threading
import typing
from typing import TYPE_CHECKING
from typing import Protocol

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensor
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .error_reporting import ErrorReporting

if TYPE_CHECKING:
    from types import TracebackType
    from typing_extensions import Self

    from torch._guards import Source

    from .. import exc
    from helion.runtime.settings import Settings

    class _TLS(Protocol):
        env: CompileEnvironment | None


tls: _TLS = typing.cast("_TLS", threading.local())


class CompileEnvironment:
    """
    Global state for the duration of a compilation.
    There is a 1:1 mapping between this an BoundKernel,
    and a single CompileEnvironment will be used for multiple Configs.
    No config or codegen specific state should be stored here.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        # TODO(jansel): self.device = ...
        self.settings = settings
        self.errors = ErrorReporting(settings)
        self.shape_env = ShapeEnv(
            specialize_zero_one=True,
            duck_shape=False,
            assume_static_by_default=False,
        )
        # TODO(jansel): check for guards in the shapeenv
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        self.input_sources: dict[FakeTensor, Source] = {}
        self.block_sizes: list[BlockSizeInfo] = []
        self.debug_shape_renames: dict[sympy.Expr, sympy.Expr] = {}

    def allocate_block_size(self, numel: int | torch.SymInt) -> int:
        idx = len(self.block_sizes)
        if isinstance(numel, torch.SymInt):
            numel_expr = numel._sympy_()
        else:
            numel_expr = sympy.sympify(numel)
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint()
            assert isinstance(sym._sympy_(), sympy.Symbol)
        self.block_sizes.append(
            info := BlockSizeInfo(
                block_size_idx=idx,
                numel=numel_expr,
                var=sym,
            )
        )
        self.debug_shape_renames[sym._sympy_()] = sympy.Symbol(
            info.name(), integer=True
        )
        return idx

    def to_fake(self, obj: object, source: Source) -> object:
        if isinstance(obj, torch.Tensor):
            return self._to_fake_tensor(obj, source)
        # TODO(jansel): support other types of args
        raise TypeError(f"unsupported argument type {type(obj)} ({source})")

    def _to_fake_tensor(self, tensor: torch.Tensor, source: Source) -> FakeTensor:
        assert CompileEnvironment.current() is self
        assert not self.fake_mode.is_our_fake(tensor)
        result = self.fake_mode.fake_tensor_converter.from_real_tensor(
            self.fake_mode, tensor, shape_env=self.shape_env, source=source
        )
        self.input_sources[result] = source
        if isinstance(source, LocalSource):
            for i, s in enumerate(result.size()):
                if isinstance(s, torch.SymInt) and isinstance(
                    s._sympy_(), sympy.Symbol
                ):
                    self.debug_shape_renames[s._sympy_()] = sympy.Symbol(
                        f"{source.local_name}_size{i}", integer=True
                    )
        return result

    def __enter__(self) -> Self:
        assert getattr(tls, "env", None) is None, "CompileEnvironment already active"
        self.fake_mode.__enter__()
        tls.env = self
        self.errors = ErrorReporting(self.settings)  # clear prior errors
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        tls.env = None
        self.fake_mode.__exit__(exc_type, exc_value, traceback)
        self.errors.raise_if_errors()

    @staticmethod
    def current() -> CompileEnvironment:
        try:
            if (env := tls.env) is not None:
                return env
        except AttributeError:
            pass
        raise NoCurrentEnvironment from None


class NoCurrentEnvironment(RuntimeError):
    pass


class BlockSizeInfo(typing.NamedTuple):
    """
    Information about a block size.
    Used to track the block size for a given dimension.
    """

    block_size_idx: int
    numel: sympy.Expr
    var: torch.SymInt

    def symbol(self) -> sympy.Symbol:
        return self.var._sympy_()

    def name(self) -> str:
        return f"block_size{self.block_size_idx}"


def warning(warning: exc.BaseWarning | type[exc.BaseWarning]) -> None:
    CompileEnvironment.current().errors.add(warning)
