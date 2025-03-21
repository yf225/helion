from __future__ import annotations

import dataclasses
import threading
import typing
from typing import TYPE_CHECKING
from typing import Protocol

from torch._subclasses import FakeTensor
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .error_reporting import ErrorReporting

if TYPE_CHECKING:
    from types import TracebackType
    from typing_extensions import Self

    import torch
    from torch._guards import Source

    from .. import exc

    class _TLS(Protocol):
        env: CompileEnvironment | None


tls: _TLS = typing.cast("_TLS", threading.local())


@dataclasses.dataclass
class Settings:
    pass


class CompileEnvironment:
    def __init__(self) -> None:
        self.errors = ErrorReporting()
        self.shape_env = ShapeEnv(
            specialize_zero_one=True,
            duck_shape=False,
            assume_static_by_default=False,
        )
        # TODO(jansel): check for guards in the shapeenv
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        self.input_sources: dict[FakeTensor, Source] = {}

    def to_fake(self, tensor: torch.Tensor, source: Source) -> FakeTensor:
        assert CompileEnvironment.current() is self
        assert not self.fake_mode.is_our_fake(tensor)
        result = self.fake_mode.fake_tensor_converter.from_real_tensor(
            self.fake_mode, tensor, shape_env=self.shape_env, source=source
        )
        self.input_sources[result] = source
        return result

    def __enter__(self) -> Self:
        assert getattr(tls, "env", None) is None
        self.fake_mode.__enter__()
        tls.env = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        tls.env = None
        self.fake_mode.__exit__(exc_type, exc_value, traceback)

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


def warning(warning: exc.BaseWarning | type[exc.BaseWarning]) -> None:
    CompileEnvironment.current().errors.add(warning)
