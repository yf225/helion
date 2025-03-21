from __future__ import annotations

import functools
import re
import sys
from typing import TYPE_CHECKING
from typing import Callable

from ..exc import Base
from ..exc import BaseError
from ..exc import BaseWarning
from ..exc import ErrorCompilingKernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    ErrorOrWarning = BaseError | BaseWarning


# pyre-ignore[9]
order_by_location: Callable[[Sequence[ErrorOrWarning]], list[ErrorOrWarning]] = (
    functools.partial(sorted, key=lambda e: e.location)
)


class ErrorReporting:
    def __init__(self) -> None:
        self.errors: list[BaseError] = []
        self.warnings: list[BaseWarning] = []
        self.ignores: dict[type[BaseWarning], bool] = {}

    def add(self, e: Base | type[Base]) -> None:
        if callable(e):
            e = e()
        if isinstance(e, BaseError):
            breakpoint()
            self.errors.append(e)
        elif isinstance(e, BaseWarning):
            if not self.ignores.get(type(e)):
                self.warnings.append(e)
        else:
            raise TypeError(f"expected error or warning, got {type(e)}")

    def ignore(self, e: type[BaseWarning]) -> None:
        self.ignores[e] = True

    def raise_if_errors(self) -> None:
        sys.stderr.write(self.report())
        if self.errors:
            if len(self.errors) > 1:
                raise ErrorCompilingKernel(len(self.errors), len(self.warnings))
            raise self.errors[0]

    def report(self, strip_paths: bool = False) -> str:
        report = "\n\n".join(
            {
                e.report(): None
                for e in order_by_location([*self.errors, *self.warnings])
            }
        )
        if strip_paths:
            report = re.sub(r'"[/\\][^"]*[/\\]', '".../', report)
        return report
