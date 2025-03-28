from __future__ import annotations

from collections import defaultdict
import functools
import re
import sys
from typing import TYPE_CHECKING
from typing import Callable

from .. import exc
from ..exc import Base
from ..exc import BaseError
from ..exc import BaseWarning
from ..exc import ErrorCompilingKernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .source_location import SourceLocation
    from .type_propagation import TypeNotAllowedOnDevice
    from helion.runtime.settings import Settings

    ErrorOrWarning = BaseError | BaseWarning


# pyre-ignore[9]
order_by_location: Callable[[Sequence[ErrorOrWarning]], list[ErrorOrWarning]] = (
    functools.partial(sorted, key=lambda e: e.location)
)


class ErrorReporting:
    def __init__(self, settings: Settings) -> None:
        self.errors: list[BaseError] = []
        self.warnings: list[BaseWarning] = []
        self.ignores: tuple[type[BaseWarning], ...] = tuple(settings.ignore_warnings)
        self.type_errors: dict[SourceLocation, list[exc.TypePropagationError]] = (
            defaultdict(list)
        )

    def add(self, e: Base | type[Base]) -> None:
        if callable(e):
            e = e()
        if isinstance(e, BaseError):
            self.errors.append(e)
        elif isinstance(e, BaseWarning):
            if not isinstance(e, self.ignores):
                self.warnings.append(e)
        else:
            raise TypeError(f"expected error or warning, got {type(e)}")

    def add_type_error(self, type_info: TypeNotAllowedOnDevice) -> None:
        locations = type_info.locations
        similar_errors = self.type_errors[locations[0]]
        similar_errors.append(e := exc.TypePropagationError(type_info, similar_errors))
        if len(similar_errors) == 1:
            self.add(e)

    def ignore(self, e: type[BaseWarning]) -> None:
        self.ignores = (*self.ignores, e)

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
