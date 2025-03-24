from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._compiler.source_location import SourceLocation
    from ._compiler.type_propagation import TypeNotAllowedOnDevice


class Base(RuntimeError):
    def report(self) -> str:
        raise NotImplementedError


class _FixedMessage(Base):
    message = ""
    location_suffix = "\nWhile processing:\n{location}"

    def __init__(self, *args: object, **kwargs: object) -> None:
        from ._compiler.source_location import current_location

        self.location: SourceLocation = current_location()
        msg = self.__class__.message.format(*args, **kwargs)
        self.base_msg_len: int = len(msg)
        if self.location:
            msg += self.location_suffix.format(location=self.location.format())
        super().__init__(msg)


class BaseError(_FixedMessage):
    message = "An error occurred."

    def report(self) -> str:
        return f"ERROR[{type(self).__name__}]: {self!s}"


class NotInsideKernel(BaseError):
    message = (
        "Functions found in helion.language.* must be called from inside a kernel. "
        "Did you forget the @helion.jit decorator?"
    )


class LoopFunctionNotInFor(BaseError):
    message = "{0} must be called from a for loop, e.g. `for ... in {0}(...):"


class DeviceLoopElseBlock(BaseError):
    message = "for...else block is not allowed in a {0} device loop."


class MultipleDeviceLoops(BaseError):
    message = "Multiple grid loops are not allowed. Support for this may be added in the future."


class NestedGridLoop(BaseError):
    message = "Grid loops must be at the top level of a function."


class RankMismatch(BaseError):
    message = "Expected rank {0} tensor, but got rank {1} tensor."


class InvalidIndexingType(BaseError):
    message = "Expected tile/int/None/etc in tensor[...], got {0!s}."


class RequiresTensorInAssignment(BaseError):
    message = "Expected tensor in right-hand side of assignment, got {0!s}."


class NotAllowedOnDevice(BaseError):
    message = "The statement {} is not allowed inside the `hl.tile` or `hl.grid` loop."


class AssignmentMultipleTargets(NotAllowedOnDevice):
    message = "Assignment with multiple targets (a=b=1) is not allowed inside the `hl.tile` or `hl.grid` loop."


class InvalidAssignment(NotAllowedOnDevice):
    message = "Assignment target must be Name or Subscript inside the `hl.tile` or `hl.grid` loop."


class NonTensorSubscriptAssign(BaseError):
    message = "Expected tensor in subscript assignment, got {0!s} and {1!s}."


class ShapeMismatch(BaseError):
    message = "Shape mismatch between {0!s} and {1!s}."


class DeviceAPIOnHost(BaseError):
    message = "{} is only allowed inside the `hl.tile` or `hl.grid` loop."


class StatementNotSupported(BaseError):
    message = "The statement {} is not supported."


class CantReadOnDevice(BaseError):
    message = "Can not read {0!s} inside the `hl.tile` or `hl.grid` loop."


class ExpectedTensorName(BaseError):
    message = "Expected tensor name, got {0!s}."


class TypePropagationError(BaseError):
    message = "{}"

    def __init__(
        self,
        type_info: TypeNotAllowedOnDevice,
        similar_errors: list[TypePropagationError],
    ) -> None:
        super(_FixedMessage, self).__init__(str(type_info))
        self.locations: list[SourceLocation] = [
            *dict.fromkeys([*type_info.locations, self.location])
        ]
        self.similar_errors = similar_errors

    def __str__(self) -> str:
        msg = super().__str__()[: self.base_msg_len]
        if len(self.similar_errors) > 1:
            msg += f"\n({len(self.similar_errors) - 1} similar errors suppressed)"
        msg += self.location_suffix.format(
            location="\n".join(loc.format() for loc in self.locations)
        )
        return msg


class ErrorCompilingKernel(BaseError):
    message = "{0} errors and {1} warnings occurred (see above)"


class _WrapException(BaseError):
    message = "{name}: {msg}"

    def __init__(self, e: Exception) -> None:
        super().__init__(name=type(e).__name__, msg=str(e))


class InternalError(_WrapException):
    pass


class TorchOpTracingError(_WrapException):
    pass


class BaseWarning(_FixedMessage):
    message = "A warning occurred."

    def report(self) -> str:
        return f"WARNING[{type(self).__name__}]: {self!s}"


class TensorOperationInWrapper(BaseWarning):
    message = (
        "A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused "
        "in the generated kernel."
    )


class TensorOperationsInHostCall(TensorOperationInWrapper):
    message = (
        "A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused "
        "in the generated kernel: {}"
    )
