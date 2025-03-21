from __future__ import annotations


class Base(RuntimeError):
    def report(self) -> str:
        raise NotImplementedError


class _FixedMessage(Base):
    message = ""
    location_suffix = "\nWhile processing:\n{location}"

    def __init__(self, *args: object, **kwargs: object) -> None:
        from ._compiler.source_location import SourceLocation
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


class NotAllowedOnDevice(BaseError):
    message = "The statement {} is not allowed inside the `hl.tile` or `hl.grid` loop."


class StatementNotSupported(BaseError):
    message = "The statement {} is not supported."


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
