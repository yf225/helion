from __future__ import annotations

import dataclasses
import threading
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from helion import Config
    from helion import exc

    class _TLS(Protocol):
        default_settings: Settings | None


_tls: _TLS = cast("_TLS", threading.local())


def set_default_settings(settings: Settings) -> AbstractContextManager[None, None]:
    """
    Set the default settings for the current thread and return a context manager
    that restores the previous settings upon exit.

    :param settings: The Settings object to set as the default.
    :return: A context manager that restores the previous settings upon exit.
    """
    prior = getattr(_tls, "default_settings", None)
    _tls.default_settings = settings

    class _RestoreContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *args: object) -> None:
            _tls.default_settings = prior

    return _RestoreContext()


@dataclasses.dataclass
class _Settings:
    # see __slots__ below for the doc strings that show up in help(Settings)
    configs: list[Config | dict[str, object]] | None = None
    ignore_warnings: list[type[exc.BaseWarning]] = dataclasses.field(
        default_factory=list
    )


class Settings(_Settings):
    """
    Settings can be passed to hl.kernel as kwargs and control the behavior of the
    compilation process. Unlike a Config, settings are not auto-tuned and are
    expected to be set by the user.
    """

    __slots__: dict[str, str] = {
        "configs": "Optional list of Config objects to search in autotuning",
        "ignore_warnings": "Subtypes of exc.BaseWarning to ignore when compiling",
    }
    assert __slots__.keys() == {field.name for field in dataclasses.fields(_Settings)}

    def __init__(self, **settings: object) -> None:
        """
        Initialize the Settings object with the provided dictionary of settings.
        If no settings are provided, the default settings are used (see `set_default_settings`).

        :param settings: Keyword arguments representing various settings.
        """
        if config := settings.pop("config", None):
            assert "configs" not in settings, (
                "Cannot specify both 'config' and 'configs'"
            )
            settings["configs"] = [config]
        if defaults := getattr(_tls, "default_settings", None):
            settings = {**defaults.to_dict(), **settings}
        # pyre-ignore[6]
        super().__init__(**settings)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the Settings object to a dictionary.

        :return: A dictionary representation of the Settings object.
        """

        def shallow_copy(x: object) -> object:
            if isinstance(x, (list, dict)):
                return x.copy()
            return x

        return {k: shallow_copy(v) for k, v in dataclasses.asdict(self).items()}

    @staticmethod
    def default() -> Settings:
        """
        Get the default Settings object. If no default settings are set, create a new one.

        :return: The default Settings object.
        """
        result = getattr(_tls, "default_settings", None)
        if result is None:
            _tls.default_settings = result = Settings()
        return result
