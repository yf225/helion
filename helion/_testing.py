from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import types


def import_path(filename: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{__name__}.{filename.stem}", filename
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
