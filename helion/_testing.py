from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch

from .runtime.config import Config

if TYPE_CHECKING:
    from pathlib import Path
    import types

    from .runtime.kernel import Kernel


DEVICE = torch.device("cuda")


def import_path(filename: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{__name__}.{filename.stem}", filename
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def code_and_output(
    # pyre-ignore[11]
    fn: Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> tuple[str, object]:
    config = Config(**kwargs)
    code = fn.bind(args).to_triton_code(config)
    compiled_kernel = fn.bind(args).compile_config(config)
    return code, compiled_kernel(*args)
