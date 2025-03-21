from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def grid(sizes: int | Sequence[int]) -> None:
    pass


def tile(sizes: int | Sequence[int]) -> None:
    pass


def masked_block(pid: int, block_size: int, count: int) -> slice:
    start = pid * block_size
    end = min(start + block_size, count)
    return slice(start, end)
