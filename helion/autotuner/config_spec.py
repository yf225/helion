from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeGuard

from ..exc import InvalidConfig
import helion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3


@dataclasses.dataclass
class ConfigSpec:
    block_size_specs: list[BlockSizeSpec] = dataclasses.field(default_factory=list)

    def loop_order_specs(self) -> Sequence[PermutationSpec]:
        """Return the specs for the loop orders."""
        return [
            PermutationSpec(len(spec))
            for spec in self.block_size_specs
            if spec.allow_reorder
        ]

    def normalize(self, config: Config | dict[str, object]) -> None:
        """Normalize the config to match the block_sizes and validate the config."""
        if isinstance(config, helion.Config):
            self.normalize(config.config)
            return

        for name in ("block_size", "loop_order"):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                config[names] = [config.pop(name)]

        config["block_sizes"] = self.normalize_block_sizes(
            config.get("block_sizes", None)
        )
        config["loop_orders"] = self.normalize_loop_orders(
            config.get("loop_orders", None)
        )
        config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        config.setdefault("num_stages", DEFAULT_NUM_STAGES)
        # TODO(jansel): include num_ctas and max_nreg

        if self.block_size_specs:
            if self.block_size_specs[0].allow_l2_grouping:
                config.setdefault("l2_grouping", 1)
            if 1 < len(self.block_size_specs[0]) <= 3:
                config.setdefault("use_yz_grid", False)

        config.setdefault("indexing", "pointer")

    def normalize_block_sizes(self, block_sizes: object) -> list[int | list[int]]:
        if len(self.block_size_specs) == 0:
            if block_sizes:
                raise InvalidConfig("block_sizes should be empty")
            return []
        if not block_sizes or not isinstance(block_sizes, (list, tuple)):
            raise InvalidConfig("block_sizes must be set to a list")
        idx = 0
        new_block_sizes: list[int | list[int]] = []
        for block_spec in self.block_size_specs:
            expected = len(block_spec)
            if idx >= len(block_sizes):
                raise InvalidConfig(
                    f"Not enough block sizes, expected {expected}, got {len(block_sizes)}"
                )
            val = block_sizes[idx]
            if (
                expected > 1
                and len(block_sizes[idx:]) == expected
                and block_spec is self.block_size_specs[-1]
            ):
                new_block_sizes.append(
                    [*map(assert_integer_power_of_two, block_sizes[idx:])]
                )
                idx += expected
            elif isinstance(val, int):
                if len(block_spec) == 1:
                    # go down the more general NDTileStrategy path
                    new_block_sizes.append([assert_integer_power_of_two(val)])
                else:
                    if not block_spec.can_be_int():
                        raise InvalidConfig(f"Block sizes must be list, got {val!r}")
                    new_block_sizes.append(assert_integer_power_of_two(val))
                idx += 1
            elif isinstance(val, (list, tuple)):
                if len(val) != expected:
                    raise InvalidConfig(f"Block size {idx} length {expected}: {val!r}")
                new_block_sizes.append([*map(assert_integer_power_of_two, val)])
                idx += 1
            else:
                raise InvalidConfig(f"Block size must be int/list, got {val!r}")
        if len(block_sizes) != idx:
            raise InvalidConfig(f"Extra block sizes, used {idx} of {len(block_sizes)}")
        return new_block_sizes

    def normalize_loop_orders(self, loop_orders: object) -> list[list[int]]:
        assert isinstance(loop_orders, (list, tuple, type(None)))
        loop_orders = [*(loop_orders or ())]
        specs = self.loop_order_specs()
        if len(loop_orders) > len(specs):
            raise InvalidConfig(
                f"Too many loop orders, expected {len(specs)}, got {len(loop_orders)}"
            )
        for i, spec in enumerate(specs):
            if i < len(loop_orders):
                loop_orders[i] = spec.normalize(loop_orders[i])
            else:
                loop_orders.append(spec.default())
        return loop_orders


def integer_power_of_two(n: object) -> TypeGuard[int]:
    return isinstance(n, int) and n != 0 and (n & (n - 1)) == 0


def assert_integer_power_of_two(n: object) -> int:
    if integer_power_of_two(n):
        return n
    raise InvalidConfig(f"Expected integer power of two, got {n}")


@dataclasses.dataclass
class BlockSizeSpec:
    size_hints: list[int]
    # TODO(jansel): need to flip this false if tl.dot is used
    allow_flattened: bool
    allow_reorder: bool
    allow_l2_grouping: bool

    def can_be_int(self) -> bool:
        return len(self.size_hints) == 1 or self.allow_flattened

    def __len__(self) -> int:
        return len(self.size_hints)


class PermutationSpec(NamedTuple):
    length: int

    def normalize(self, ordering: object) -> list[int]:
        if type(ordering) is not list:
            if not isinstance(ordering, tuple):
                raise InvalidConfig(f"ordering must be a list: {ordering!r}")
            ordering = [*ordering]
        if len(ordering) != self.length:
            raise InvalidConfig(
                f"Expected {self.length} permutations, got {len(ordering)}"
            )
        if {*ordering} != {*range(self.length)}:
            raise InvalidConfig(f"Invalid permutation {ordering!r}")
        return ordering

    def default(self) -> list[int]:
        return [*range(self.length)]
