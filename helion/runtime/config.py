from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from typing import Literal
from typing import cast

from helion.autotuner.config_spec import DEFAULT_NUM_STAGES
from helion.autotuner.config_spec import DEFAULT_NUM_WARPS

IndexingLiteral = Literal["pointer", "tensor_descriptor", "block_ptr"]


class Config(Mapping[str, object]):
    config: dict[str, object]

    def __init__(self, config: object = None, **kwargs: object) -> None:
        if config is not None:
            assert not kwargs
            assert isinstance(config, (dict, Config))
            self.config: dict[str, object] = {**config}
        else:
            self.config: dict[str, object] = kwargs

    def __getitem__(self, key: str) -> object:
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def __repr__(self) -> str:
        args = [f"{key}={value!r}" for key, value in self.config.items()]
        return f"Config({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.config == other.config

    def __hash__(self) -> int:
        return hash(frozenset(self.config.items()))

    @property
    def block_sizes(self) -> list[int | list[int]]:
        return cast("list[int | list[int]]", self.config["block_sizes"])

    @property
    def loop_orders(self) -> list[list[int]]:
        return cast("list[list[int]]", self.config.get("loop_orders", []))

    @property
    def num_warps(self) -> int:
        return cast("int", self.config.get("num_warps", DEFAULT_NUM_WARPS))

    @property
    def num_stages(self) -> int:
        return cast("int", self.config.get("num_stages", DEFAULT_NUM_STAGES))

    @property
    def l2_grouping(self) -> int:
        return cast("int", self.config.get("l2_grouping", 1))

    @property
    def use_yz_grid(self) -> int:
        return cast("bool", self.config.get("use_yz_grid", False))

    @property
    def indexing(self) -> IndexingLiteral:
        return cast("IndexingLiteral", self.config.get("indexing", "pointer"))
