"""Deterministic assignment of keys to partitions."""

import hashlib
from dataclasses import dataclass
from typing import Self, override


def stable_shard(key: str, count: int) -> int:
    """Map `key` to a shard in ``range(count)``, identically in every process.

    Uses MD5 rather than ``hash()``, which varies with ``PYTHONHASHSEED``.

    Raises:
        ValueError: If `count` is less than 1.
    """
    if count < 1:
        msg = f"shard count must be at least 1, got {count}"
        raise ValueError(msg)
    digest = hashlib.md5(key.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest) % count


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """One partition out of `count`, identified by its zero-based `index`."""

    index: int
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            msg = f"partition count must be at least 1, got {self.count}"
            raise ValueError(msg)
        if not 0 <= self.index < self.count:
            msg = f"partition index must be in [0, {self.count}), got {self.index}"
            raise ValueError(msg)

    @classmethod
    def single(cls) -> Self:
        """Return the only partition of an unpartitioned component."""
        return cls(index=0, count=1)

    def owns(self, key: str) -> bool:
        """Return whether `key` belongs to this partition."""
        return stable_shard(key, self.count) == self.index

    @override
    def __str__(self) -> str:
        """``p<index>/<count>``, matching ``--partition`` and the Execution name suffix."""
        return f"p{self.index}/{self.count}"
