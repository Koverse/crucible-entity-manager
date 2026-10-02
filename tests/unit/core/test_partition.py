import pytest

from crucible_entity_manager.core.partition import PartitionSpec, stable_shard

KEYS = [f"track-{index}" for index in range(2000)]


class TestStableShard:
    # Routing must not change between releases, or keys move between partitions.
    def test_known_assignments_are_stable(self) -> None:
        assert stable_shard("track-a", 7) == 5
        assert stable_shard("abc", 1000) == 570

    def test_results_are_in_range(self) -> None:
        assert {stable_shard(key, 4) for key in KEYS} == {0, 1, 2, 3}

    def test_single_shard_takes_everything(self) -> None:
        assert {stable_shard(key, 1) for key in KEYS} == {0}

    @pytest.mark.parametrize("count", [0, -1])
    def test_rejects_non_positive_counts(self, count: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            stable_shard("key", count)


class TestPartitionSpec:
    def test_partitions_are_disjoint_and_cover_every_key(self) -> None:
        partitions = [PartitionSpec(index, 3) for index in range(3)]
        for key in KEYS:
            assert sum(partition.owns(key) for partition in partitions) == 1

    def test_single_partition_owns_every_key(self) -> None:
        assert all(PartitionSpec.single().owns(key) for key in KEYS)

    @pytest.mark.parametrize(("index", "count"), [(0, 0), (-1, 2), (2, 2)])
    def test_rejects_invalid_specs(self, index: int, count: int) -> None:
        with pytest.raises(ValueError, match="partition"):
            PartitionSpec(index, count)

    def test_renders_one_based(self) -> None:
        assert str(PartitionSpec(0, 4)) == "p0/4"
