import json
from typing import TYPE_CHECKING

import pytest

from crucible_entity_manager.components.keyed import (
    APPLIED_INPUTS_PER_KEY,
    AppliedInputs,
    KeyedState,
    input_fingerprint,
)

if TYPE_CHECKING:
    from crucible_entity_manager.core.aliases import JSONObject


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestKeyedState:
    def test_creates_state_on_first_use(self) -> None:
        state: KeyedState[list[int]] = KeyedState(list)
        state.get("a").append(1)
        assert state.get("a") == [1]
        assert state.peek("b") is None
        assert "a" in state
        assert "b" not in state
        assert len(state) == 1

    def test_the_least_recently_used_key_is_evicted_past_the_cap(self) -> None:
        evicted: list[tuple[str, list[int]]] = []
        state: KeyedState[list[int]] = KeyedState(
            list, max_keys=2, on_evict=lambda key, value: evicted.append((key, value))
        )
        state.get("a").append(1)
        state.get("b")
        state.get("a")
        state.peek("b")
        state.get("c")
        assert list(state) == ["a", "c"]
        assert evicted == [("b", [])]

    def test_idle_keys_expire_oldest_first(self) -> None:
        clock = Clock()
        evicted: list[str] = []
        state: KeyedState[int] = KeyedState(
            int, idle_seconds=10.0, on_evict=lambda key, value: evicted.append(key), clock=clock
        )
        state.get("a")
        clock.now = 5.0
        state.get("b")
        clock.now = 10.0
        assert state.expire() == 1
        assert evicted == ["a"]
        clock.now = 100.0
        assert state.expire() == 1
        assert list(state) == []

    def test_expiry_can_be_disabled(self) -> None:
        clock = Clock()
        state: KeyedState[int] = KeyedState(int, idle_seconds=None, clock=clock)
        state.get("a")
        clock.now = 1e9
        assert state.expire() == 0
        assert "a" in state

    def test_pop_is_not_an_eviction(self) -> None:
        evicted: list[str] = []
        state: KeyedState[int] = KeyedState(int, on_evict=lambda key, value: evicted.append(key))
        state.get("a")
        assert state.pop("a") == 0
        assert state.pop("a") is None
        assert evicted == []

    def test_rejects_a_cap_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            KeyedState(int, max_keys=0)


class TestFingerprint:
    def test_ignores_the_crucible_header_and_key_order(self) -> None:
        first: JSONObject = {"a": 1.5, "b": {"c": [1, 2]}, "crucibleHeader": {"uuid": "x"}}
        second: JSONObject = {"b": {"c": [1, 2]}, "a": 1.5, "crucibleHeader": {"uuid": "y"}}
        assert input_fingerprint(first) == input_fingerprint(second)

    def test_any_content_change_changes_it(self) -> None:
        assert input_fingerprint({"a": 1.5}) != input_fingerprint({"a": 1.5000000000000002})
        assert input_fingerprint({"a": 1}) != input_fingerprint({"a": "1"})

    def test_survives_a_json_round_trip(self) -> None:
        record: JSONObject = {"x": 0.1 + 0.2, "y": [1e-320, -0.0], "z": "é"}
        assert input_fingerprint(json.loads(json.dumps(record))) == input_fingerprint(record)


def test_applied_inputs_keep_a_bounded_window() -> None:
    applied = AppliedInputs()
    for index in range(APPLIED_INPUTS_PER_KEY + 1):
        applied.add(("key", str(index)))
    applied.add(("key", "1"))
    assert ("key", "0") not in applied
    assert ("key", "1") in applied
    assert ("key", str(APPLIED_INPUTS_PER_KEY)) in applied
