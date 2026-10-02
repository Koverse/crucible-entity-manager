from datetime import UTC, datetime, timedelta
from typing import ClassVar

import numpy as np
import pytest

from crucible_entity_manager.components.fusion import (
    DEFAULT_PROCESS_NOISE,
    POSITION_VARIANCE,
    ComponentMeasurement,
    FusedIdentity,
    Fusion,
    FusionParams,
    PrincipalFilter,
    PrincipalState,
    component_measurement,
    principal_record,
    state_fields,
)
from crucible_entity_manager.components.keyed import KeyedState
from crucible_entity_manager.core.aliases import JSONObject, JSONValue
from crucible_entity_manager.core.kalman import CiObjective, GaussianState, covariance_intersection

T0 = datetime(2026, 9, 30, tzinfo=UTC)
POSITION: JSONObject = {"x": 1_000_000.0, "y": 2_000_000.0, "z": 3_000_000.0}


def component(**extra: JSONValue) -> JSONObject:
    return {"ecefPosition": dict(POSITION), **extra}


def make_filter(
    fusion: Fusion = Fusion.KALMAN, omega: float | None = None
) -> tuple[PrincipalFilter, KeyedState[PrincipalState]]:
    principals: KeyedState[PrincipalState] = KeyedState(PrincipalState, idle_seconds=None)
    return PrincipalFilter(FusionParams(fusion, omega), principals, "[test] "), principals


def measurement(
    seconds: float, x: float = 1_000_000.0, variance: float = 100.0
) -> ComponentMeasurement:
    values = np.array([x, 1.0, 2_000_000.0, 0.0, 3_000_000.0, 0.0])
    return ComponentMeasurement(values, np.eye(6) * variance, T0 + timedelta(seconds=seconds))


class TestComponentMeasurement:
    def test_missing_velocity_is_zero_with_the_environment_variance(self) -> None:
        result = component_measurement(component(), T0, 9.0)
        assert result is not None
        np.testing.assert_array_equal(
            result.values, [1_000_000.0, 0.0, 2_000_000.0, 0.0, 3_000_000.0, 0.0]
        )
        np.testing.assert_array_equal(
            np.diag(result.noise),
            [POSITION_VARIANCE, 9.0, POSITION_VARIANCE, 9.0, POSITION_VARIANCE, 9.0],
        )

    def test_covariance_terms(self) -> None:
        result = component_measurement(
            component(
                ecefVelocity={"x": 1.0, "y": "2", "z": None},
                positionCovariance={"xx": 4.0, "xy": 1.0, "yz": 0.5},
                velocityCovariance={"dxdx": 2.0, "dydz": 0.25},
                positionVelocityCovariance={"xdx": 99.0},
            ),
            T0,
            25.0,
        )
        assert result is not None
        np.testing.assert_array_equal(result.values[[1, 3, 5]], [1.0, 2.0, 0.0])
        assert result.noise[0, 0] == 4.0
        assert result.noise[0, 2] == result.noise[2, 0] == 1.0
        assert result.noise[2, 4] == result.noise[4, 2] == 0.5
        assert result.noise[1, 1] == 2.0
        assert result.noise[3, 5] == result.noise[5, 3] == 0.25
        assert result.noise[0, 1] == 0.0

    @pytest.mark.parametrize("position", [{"x": 1.0, "y": 2.0}, {"x": "a", "y": 2.0, "z": 3.0}])
    def test_an_incomplete_position_gives_none(self, position: JSONObject) -> None:
        assert component_measurement({"ecefPosition": position}, T0, 25.0) is None


class TestPrincipalFilter:
    def test_a_new_principal_starts_and_updates_with_its_first_measurement(self) -> None:
        fusion, principals = make_filter()
        posterior = fusion.apply("p", measurement(0))
        assert posterior is not None
        np.testing.assert_allclose(posterior.covariance, np.eye(6) * 50.0)
        assert principals.get("p").process_noise == DEFAULT_PROCESS_NOISE

    def test_updates_reduce_uncertainty_and_track_the_truth(self) -> None:
        fusion, _ = make_filter()
        rng = np.random.default_rng(3)
        errors, traces = [], []
        for step in range(40):
            truth = 1_000_000.0 + step * 1.0
            noisy = truth + rng.normal(0.0, 10.0)
            posterior = fusion.apply("p", measurement(step, x=noisy))
            assert posterior is not None
            errors.append((posterior.mean[0] - truth, noisy - truth))
            traces.append(np.trace(posterior.covariance))
        filtered = np.sqrt(np.mean([error**2 for error, _ in errors[10:]]))
        raw = np.sqrt(np.mean([error**2 for _, error in errors[10:]]))
        assert filtered < raw
        assert traces[-1] < traces[0]

    def test_a_stale_component_changes_nothing(self) -> None:
        fusion, principals = make_filter()
        fusion.apply("p", measurement(10))
        before = principals.get("p").prior
        assert fusion.apply("p", measurement(5)) is None
        assert principals.get("p").prior is before

    def test_a_long_gap_resets_to_the_measurement(self) -> None:
        fusion, _ = make_filter()
        fusion.apply("p", measurement(0))
        late = measurement(16 * 60, x=1_000_500.0)
        posterior = fusion.apply("p", late)
        assert posterior is not None
        np.testing.assert_array_equal(posterior.mean, late.values)
        np.testing.assert_array_equal(posterior.covariance, late.noise)

    @pytest.mark.parametrize(
        "prior",
        [
            GaussianState(np.full(6, np.nan), np.eye(6), T0),
            GaussianState(np.zeros(6), np.diag([np.inf] * 6), T0),
            GaussianState(np.zeros(6), np.eye(6) * 1e16, T0),
        ],
    )
    def test_numerical_trouble_resets_to_the_measurement(self, prior: GaussianState) -> None:
        fusion, principals = make_filter()
        principals.get("p").prior = prior
        principals.get("p").process_noise = 1.0
        update = measurement(1)
        posterior = fusion.apply("p", update)
        assert posterior is not None
        np.testing.assert_array_equal(posterior.mean, update.values)

    def test_a_singular_update_resets(self) -> None:
        fusion, principals = make_filter()
        principals.get("p").prior = GaussianState(np.zeros(6), np.zeros((6, 6)), T0)
        principals.get("p").process_noise = 0.0
        exact = ComponentMeasurement(np.ones(6), np.zeros((6, 6)), T0)
        posterior = fusion.apply("p", exact)
        assert posterior is not None
        np.testing.assert_array_equal(posterior.mean, np.ones(6))

    def test_principals_are_independent(self) -> None:
        fusion, principals = make_filter()
        fusion.apply("a", measurement(0, x=1.0))
        fusion.apply("b", measurement(0, x=2.0))
        first, second = principals.get("a").prior, principals.get("b").prior
        assert first is not None
        assert second is not None
        assert first.mean[0] != second.mean[0]

    def test_environment_selects_noise_and_the_latest_wins(self) -> None:
        fusion, principals = make_filter()
        fusion.note_environment("p", "SEA_SURFACE")
        fusion.note_environment("p", None)
        fusion.note_environment("p", "AIR")
        assert fusion.velocity_variance("p") == 25.0
        fusion.apply("p", measurement(0))
        assert principals.get("p").process_noise == 5.0
        fusion.note_environment("q", "GROUND")
        assert fusion.velocity_variance("q") == 9.0
        assert fusion.velocity_variance("unknown") == 25.0

    def test_restore_uses_the_default_noise(self) -> None:
        fusion, principals = make_filter()
        fusion.restore("p", component(), T0)
        state = principals.get("p")
        assert state.process_noise == DEFAULT_PROCESS_NOISE
        assert state.prior is not None
        assert state.prior.covariance[1, 1] == 25.0
        fusion.restore("q", {"ecefPosition": {"x": 1.0}}, T0)
        assert principals.peek("q") is None

    def test_forget(self) -> None:
        fusion, principals = make_filter()
        fusion.apply("p", measurement(0))
        fusion.forget("p")
        assert principals.peek("p") is None


class TestCovarianceIntersection:
    @pytest.mark.parametrize(
        ("omega", "used"), [(0.3, 0.3), (0.0, 0.01), (1.5, 0.99), (None, None)]
    )
    def test_matches_core_with_the_position_trace_objective(
        self, omega: float | None, used: float | None
    ) -> None:
        fusion, principals = make_filter(Fusion.COVARIANCE_INTERSECTION, omega)
        prior = GaussianState(np.zeros(6), np.eye(6) * 400.0, T0)
        principals.get("p").prior = prior
        principals.get("p").process_noise = 0.5
        update = measurement(0)
        posterior = fusion.apply("p", update)
        expected = covariance_intersection(
            prior.mean,
            prior.covariance,
            update.values,
            update.noise,
            objective=CiObjective.POSITION_TRACE,
            omega=used,
        )
        assert posterior is not None
        assert expected is not None
        np.testing.assert_allclose(posterior.mean, expected[0])
        np.testing.assert_allclose(posterior.covariance, expected[1])

    def test_falls_back_to_kalman_when_singular(self) -> None:
        fusion, principals = make_filter(Fusion.COVARIANCE_INTERSECTION)
        principals.get("p").prior = GaussianState(np.zeros(6), np.zeros((6, 6)), T0)
        principals.get("p").process_noise = 0.0
        posterior = fusion.apply("p", measurement(0))
        assert posterior is not None
        np.testing.assert_array_equal(posterior.covariance, np.zeros((6, 6)))

    def test_is_more_conservative_than_kalman(self) -> None:
        kalman, _ = make_filter(Fusion.KALMAN)
        intersection, _ = make_filter(Fusion.COVARIANCE_INTERSECTION)
        for step in range(30):
            kalman.apply("p", measurement(step))
            intersection.apply("p", measurement(step))
        kalman_state = kalman._principals.get("p").prior
        ci_state = intersection._principals.get("p").prior
        assert kalman_state is not None
        assert ci_state is not None
        assert np.trace(ci_state.covariance) >= np.trace(kalman_state.covariance) * 0.99


class TestFusedIdentity:
    def test_a_superseded_source_beats_a_later_survivor(self) -> None:
        fused = FusedIdentity()
        fused.merge({"callsign": "OLD"}, superseded=True)
        fused.merge({"callsign": "NEW"}, superseded=False)
        assert fused.values() == {"callsign": "OLD"}

    def test_a_superseded_source_overrides_an_earlier_survivor(self) -> None:
        fused = FusedIdentity()
        fused.merge({"callsign": "ROOT"}, superseded=False)
        fused.merge({"callsign": "DUP"}, superseded=True)
        assert fused.values() == {"callsign": "DUP"}

    def test_at_equal_rank_the_newest_wins_and_fields_union(self) -> None:
        fused = FusedIdentity()
        fused.merge({"callsign": "A", "mmsi": 1}, superseded=False)
        fused.merge({"callsign": "B", "name": "x"}, superseded=False)
        assert fused.values() == {"callsign": "B", "mmsi": 1, "name": "x"}

    def test_empty_values_and_non_objects_are_ignored(self) -> None:
        fused = FusedIdentity()
        fused.merge({"callsign": "A"}, superseded=False)
        fused.merge({"callsign": None, "mmsi": ""}, superseded=True)
        fused.merge("not an object", superseded=True)
        assert fused.values() == {"callsign": "A"}

    def test_values_are_copies(self) -> None:
        fused = FusedIdentity()
        fused.merge({"tags": ["a"]}, superseded=False)
        tags = fused.values()["tags"]
        assert isinstance(tags, list)
        tags.append("b")
        assert fused.values() == {"tags": ["a"]}


class TestRecords:
    SOURCE: ClassVar[JSONObject] = {
        "trackId": "c",
        "standardIdentity": "FRIEND",
        "environment": "AIR",
        "trackOriginatedTimestamp": "O",
        "edhControlSet": {"classification": "U"},
        "mode": "LIVE",
        "trackQuality": 0.5,
        "interceptTimestamp": "T",
        "geodetic": {"latitude": 0.1},
        "ecefPosition": dict(POSITION),
        "identity": {"callsign": "A"},
        "speed": 3.0,
        "heading": 1.0,
        "reportIds": ["r"],
    }

    def test_event(self) -> None:
        event = principal_record(self.SOURCE, "p", head=False, stale="S")
        assert event["trackId"] == "p"
        assert (event["trackQuality"], event["interceptTimestamp"]) == (0.5, "T")
        assert "stale" not in event
        assert "reportIds" not in event
        for key in ("geodetic", "ecefPosition", "identity", "speed", "heading", "edhControlSet"):
            assert event[key] == self.SOURCE[key]
        assert event["identity"] is not self.SOURCE["identity"]

    def test_head(self) -> None:
        head = principal_record(self.SOURCE, "p", head=True, stale="S")
        assert (head["trackUpdatedTimestamp"], head["stale"]) == ("T", "S")
        assert "trackQuality" not in head
        assert "interceptTimestamp" not in head

    def test_state_fields(self) -> None:
        covariance = np.arange(36.0).reshape(6, 6)
        fields = state_fields(GaussianState(np.arange(6.0), covariance, T0))
        assert fields["ecefPosition"] == {"x": 0.0, "y": 2.0, "z": 4.0}
        assert fields["ecefVelocity"] == {"x": 1.0, "y": 3.0, "z": 5.0}
        assert fields["velocityCovariance"] == {
            "dxdx": 7.0,
            "dxdy": 9.0,
            "dxdz": 11.0,
            "dydy": 21.0,
            "dydz": 23.0,
            "dzdz": 35.0,
        }
        assert "positionVelocityCovariance" not in fields
