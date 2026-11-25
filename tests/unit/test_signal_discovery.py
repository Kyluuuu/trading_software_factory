"""Offline signal-discovery invariants."""

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnq_system.models import signal_discovery as discovery
from mnq_system.models.baselines import APPROVED, Sample
from mnq_system.models.signal_discovery import (
    GROUPS,
    ablation_columns,
    feature_statistics,
    group_columns,
)


def test_catalog_groups_are_disjoint_and_complete() -> None:
    from mnq_system.features.metadata import DEFINITIONS

    names = [name for members in GROUPS.values() for name in members]
    assert len(names) == len(set(names))
    assert set(names) == {definition.name for definition in DEFINITIONS}
    assert group_columns(("open", "minute_of_day", "returns")) == {
        "time": ("minute_of_day",),
        "price": ("open", "returns"),
        "volatility": (),
        "mean_reversion": (),
        "trend": (),
        "volume": (),
        "breakout": (),
        "market_structure": (),
    }


def test_ablation_requires_remaining_columns() -> None:
    assert ablation_columns(("open", "volume", "hour"), "time") == ("open", "volume")
    with pytest.raises(ValueError, match="no remaining"):
        ablation_columns(("open",), "price")
    with pytest.raises(ValueError, match="no admissible"):
        ablation_columns(("open",), "trend")


def test_univariate_statistics_handle_constant_column() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([0, 1, 2, 0])
    returns = np.array([-0.02, 0.01, 0.02, -0.01])
    result = feature_statistics(x, y, returns, ("open", "hour"), sample_limit=4)
    assert result["open"]["ic"] is None
    assert result["open"]["rank_ic"] is None
    assert result["open"]["mutual_information"] == 0.0
    assert result["hour"]["ic"] is not None


def test_group_ablation_keeps_same_rows_without_fitting_new_model_types() -> None:
    at = datetime(2026, 1, 5, tzinfo=UTC)
    train: list[Sample] = [
        (
            (float(i), float(i % 5), float(i % 3)),
            i % 3,
            float(i % 3) / 100,
            at + timedelta(minutes=i),
            at + timedelta(minutes=i + 1),
        )
        for i in range(60)
    ]
    test = train[:12]

    class Estimator:
        def fit(self, X: np.ndarray, y: np.ndarray) -> object:
            assert len(X) == len(train)
            assert len(y) == len(train)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            assert len(X) == len(test)
            return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(X), 1))

    with patch.object(discovery, "_models", return_value={"baseline": Estimator()}):
        result = discovery.evaluate_groups(
            train, test, ("open", "volume", "hour"), {"baseline": {"accuracy": 0.4}}
        )
    variants = cast("dict[str, object]", result["variants"])
    assert set(variants) == {
        "only_time",
        "only_price",
        "only_volume",
        "without_time",
        "without_price",
        "without_volume",
    }
    assert result["not_testable_groups"] == [
        "volatility",
        "mean_reversion",
        "trend",
        "breakout",
        "market_structure",
    ]


def _phase4_fixture() -> dict[str, object]:
    return {
        "symbols": sorted(APPROVED),
        "start": "2025-11-25T00:00:00+00:00",
        "end": "2026-11-25T00:00:00+00:00",
        "validation_start": "2026-08-25T00:00:00+00:00",
        "test_start": "2026-10-19T00:00:00+00:00",
        "embargo_minutes": 60.0,
        "horizons": [15, 30],
        "results": {
            str(h): {
                "across_symbols": {
                    "features": ["open"],
                    "counts": {"train": 6, "test": 6},
                    "logistic_regression": {"test": {"accuracy": 0.4}},
                },
                "by_symbol": {symbol: {"counts": {"train": 1, "test": 1}} for symbol in APPROVED},
            }
            for h in (15, 30)
        },
    }


def test_runner_rejects_nonapproved_and_mismatched_phase4_report() -> None:
    phase4 = _phase4_fixture()
    with (
        patch.object(discovery, "load_symbol", return_value=((), (), ("open",))),
        pytest.raises(ValueError, match="symbol split mismatch"),
    ):
        discovery.run(phase4, MagicMock(), MagicMock(), MagicMock())
    phase4["symbols"] = ["AMD", *sorted(APPROVED)[1:]]
    with pytest.raises(ValueError, match="not approved"):
        discovery.run(phase4, MagicMock(), MagicMock(), MagicMock())
    phase4["symbols"] = sorted(APPROVED)[:-1]
    with pytest.raises(ValueError, match="all six"):
        discovery.run(phase4, MagicMock(), MagicMock(), MagicMock())


def test_runner_assembles_exact_synthetic_cohort_and_checks_pooled_counts() -> None:
    phase4 = _phase4_fixture()
    at = datetime(2026, 1, 5, tzinfo=UTC)
    sample: Sample = ((1.0,), 2, 0.01, at, at + timedelta(minutes=15))
    split = MagicMock(train=(sample,), test=(sample,))
    with (
        patch.object(discovery, "load_symbol", return_value=((), (), ("open",))),
        patch.object(discovery, "split_samples", return_value=split),
        patch.object(discovery, "align", return_value=(sample,)),
        patch.object(discovery, "feature_statistics", return_value={"open": {"ic": 0.1}}),
        patch.object(discovery, "_models", return_value={"logistic_regression": MagicMock()}),
        patch.object(discovery, "evaluate_groups", return_value={"variants": {}}),
    ):
        report = discovery.run(phase4, MagicMock(), MagicMock(), MagicMock())
        assert cast("dict[str, object]", report["pooled"])["15"] == {"variants": {}}
        counts = cast("dict[str, object]", cast("dict[str, object]", phase4["results"])["15"])
        pooled = cast("dict[str, object]", counts["across_symbols"])
        cast("dict[str, int]", pooled["counts"])["test"] = 7
        with pytest.raises(ValueError, match="pooled split mismatch"):
            discovery.run(phase4, MagicMock(), MagicMock(), MagicMock())
