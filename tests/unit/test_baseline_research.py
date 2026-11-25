"""Research-only baseline dataset and split guard regressions."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mnq_system.domain.values import Symbol
from mnq_system.features.engine import FeatureRow
from mnq_system.labels.engine import LabelRow, Outcome
from mnq_system.models.baselines import (
    APPROVED,
    _evaluate,  # pyright: ignore[reportPrivateUsage]
    align,
    choose_symbols,
    split_samples,
)

START = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def label(i: int, maturity: int) -> LabelRow:
    at = START + timedelta(minutes=i)
    outcome = Outcome(
        START + timedelta(minutes=maturity - 1),
        START + timedelta(minutes=maturity),
        Decimal("0.002"),
        None,
        None,
        "up",
        "time",
        START + timedelta(minutes=maturity - 1),
        maturity - i,
    )
    return LabelRow(at, at, at, outcome.maturity_timestamp, {15: outcome, 30: outcome}, {})


def test_approved_symbols_only() -> None:
    assert len(APPROVED) == 6
    assert choose_symbols(("SPY", "QQQ")) == (Symbol("SPY"), Symbol("QQQ"))
    with pytest.raises(ValueError, match="not approved"):
        choose_symbols(("SPY", "AMD"))
    with pytest.raises(ValueError, match="unique"):
        choose_symbols(("SPY", "SPY"))


def test_alignment_rejects_future_feature_and_drops_nulls() -> None:
    rows = (FeatureRow(START, START + timedelta(minutes=1), {"close": Decimal(100)}),)
    assert len(align((label(1, 32),), rows, ("close",), 15)) == 1
    with pytest.raises(ValueError, match="not yet available"):
        align((label(0, 32),), rows, ("close",), 15)
    nulls = (FeatureRow(START, START + timedelta(minutes=1), {"close": None}),)
    assert align((label(1, 32),), nulls, ("close",), 15) == ()


def test_split_uses_full_maturity_and_embargo(tmp_path: Path) -> None:
    del tmp_path
    rows = tuple(label(i, i + 2 if i != 4 else 9) for i in range(1, 15))
    split = split_samples(
        rows,
        START + timedelta(minutes=8),
        START + timedelta(minutes=12),
        timedelta(minutes=1),
    )
    assert label(4, 9) not in split.train
    assert all(r.maturity_timestamp <= START + timedelta(minutes=8) for r in split.train)
    assert all(r.prediction_timestamp >= START + timedelta(minutes=9) for r in split.validation)
    assert all(r.prediction_timestamp >= START + timedelta(minutes=13) for r in split.test)


def test_multiclass_metrics_and_neutral_no_trade() -> None:
    at = START + timedelta(minutes=1)
    samples = (
        ((1.0,), 0, -0.002, at, at + timedelta(minutes=15)),
        ((2.0,), 1, 0.0, at + timedelta(minutes=16), at + timedelta(minutes=31)),
        ((3.0,), 2, 0.002, at + timedelta(minutes=32), at + timedelta(minutes=47)),
    )
    perfect = _evaluate(samples, np.eye(3) * 0.997 + 0.001, 15)
    assert perfect["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert perfect["accuracy"] == 1.0
    assert cast("dict[str, object]", perfect["trading"])["trades"] == 2
    neutral = _evaluate(samples, np.tile([0.001, 0.998, 0.001], (3, 1)), 15)
    trading = cast("dict[str, object]", neutral["trading"])
    assert trading["trades"] == 0
    assert trading["profit_factor"] is None
