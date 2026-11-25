"""Rolling selection and model refits must use only earlier windows."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mnq_system.backtest.engine import Candle, ReplayEngine
from mnq_system.backtest.walkforward import evaluate
from mnq_system.config.settings import Settings
from mnq_system.domain.values import Symbol
from mnq_system.features.strategies import default_strategies


def bars(count: int) -> tuple[Candle, ...]:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    return tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(100 + (i % 5)),
            Decimal(101 + (i % 5)),
            Decimal(99 + (i % 5)),
            Decimal(100 + (i % 5)),
            100 + i,
        )
        for i in range(count)
    )


def test_catalog_contains_distinct_causal_strategies() -> None:
    strategies = default_strategies()
    names = [s.name for s in strategies]
    assert len(names) >= 7
    assert len(names) == len(set(names))
    for strategy in strategies:
        assert strategy.history_limit > 0
        assert strategy.generate(bars(1)) is None


def test_walkforward_trains_on_prior_bars_only() -> None:
    result = evaluate(
        ReplayEngine(Settings()),
        default_strategies(),
        bars(20),
        Decimal("10000"),
        train_bars=8,
        test_bars=5,
    )
    assert len(result) == 2
    assert result[0].train_end == 8
    assert result[0].test_start == 8
    assert result[0].test_end == 13
    assert result[1].train_end == 13
    assert all(item.test_start == item.train_end for item in result)
    assert all(
        sum((fraction for _, fraction in item.allocations), Decimal(0)) <= 1 for item in result
    )
    for item in result:
        eligible = [name for name, allocation in item.allocations if allocation > 0]
        if eligible:
            assert item.selected_strategy == eligible[0]
            assert item.selected_result is not None
            assert item.selected_result.name == eligible[0]
        else:
            assert item.selected_strategy is None
            assert item.selected_result is None


def test_invalid_walkforward_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        evaluate(
            ReplayEngine(Settings()),
            default_strategies(),
            bars(10),
            Decimal("10000"),
            train_bars=0,
            test_bars=2,
        )
