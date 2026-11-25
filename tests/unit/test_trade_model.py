"""A rolling trade-outcome model must learn only from closed historical trades."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mnq_system.backtest.engine import Candle, ReplayEngine
from mnq_system.backtest.walkforward import evaluate
from mnq_system.config.settings import Settings
from mnq_system.domain.orders import OrderSide
from mnq_system.domain.values import Symbol
from mnq_system.features.strategies import Signal, default_strategies
from mnq_system.models.trade_outcomes import Observation, Regime, fit, regime_at


def test_model_retrains_from_outcomes_and_gates_unknown_regimes() -> None:
    up = Regime.UP
    samples = tuple(Observation("trend", up, Decimal("1")) for _ in range(4))
    first = fit(samples, min_samples=3)
    assert first.approve("trend", up)
    assert not first.approve("breakout", up)
    assert not first.approve("trend", Regime.DOWN)
    second = fit(
        samples + tuple(Observation("trend", up, Decimal("-2")) for _ in range(6)), min_samples=3
    )
    assert not second.approve("trend", up)
    assert first.approve("trend", up)  # frozen snapshot unaffected by retraining
    with pytest.raises(ValueError, match="samples"):
        fit(samples, min_samples=0)


def test_regime_uses_only_completed_prior_bars() -> None:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    bars = tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(p),
            Decimal(p),
            Decimal(p),
            Decimal(p),
            100,
        )
        for i, p in enumerate((100, 101, 102, 50))
    )
    assert regime_at(bars, 3) is Regime.UP
    assert regime_at(bars, 2) is Regime.UP
    assert regime_at(bars, 1) is None


class AlwaysLong:
    name = "always_long"
    history_limit = 1

    def generate(self, history: tuple[Candle, ...]) -> Signal | None:
        return Signal(OrderSide.BUY, Decimal("1")) if history else None


def test_learned_pattern_gates_real_test_entries() -> None:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    bars = tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(100 + i),
            Decimal(101 + i),
            Decimal(100 + i),
            Decimal(101 + i),
            100,
        )
        for i in range(16)
    )
    windows = evaluate(
        ReplayEngine(Settings()), (AlwaysLong(),), bars, Decimal("10000"), train_bars=8, test_bars=4
    )
    assert windows[0].model_training_trades >= 3
    assert ("always_long", "up") in windows[0].model_approved
    assert windows[0].model_results[0].metrics.number_of_trades > 0
    assert all(t.entry_time >= bars[8].timestamp for t in windows[0].model_trades)


def test_walkforward_model_never_trains_on_test_trades() -> None:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    bars = tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(100 + i),
            Decimal(100 + i),
            Decimal(100 + i),
            Decimal(100 + i),
            100,
        )
        for i in range(24)
    )
    engine = ReplayEngine(Settings())
    windows = evaluate(
        engine, default_strategies(), bars, Decimal("10000"), train_bars=8, test_bars=5
    )
    assert len(windows) == 3
    for window in windows:
        assert window.model_training_trades >= 0
        assert all(
            trade.entry_time >= bars[window.test_start].timestamp for trade in window.model_trades
        )
        assert window.model_training_end == bars[window.train_end - 1].timestamp
    modified = bars[:8] + tuple(
        Candle(
            b.symbol, b.timestamp, b.open + 100, b.high + 100, b.low + 100, b.close + 100, b.volume
        )
        for b in bars[8:]
    )
    changed = evaluate(
        engine, default_strategies(), modified, Decimal("10000"), train_bars=8, test_bars=5
    )
    assert windows[0].model_training_trades == changed[0].model_training_trades
    assert windows[0].model_approved == changed[0].model_approved
