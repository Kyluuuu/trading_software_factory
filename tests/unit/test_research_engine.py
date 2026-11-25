"""Offline replay regressions: no signal may see its execution candle."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from mnq_system.backtest.engine import Candle, ReplayEngine
from mnq_system.config.settings import Settings
from mnq_system.domain.orders import OrderSide
from mnq_system.domain.values import Symbol
from mnq_system.features.strategies import Breakout, MeanReversion, Signal
from mnq_system.risk.limits import AccountSnapshot, ConfiguredRiskPolicy, RiskLimitExceeded


class FixedStrategy:
    name = "fixed"
    history_limit = 1

    def generate(self, history: tuple[Candle, ...]) -> Signal | None:
        return Signal(OrderSide.BUY, Decimal("1")) if history else None


def candles(closes: list[str]) -> tuple[Candle, ...]:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    return tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(value),
            Decimal(value),
            Decimal(value),
            Decimal(value),
            100,
        )
        for i, value in enumerate(closes)
    )


def test_requested_limits_are_defaults_and_cannot_be_weakened() -> None:
    settings = Settings()
    assert (
        settings.maximum_risk_fraction,
        settings.maximum_daily_loss_fraction,
        settings.maximum_drawdown_fraction,
    ) == (Decimal("0.001"), Decimal("0.05"), Decimal("0.30"))
    for key, value in (
        ("maximum_risk_fraction", "0.002"),
        ("maximum_daily_loss_fraction", "0.06"),
        ("maximum_drawdown_fraction", "0.31"),
    ):
        with pytest.raises(ValidationError):
            Settings.model_validate({key: value})


def test_equity_and_stop_distance_determine_whole_share_size() -> None:
    policy = ConfiguredRiskPolicy(
        Settings(), AccountSnapshot(Decimal("10000"), Decimal("10000"), Decimal("10000"))
    )
    assert policy.size_quantity(Decimal("100"), Decimal("98")) == 5
    assert policy.size_quantity(Decimal("100"), Decimal("98"), allocation=Decimal("0.5")) == 2
    with pytest.raises(RiskLimitExceeded):
        policy.size_quantity(Decimal("100"), Decimal("100"))


def test_replay_uses_only_prior_candles_and_deducts_round_trip_costs() -> None:
    settings = Settings.model_validate(
        {"costs": {"commission_per_contract": "0.1", "slippage_points": "0.1"}}
    )
    result = ReplayEngine(settings).run(
        FixedStrategy(), candles(["100", "100", "100"]), Decimal("10000")
    )
    assert result.metrics.number_of_trades == 2
    assert result.metrics.net_profit == Decimal("-5.6")  # 7 shares each, 0.4 round trip
    assert result.metrics.win_rate == 0


def test_daily_halt_resets_next_session_but_total_drawdown_does_not() -> None:
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    bars = (
        Candle(
            Symbol("AAPL"),
            start,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            100,
        ),
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=1),
            Decimal("100"),
            Decimal("100"),
            Decimal("1"),
            Decimal("1"),
            100,
        ),
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=2),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            100,
        ),
        Candle(
            Symbol("AAPL"),
            start + timedelta(days=1),
            Decimal("100"),
            Decimal("100"),
            Decimal("99"),
            Decimal("99"),
            100,
        ),
    )
    daily = ReplayEngine(Settings()).run(FixedStrategy(), bars, Decimal("10000"))
    assert daily.metrics.number_of_trades == 2  # gap breaches daily limit; next session reopens
    assert daily.metrics.maximum_drawdown > Decimal("0.05")
    total_bars = (
        Candle(
            Symbol("AAPL"),
            start,
            Decimal("1000"),
            Decimal("1000"),
            Decimal("1000"),
            Decimal("1000"),
            100,
        ),
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=1),
            Decimal("1000"),
            Decimal("1000"),
            Decimal("1"),
            Decimal("1"),
            100,
        ),
        Candle(
            Symbol("AAPL"),
            start + timedelta(days=1),
            Decimal("1000"),
            Decimal("1000"),
            Decimal("1000"),
            Decimal("1000"),
            100,
        ),
    )
    total = ReplayEngine(Settings()).run(FixedStrategy(), total_bars, Decimal("10000"))
    assert total.total_halt is True
    assert total.metrics.number_of_trades == 1


def test_pure_strategy_signals_and_research_ranking() -> None:
    bars = candles(["100", "100", "110", "120"])
    assert MeanReversion(lookback=2).generate(bars[:1]) is None
    assert MeanReversion(lookback=2).generate(bars[:3]) == Signal(OrderSide.SELL, Decimal("1"))
    assert Breakout(lookback=2).generate(bars[:3]) == Signal(OrderSide.BUY, Decimal("1"))
    ranked = ReplayEngine(Settings()).compare(
        (FixedStrategy(), MeanReversion(lookback=2)), bars, Decimal("10000")
    )
    assert {item.result.name for item in ranked} == {"fixed", "mean_reversion"}
    assert sum((item.allocation for item in ranked), Decimal(0)) <= 1
    assert all(Decimal(0) <= item.allocation <= 1 for item in ranked)


def test_invalid_candle_order_and_duplicate_strategy_names_fail_closed() -> None:
    engine = ReplayEngine(Settings())
    with pytest.raises(ValueError, match="increasing"):
        engine.run(FixedStrategy(), candles(["100", "100"])[::-1], Decimal("10000"))
    with pytest.raises(ValueError, match="unique"):
        engine.compare(
            (FixedStrategy(), FixedStrategy()), candles(["100", "100"]), Decimal("10000")
        )
