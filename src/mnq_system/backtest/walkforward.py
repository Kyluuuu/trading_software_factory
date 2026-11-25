"""Offline walk-forward selection: rank past windows, report subsequent independent replays."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import pairwise

from mnq_system.backtest.engine import Candle, ReplayEngine, Result, Trade
from mnq_system.features.strategies import Strategy
from mnq_system.models.trade_outcomes import Observation, OutcomeModel, fit, regime_at


@dataclass(frozen=True, slots=True)
class Window:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    allocations: tuple[tuple[str, Decimal], ...]
    results: tuple[Result, ...]
    selected_strategy: str | None
    selected_result: Result | None
    model_training_end: datetime
    model_training_trades: int
    model_approved: tuple[tuple[str, str], ...]
    model_trades: tuple[Trade, ...]
    model_results: tuple[Result, ...]


def evaluate(
    engine: ReplayEngine,
    strategies: tuple[Strategy, ...],
    candles: tuple[Candle, ...],
    equity: Decimal,
    *,
    train_bars: int = 7800,
    test_bars: int = 1950,
) -> tuple[Window, ...]:
    """Rolling comparisons plus supervised refits, not a shared portfolio.

    Training scores use only [train_start, train_end); scores then determine
    recommendations for [test_start, test_end). Results are independent risk
    snapshots and cannot be summed to imply an executable portfolio P&L.
    """
    if train_bars < 2 or test_bars < 1 or len(candles) < train_bars + test_bars:
        raise ValueError("window sizes require at least one full training and test window")
    if any(c.symbol != candles[0].symbol for c in candles) or any(
        a.timestamp >= b.timestamp for a, b in pairwise(candles)
    ):
        raise ValueError("candles must be one symbol and strictly ordered")
    if not strategies or len({s.name for s in strategies}) != len(strategies):
        raise ValueError("strategies must have unique names")
    windows: list[Window] = []
    for test_start in range(train_bars, len(candles) - test_bars + 1, test_bars):
        train_start = test_start - train_bars
        test_end = test_start + test_bars
        training = candles[train_start:test_start]
        ranked = engine.compare(strategies, training, equity)
        indices = {bar.timestamp: index for index, bar in enumerate(training)}
        observations = tuple(
            Observation(item.result.name, regime, trade.net_profit)
            for item in ranked
            for trade in item.result.trades
            for regime in (regime_at(training, indices[trade.entry_time]),)
            if regime is not None
        )
        model = fit(observations)
        approved = tuple(
            (estimate.strategy, estimate.regime.value)
            for estimate in model.estimates
            if model.approve(estimate.strategy, estimate.regime)
        )
        allocations = tuple((item.result.name, item.allocation) for item in ranked)
        results = tuple(
            engine.run(strategy, candles[train_start:test_end], equity, trading_start=train_bars)
            for strategy in strategies
        )
        selected = next((name for name, weight in allocations if weight > 0), None)
        selected_result = next((result for result in results if result.name == selected), None)
        evaluation = candles[train_start:test_end]

        def gate(
            name: str, history: tuple[Candle, ...], index: int, *, trained: OutcomeModel = model
        ) -> bool:
            return trained.approve(name, regime_at(history, index))

        model_results = tuple(
            engine.run(strategy, evaluation, equity, trading_start=train_bars, entry_gate=gate)
            for strategy in strategies
        )
        windows.append(
            Window(
                train_start,
                test_start,
                test_start,
                test_end,
                allocations,
                results,
                selected,
                selected_result,
                training[-1].timestamp,
                len(observations),
                approved,
                tuple(trade for result in model_results for trade in result.trades),
                model_results,
            )
        )
    return tuple(windows)
