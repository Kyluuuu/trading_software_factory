"""Offline, one-bar hold candle replay. Not an execution or HFT simulator."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from math import sqrt
from zoneinfo import ZoneInfo

from mnq_system.config.settings import Settings
from mnq_system.domain.orders import OrderIntent, OrderSide
from mnq_system.domain.values import ContractQuantity, Price, Symbol, UtcTimestamp
from mnq_system.features.strategies import Strategy
from mnq_system.risk.limits import AccountSnapshot, ConfiguredRiskPolicy, RiskLimitExceeded


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: Symbol
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("candle timestamp must be timezone aware")
        if (
            any(
                not value.is_finite() or value <= 0
                for value in (self.open, self.high, self.low, self.close)
            )
            or self.low > min(self.open, self.close)
            or self.high < max(self.open, self.close)
            or self.high < self.low
            or type(self.volume) is not int
            or self.volume < 0
        ):
            raise ValueError("invalid OHLCV candle")


@dataclass(frozen=True, slots=True)
class Trade:
    strategy: str
    entry_time: datetime
    exit_time: datetime
    quantity: int
    net_profit: Decimal
    return_fraction: Decimal


@dataclass(frozen=True, slots=True)
class Metrics:
    net_profit: Decimal
    profit_factor: Decimal | None
    win_rate: Decimal
    sharpe_ratio: float | None
    maximum_drawdown: Decimal
    average_trade_return: Decimal
    number_of_trades: int


@dataclass(frozen=True, slots=True)
class Result:
    name: str
    trades: tuple[Trade, ...]
    metrics: Metrics
    daily_halt: bool
    total_halt: bool


@dataclass(frozen=True, slots=True)
class RankedResult:
    result: Result
    allocation: Decimal


class ReplayEngine:
    """Independent per-strategy comparison; allocations are recommendations only.

    Signals see completed history through bar i-1; fills use bar i open and
    one-bar exits at bar i close, with adverse stop/gap handling. No same-bar
    high/low ordering inference, partial fills, liquidity or portfolio netting.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        strategy: Strategy,
        candles: tuple[Candle, ...],
        initial_equity: Decimal,
        trading_start: int = 1,
        entry_gate: Callable[[str, tuple[Candle, ...], int], bool] | None = None,
    ) -> Result:
        if trading_start < 1 or trading_start >= len(candles):
            raise ValueError("trading start must be within candles")
        if strategy.history_limit < 1:
            raise ValueError("strategy history limit must be positive")
        if not initial_equity.is_finite() or initial_equity <= 0:
            raise ValueError("initial equity must be positive and finite")
        if not strategy.name or not candles:
            raise ValueError("strategy name and candles are required")
        if any(c.symbol != candles[0].symbol for c in candles) or any(
            left.timestamp >= right.timestamp for left, right in pairwise(candles)
        ):
            raise ValueError("candles must contain one symbol in strictly increasing order")
        equity = opening = peak = initial_equity
        session_date = None
        daily_halt = total_halt = False
        trades: list[Trade] = []
        curve = [initial_equity]
        tz = ZoneInfo(self.settings.session.timezone)
        for i in range(trading_start, len(candles)):
            bar = candles[i]
            local_date = bar.timestamp.astimezone(tz).date()
            if local_date != session_date:
                session_date, opening, daily_halt = local_date, equity, False
            if total_halt:
                continue
            if opening - equity >= opening * self.settings.maximum_daily_loss_fraction:
                daily_halt = True
            if peak - equity >= peak * self.settings.maximum_drawdown_fraction:
                total_halt = True
            if daily_halt or total_halt:
                continue
            signal = strategy.generate(candles[max(0, i - strategy.history_limit) : i])
            if signal is None or (
                entry_gate is not None and not entry_gate(strategy.name, candles, i)
            ):
                continue
            entry = bar.open
            stop = (
                entry - signal.stop_distance
                if signal.side is OrderSide.BUY
                else entry + signal.stop_distance
            )
            if stop <= 0:
                continue
            snapshot = AccountSnapshot(equity, opening, peak)
            policy = ConfiguredRiskPolicy(self.settings, snapshot)
            quantity = policy.size_quantity(entry, stop)
            if quantity == 0:
                continue
            intent = OrderIntent(
                bar.symbol,
                signal.side,
                ContractQuantity(quantity),
                Price(entry),
                UtcTimestamp(bar.timestamp),
                Price(stop),
            )
            try:
                policy.assess(intent)
            except RiskLimitExceeded:
                continue
            # Stops may gap; conservative worst available close/stop proxy, never a guaranteed fill.
            exit_price = (
                min(bar.close, stop)
                if signal.side is OrderSide.BUY and bar.low <= stop
                else bar.close
            )
            if signal.side is OrderSide.SELL and bar.high >= stop:
                exit_price = max(bar.close, stop)
            change = (exit_price - entry) if signal.side is OrderSide.BUY else (entry - exit_price)
            cost = Decimal(2) * (
                self.settings.costs.commission_per_contract + self.settings.costs.slippage_points
            )
            pnl = (change - cost) * quantity
            trades.append(
                Trade(strategy.name, bar.timestamp, bar.timestamp, quantity, pnl, pnl / equity)
            )
            equity += pnl
            curve.append(equity)
            peak = max(peak, equity)
            if opening - equity >= opening * self.settings.maximum_daily_loss_fraction:
                daily_halt = True
            if peak - equity >= peak * self.settings.maximum_drawdown_fraction or equity <= 0:
                total_halt = True
        gains = sum((t.net_profit for t in trades if t.net_profit > 0), Decimal(0))
        losses = -sum((t.net_profit for t in trades if t.net_profit < 0), Decimal(0))
        returns = [float(t.return_fraction) for t in trades]
        sharpe = None
        if len(returns) >= 2:
            average = sum(returns) / len(returns)
            variance = sum((x - average) ** 2 for x in returns) / (len(returns) - 1)
            if variance > 0:
                sharpe = average / sqrt(variance)  # per-trade, unannualized
        high = initial_equity
        drawdown = Decimal(0)
        for value in curve:
            high = max(high, value)
            drawdown = max(drawdown, (high - value) / high)
        count = len(trades)
        metrics = Metrics(
            equity - initial_equity,
            gains / losses if losses else None,
            Decimal(sum(t.net_profit > 0 for t in trades)) / count if count else Decimal(0),
            sharpe,
            drawdown,
            sum((t.return_fraction for t in trades), Decimal(0)) / count if count else Decimal(0),
            count,
        )
        return Result(strategy.name, tuple(trades), metrics, daily_halt, total_halt)

    def compare(
        self, strategies: tuple[Strategy, ...], candles: tuple[Candle, ...], initial_equity: Decimal
    ) -> tuple[RankedResult, ...]:
        names = [s.name for s in strategies]
        if not names or len(set(names)) != len(names):
            raise ValueError("strategies must have unique nonempty names")
        results = [self.run(strategy, candles, initial_equity) for strategy in strategies]
        # Only positive simulated performance qualifies. No winning results means no allocation.
        scores = [
            max(Decimal(0), r.metrics.net_profit) if r.metrics.number_of_trades >= 2 else Decimal(0)
            for r in results
        ]
        total = sum(scores, Decimal(0))
        return tuple(
            RankedResult(r, score / total if total else Decimal(0))
            for r, score in sorted(
                zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True
            )
        )
