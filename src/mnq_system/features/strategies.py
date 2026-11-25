"""Pure, causal strategy proposals for offline candle research."""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from mnq_system.domain.orders import OrderSide

if TYPE_CHECKING:
    from mnq_system.backtest.engine import Candle


@dataclass(frozen=True, slots=True)
class Signal:
    side: OrderSide
    stop_distance: Decimal

    def __post_init__(self) -> None:
        if not self.stop_distance.is_finite() or self.stop_distance <= 0:
            raise ValueError("signal requires a side and positive finite stop distance")


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def history_limit(self) -> int: ...

    def generate(self, history: tuple["Candle", ...]) -> Signal | None: ...


@dataclass(frozen=True, slots=True)
class MeanReversion:
    lookback: int = 20
    threshold: Decimal = Decimal("0.01")
    stop_distance: Decimal = Decimal("1")
    name: str = "mean_reversion"

    def __post_init__(self) -> None:
        if self.lookback < 2 or not self.threshold.is_finite() or self.threshold <= 0:
            raise ValueError("invalid mean reversion parameters")
        if not self.stop_distance.is_finite() or self.stop_distance <= 0:
            raise ValueError("invalid stop distance")

    @property
    def history_limit(self) -> int:
        return self.lookback

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.lookback:
            return None
        window = history[-self.lookback :]
        mean = sum((c.close for c in window), Decimal(0)) / self.lookback
        last = window[-1].close
        if last < mean * (1 - self.threshold):
            return Signal(OrderSide.BUY, self.stop_distance)
        if last > mean * (1 + self.threshold):
            return Signal(OrderSide.SELL, self.stop_distance)
        return None


@dataclass(frozen=True, slots=True)
class Breakout:
    lookback: int = 20
    stop_distance: Decimal = Decimal("1")
    name: str = "breakout"

    def __post_init__(self) -> None:
        if self.lookback < 1 or not self.stop_distance.is_finite() or self.stop_distance <= 0:
            raise ValueError("invalid breakout parameters")

    @property
    def history_limit(self) -> int:
        return self.lookback + 1

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) <= self.lookback:
            return None
        previous = history[-self.lookback - 1 : -1]
        last = history[-1].close
        if last > max(c.high for c in previous):
            return Signal(OrderSide.BUY, self.stop_distance)
        if last < min(c.low for c in previous):
            return Signal(OrderSide.SELL, self.stop_distance)
        return None


def default_strategies() -> tuple[Strategy, ...]:
    """Independent, fixed-parameter rules; no ML, tuning or shared portfolio."""
    from mnq_system.features.extra import (
        Momentum,
        TrendFollowing,
        VolatilityExpansion,
        VolumeSpike,
        VWAPReversion,
    )

    return (
        MeanReversion(),
        Breakout(),
        TrendFollowing(),
        Momentum(),
        VolumeSpike(),
        VWAPReversion(),
        VolatilityExpansion(),
    )
