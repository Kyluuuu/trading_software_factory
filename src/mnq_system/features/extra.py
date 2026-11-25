"""Additional causal stock-candle research signals; no model training or execution."""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from mnq_system.domain.orders import OrderSide
from mnq_system.features.strategies import Signal

if TYPE_CHECKING:
    from mnq_system.backtest.engine import Candle


def _parameters(lookback: int, stop_distance: Decimal) -> None:
    if lookback < 2 or not stop_distance.is_finite() or stop_distance <= 0:
        raise ValueError("lookback must be >= 2 and stop distance positive")


@dataclass(frozen=True, slots=True)
class TrendFollowing:
    lookback: int = 20
    stop_distance: Decimal = Decimal("1")
    name: str = "trend_following"

    def __post_init__(self) -> None:
        _parameters(self.lookback, self.stop_distance)

    @property
    def history_limit(self) -> int:
        return self.lookback

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.lookback:
            return None
        half = self.lookback // 2
        recent = sum(c.close for c in history[-half:]) / half
        earlier = sum(c.close for c in history[-self.lookback : -half]) / (self.lookback - half)
        if recent == earlier:
            return None
        return Signal(OrderSide.BUY if recent > earlier else OrderSide.SELL, self.stop_distance)


@dataclass(frozen=True, slots=True)
class Momentum:
    lookback: int = 10
    stop_distance: Decimal = Decimal("1")
    name: str = "momentum"

    def __post_init__(self) -> None:
        _parameters(self.lookback, self.stop_distance)

    @property
    def history_limit(self) -> int:
        return self.lookback

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.lookback:
            return None
        change = history[-1].close - history[-self.lookback].close
        if change == 0:
            return None
        return Signal(OrderSide.BUY if change > 0 else OrderSide.SELL, self.stop_distance)


@dataclass(frozen=True, slots=True)
class VolumeSpike:
    lookback: int = 20
    multiple: Decimal = Decimal("2")
    stop_distance: Decimal = Decimal("1")
    name: str = "volume_spike"

    def __post_init__(self) -> None:
        _parameters(self.lookback, self.stop_distance)
        if not self.multiple.is_finite() or self.multiple <= 1:
            raise ValueError("volume multiple must exceed 1")

    @property
    def history_limit(self) -> int:
        return self.lookback + 1

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.history_limit:
            return None
        earlier = history[-self.history_limit : -1]
        average = Decimal(sum(c.volume for c in earlier)) / self.lookback
        last = history[-1]
        if average == 0 or last.volume < self.multiple * average or last.close == last.open:
            return None
        return Signal(
            OrderSide.BUY if last.close > last.open else OrderSide.SELL, self.stop_distance
        )


@dataclass(frozen=True, slots=True)
class VWAPReversion:
    lookback: int = 20
    threshold: Decimal = Decimal("0.01")
    stop_distance: Decimal = Decimal("1")
    name: str = "vwap_reversion"

    def __post_init__(self) -> None:
        _parameters(self.lookback, self.stop_distance)
        if not self.threshold.is_finite() or self.threshold <= 0:
            raise ValueError("threshold must be positive")

    @property
    def history_limit(self) -> int:
        return self.lookback

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.lookback:
            return None
        window = history[-self.lookback :]
        volume = sum(c.volume for c in window)
        if volume == 0:
            return None
        vwap = sum(((c.high + c.low + c.close) / 3 * c.volume for c in window), Decimal(0)) / volume
        last = window[-1].close
        if last < vwap * (1 - self.threshold):
            return Signal(OrderSide.BUY, self.stop_distance)
        if last > vwap * (1 + self.threshold):
            return Signal(OrderSide.SELL, self.stop_distance)
        return None


@dataclass(frozen=True, slots=True)
class VolatilityExpansion:
    lookback: int = 20
    multiple: Decimal = Decimal("2")
    stop_distance: Decimal = Decimal("1")
    name: str = "volatility_expansion"

    def __post_init__(self) -> None:
        _parameters(self.lookback, self.stop_distance)
        if not self.multiple.is_finite() or self.multiple <= 1:
            raise ValueError("range multiple must exceed 1")

    @property
    def history_limit(self) -> int:
        return self.lookback + 1

    def generate(self, history: tuple["Candle", ...]) -> Signal | None:
        if len(history) < self.history_limit:
            return None
        previous = history[-self.history_limit : -1]
        average_range = sum((c.high - c.low for c in previous), Decimal(0)) / self.lookback
        last = history[-1]
        if average_range == 0 or last.high - last.low < self.multiple * average_range:
            return None
        if last.close == last.open:
            return None
        return Signal(
            OrderSide.BUY if last.close > last.open else OrderSide.SELL, self.stop_distance
        )
