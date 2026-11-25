"""Frozen, supervised trade-outcome estimates for offline walk-forward gating."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Protocol


class Regime(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class CloseBar(Protocol):
    @property
    def close(self) -> Decimal: ...


def regime_at(bars: Sequence[CloseBar], index: int) -> Regime | None:
    """Use only completed bars before index; the bar at index is never inspected."""
    if index < 0 or index > len(bars):
        raise ValueError("regime index outside history")
    if index < 2:
        return None
    previous, latest = bars[index - 2].close, bars[index - 1].close
    if latest > previous:
        return Regime.UP
    if latest < previous:
        return Regime.DOWN
    return Regime.FLAT


@dataclass(frozen=True, slots=True)
class Observation:
    strategy: str
    regime: Regime
    net_profit: Decimal

    def __post_init__(self) -> None:
        if not self.strategy or not self.net_profit.is_finite():
            raise ValueError("invalid training observation")


@dataclass(frozen=True, slots=True)
class Estimate:
    strategy: str
    regime: Regime
    count: int
    mean_profit: Decimal
    win_rate: Decimal


@dataclass(frozen=True, slots=True)
class OutcomeModel:
    estimates: tuple[Estimate, ...]
    min_samples: int

    def approve(self, strategy: str, regime: Regime | None) -> bool:
        if regime is None:
            return False
        return any(
            value.strategy == strategy
            and value.regime is regime
            and value.count >= self.min_samples
            and value.mean_profit > 0
            and value.win_rate > Decimal("0.5")
            for value in self.estimates
        )


def fit(observations: tuple[Observation, ...], *, min_samples: int = 3) -> OutcomeModel:
    """Refit estimates from scratch; no future outcomes or mutable model state."""
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    groups: dict[tuple[str, Regime], list[Decimal]] = {}
    for observation in observations:
        groups.setdefault((observation.strategy, observation.regime), []).append(
            observation.net_profit
        )
    estimates = tuple(
        Estimate(
            strategy,
            regime,
            len(profits),
            sum(profits, Decimal(0)) / len(profits),
            Decimal(sum(profit > 0 for profit in profits)) / len(profits),
        )
        for (strategy, regime), profits in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1].value)
        )
    )
    return OutcomeModel(estimates, min_samples)
