"""Offline stock/ETF risk decisions using explicit account snapshots.

This policy does not submit orders or maintain account state. A caller must supply
fresh, independently verified equity, session-opening equity and high-water equity.
"""

from dataclasses import dataclass
from decimal import Decimal
from zoneinfo import ZoneInfo

from mnq_system.config.settings import Settings
from mnq_system.domain.orders import OrderIntent, OrderSide
from mnq_system.domain.values import Money, RiskAmount


class RiskLimitExceeded(ValueError):
    """The proposal fails a configured safety limit."""


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: Decimal
    session_opening_equity: Decimal
    peak_equity: Decimal

    def __post_init__(self) -> None:
        amounts = (self.equity, self.session_opening_equity, self.peak_equity)
        if any(not value.is_finite() for value in amounts):
            raise ValueError("account values must be finite Decimals")
        if self.equity <= 0 or self.session_opening_equity <= 0 or self.peak_equity <= 0:
            raise ValueError("account equity values must be positive")
        if self.peak_equity < self.equity or self.peak_equity < self.session_opening_equity:
            raise ValueError("peak equity cannot be below current or opening equity")


class ConfiguredRiskPolicy:
    """Validate one stock/ETF proposal against the current snapshot and limits.

    This is not a continuous loss monitor: each new decision needs a new snapshot.
    Stop prices are estimates, not guaranteed fill prices.
    """

    def __init__(self, settings: Settings, snapshot: AccountSnapshot) -> None:
        self._settings = settings
        self._snapshot = snapshot

    def size_quantity(self, entry: Decimal, stop: Decimal, allocation: Decimal = Decimal(1)) -> int:
        """Maximum whole shares under trade, daily and peak-loss budgets.

        A zero result means no feasible trade; assess the resulting intent before use.
        Allocation is a fraction of the per-trade risk budget, not account capital.
        """
        if (
            not entry.is_finite()
            or not stop.is_finite()
            or entry <= 0
            or stop <= 0
            or not allocation.is_finite()
            or not Decimal(0) < allocation <= 1
        ):
            raise RiskLimitExceeded("invalid sizing input")
        distance = abs(entry - stop)
        if distance == 0:
            raise RiskLimitExceeded("stop must differ from entry")
        state, settings = self._snapshot, self._settings
        daily = state.session_opening_equity * settings.maximum_daily_loss_fraction - (
            state.session_opening_equity - state.equity
        )
        drawdown = state.peak_equity * settings.maximum_drawdown_fraction - (
            state.peak_equity - state.equity
        )
        if daily <= 0 or drawdown <= 0:
            raise RiskLimitExceeded("account loss limit reached")
        budget = min(state.equity * settings.maximum_risk_fraction * allocation, daily, drawdown)
        cost = Decimal(2) * (
            settings.costs.commission_per_contract + settings.costs.slippage_points
        )
        return max(0, int(budget // (distance + cost)))

    def assess(self, intent: OrderIntent) -> RiskAmount:
        settings, state = self._settings, self._snapshot
        if intent.symbol.value not in settings.allowed_symbols:
            raise RiskLimitExceeded("symbol is not allowed")
        local = intent.created_at.value.astimezone(ZoneInfo(settings.session.timezone))
        if local.weekday() >= 5 or not (
            settings.session.start <= local.time() < settings.session.end
        ):
            raise RiskLimitExceeded("outside configured session")
        if state.session_opening_equity - state.equity >= (
            state.session_opening_equity * settings.maximum_daily_loss_fraction
        ):
            raise RiskLimitExceeded("daily loss limit reached")
        if state.peak_equity - state.equity >= (
            state.peak_equity * settings.maximum_drawdown_fraction
        ):
            raise RiskLimitExceeded("drawdown limit reached")
        stop = intent.stop_price
        if stop is None:
            raise RiskLimitExceeded("a protective stop is required")
        entry = intent.reference_price.value
        distance = entry - stop.value if intent.side is OrderSide.BUY else stop.value - entry
        if distance <= 0:
            raise RiskLimitExceeded("stop must be on the loss side of the entry")
        # Stock and ETF units are shares. Cost estimates cover entry AND exit.
        cost = Decimal(2) * (
            settings.costs.commission_per_contract + settings.costs.slippage_points
        )
        amount = (distance + cost) * intent.quantity.value
        if amount > state.equity * settings.maximum_risk_fraction:
            raise RiskLimitExceeded("per-order risk limit exceeded")
        if state.session_opening_equity - state.equity + amount > (
            state.session_opening_equity * settings.maximum_daily_loss_fraction
        ):
            raise RiskLimitExceeded("insufficient daily loss headroom")
        if state.peak_equity - state.equity + amount > (
            state.peak_equity * settings.maximum_drawdown_fraction
        ):
            raise RiskLimitExceeded("insufficient drawdown headroom")
        return RiskAmount(Money(amount))
