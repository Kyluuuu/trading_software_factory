"""Offline risk limits must reject incomplete or excessive exposure."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from mnq_system.config.settings import Settings
from mnq_system.domain.orders import OrderIntent, OrderSide
from mnq_system.domain.values import ContractQuantity, Price, Symbol, UtcTimestamp
from mnq_system.risk.limits import AccountSnapshot, ConfiguredRiskPolicy, RiskLimitExceeded


def intent(symbol: str = "AAPL", stop: str = "99", quantity: int = 1) -> OrderIntent:
    return OrderIntent(
        Symbol(symbol),
        OrderSide.BUY,
        ContractQuantity(quantity),
        Price(Decimal("100")),
        UtcTimestamp(datetime(2026, 1, 5, 15, tzinfo=UTC)),
        Price(Decimal(stop)),
    )


def snapshot(equity: str = "10000", opening: str = "10000", peak: str = "10000") -> AccountSnapshot:
    return AccountSnapshot(Decimal(equity), Decimal(opening), Decimal(peak))


def policy(
    state: AccountSnapshot | None = None, settings: Settings | None = None
) -> ConfiguredRiskPolicy:
    return ConfiguredRiskPolicy(settings or Settings(), state or snapshot())


def test_approves_within_one_tenth_percent_risk_budget() -> None:
    assert policy().assess(intent(quantity=9)).value.amount == Decimal("9")
    assert policy().assess(intent(quantity=10)).value.amount == Decimal("10")


@pytest.mark.parametrize("candidate", [intent(quantity=11), intent("MNQ"), intent(stop="100")])
def test_rejects_over_budget_unsupported_symbol_or_zero_stop(candidate: OrderIntent) -> None:
    with pytest.raises(RiskLimitExceeded):
        policy().assess(candidate)


def test_rejects_missing_or_wrong_side_stop() -> None:
    with pytest.raises(RiskLimitExceeded):
        policy().assess(replace(intent(), stop_price=None))
    with pytest.raises(RiskLimitExceeded):
        policy().assess(replace(intent(), stop_price=Price(Decimal("110"))))
    with pytest.raises(RiskLimitExceeded):
        policy().assess(replace(intent(), side=OrderSide.SELL))


def test_daily_loss_and_drawdown_halt_new_approvals() -> None:
    with pytest.raises(RiskLimitExceeded, match="daily loss"):
        policy(snapshot("9500")).assess(intent())
    with pytest.raises(RiskLimitExceeded, match="drawdown"):
        policy(snapshot("7000", "7000", "10000")).assess(intent())


def test_snapshot_validation_and_costs_count_against_budget() -> None:
    for values in [
        ("-1", "10000", "10000"),
        ("10001", "10000", "10000"),
        ("10000", "10000", "9999"),
    ]:
        with pytest.raises(ValueError, match=r"positive|peak"):
            snapshot(*values)
    settings = Settings.model_validate(
        {"costs": {"commission_per_contract": "1", "slippage_points": "0.5"}}
    )
    assert policy(settings=settings).assess(intent(quantity=2)).value.amount == Decimal("8")
    with pytest.raises(RiskLimitExceeded):
        policy(settings=settings).assess(intent(quantity=3))


def test_remaining_loss_headroom_must_cover_the_new_order() -> None:
    with pytest.raises(RiskLimitExceeded, match="daily loss"):
        policy(snapshot("9501")).assess(intent(quantity=2))
    with pytest.raises(RiskLimitExceeded, match="drawdown"):
        policy(snapshot("7001", "7001", "10000")).assess(intent(quantity=2))


def test_outside_session_and_weekends_are_rejected() -> None:
    for instant in [datetime(2026, 1, 5, 14, tzinfo=UTC), datetime(2026, 1, 3, 15, tzinfo=UTC)]:
        with pytest.raises(RiskLimitExceeded, match="session"):
            policy().assess(replace(intent(), created_at=UtcTimestamp(instant)))


def test_assessment_is_not_an_order_submission() -> None:
    from mnq_system.risk.approval import RiskApprover

    assert RiskApprover(policy()).approve(intent()).approved_risk.value.amount == Decimal("1")
