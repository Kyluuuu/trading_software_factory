from datetime import UTC, datetime
from decimal import Decimal

import pytest

from mnq_system.config.symbols import require_permitted_symbol
from mnq_system.domain.values import (
    ContractQuantity,
    DomainValidationError,
    Money,
    PortfolioEquity,
    Price,
    RiskFraction,
    StopDistance,
    Symbol,
    UtcTimestamp,
)


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_contract_quantity_is_rejected(value: int) -> None:
    with pytest.raises(DomainValidationError):
        ContractQuantity(value)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_non_finite_money_and_price_are_rejected(value: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        Money(value)
    with pytest.raises(DomainValidationError):
        Price(value)


def test_negative_equity_and_non_positive_stop_are_rejected() -> None:
    with pytest.raises(DomainValidationError):
        PortfolioEquity(Decimal("-0.01"))
    with pytest.raises(DomainValidationError):
        StopDistance(Decimal(0))


@pytest.mark.parametrize("value", [Decimal(0), Decimal("1.01"), Decimal("NaN")])
def test_invalid_risk_fraction_is_rejected(value: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        RiskFraction(value)


def test_timestamp_requires_timezone_and_normalizes_to_utc() -> None:
    with pytest.raises(DomainValidationError):
        UtcTimestamp(datetime(2026, 1, 1))  # noqa: DTZ001 - deliberately invalid input
    timestamp = UtcTimestamp(datetime(2026, 1, 1, tzinfo=UTC))
    assert timestamp.value.tzinfo is UTC


def test_only_permitted_symbol_passes_guard() -> None:
    permitted = Symbol("MNQ")
    assert require_permitted_symbol(permitted, permitted) is permitted
    with pytest.raises(DomainValidationError, match="not permitted"):
        require_permitted_symbol(Symbol("MES"), permitted)


@pytest.mark.parametrize("value", ["", " mnq", "mnq"])
def test_invalid_symbol_representation_is_rejected(value: str) -> None:
    with pytest.raises(DomainValidationError):
        Symbol(value)
