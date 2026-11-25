from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

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


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, 2.0, Decimal("1.5"), Decimal(1), "1"])
def test_invalid_contract_quantity_is_rejected(value: object) -> None:
    with pytest.raises(DomainValidationError, match="positive integer"):
        ContractQuantity(cast("int", value))


@pytest.mark.parametrize("value", [1, 2, 10_000])
def test_exact_positive_integer_quantities_are_accepted(value: int) -> None:
    assert ContractQuantity(value).value == value


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
@pytest.mark.parametrize(
    "factory",
    [Money, Price, PortfolioEquity, RiskFraction, StopDistance],
)
def test_non_finite_values_are_rejected_by_every_decimal_type(
    factory: Callable[[Decimal], object],
    value: Decimal,
) -> None:
    with pytest.raises(DomainValidationError, match="finite"):
        factory(value)


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
