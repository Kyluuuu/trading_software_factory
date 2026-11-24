from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mnq_system.domain.values import (
    ContractQuantity,
    DomainValidationError,
    PortfolioEquity,
    Price,
    RiskFraction,
    StopDistance,
)

FINITE_DECIMALS = st.decimals(
    min_value=Decimal("-1000000"),
    max_value=Decimal("1000000"),
    allow_nan=False,
    allow_infinity=False,
    places=6,
)


@settings(derandomize=True)
@given(FINITE_DECIMALS)
def test_risk_fraction_acceptance_matches_closed_boundary(value: Decimal) -> None:
    if Decimal(0) < value <= Decimal(1):
        assert RiskFraction(value).value == value
    else:
        with pytest.raises(DomainValidationError):
            RiskFraction(value)


@settings(derandomize=True)
@given(FINITE_DECIMALS)
def test_equity_acceptance_matches_non_negative_boundary(value: Decimal) -> None:
    if value >= 0:
        assert PortfolioEquity(value).value == value
    else:
        with pytest.raises(DomainValidationError):
            PortfolioEquity(value)


@settings(derandomize=True)
@given(FINITE_DECIMALS)
def test_price_and_stop_require_positive_values(value: Decimal) -> None:
    if value > 0:
        assert Price(value).value == value
        assert StopDistance(value).value == value
    else:
        with pytest.raises(DomainValidationError):
            Price(value)
        with pytest.raises(DomainValidationError):
            StopDistance(value)


@settings(derandomize=True)
@given(st.integers(min_value=-10_000, max_value=10_000))
def test_quantity_requires_positive_integer(value: int) -> None:
    if value > 0:
        assert ContractQuantity(value).value == value
    else:
        with pytest.raises(DomainValidationError):
            ContractQuantity(value)
