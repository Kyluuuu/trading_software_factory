"""Immutable, validated domain values."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Self


class DomainValidationError(ValueError):
    """Raised when a value violates a domain invariant."""


def _finite(value: Decimal, name: str) -> Decimal:
    if not value.is_finite():
        raise DomainValidationError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class Symbol:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().upper()
        if not normalized or normalized != self.value:
            raise DomainValidationError("symbol must be non-empty, trimmed, and uppercase")


@dataclass(frozen=True, slots=True)
class ContractQuantity:
    value: int

    def __post_init__(self) -> None:
        # `type(...) is not int` rejects bool, float, Decimal and numeric subclasses.
        if type(self.value) is not int or self.value <= 0:
            raise DomainValidationError("contract quantity must be a positive integer")


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        _finite(self.amount, "money")
        if self.currency != "USD":
            raise DomainValidationError("only USD is supported by this foundation")


@dataclass(frozen=True, slots=True)
class Price:
    value: Decimal

    def __post_init__(self) -> None:
        _finite(self.value, "price")
        if self.value <= 0:
            raise DomainValidationError("price must be positive")


@dataclass(frozen=True, slots=True)
class PortfolioEquity:
    value: Decimal

    def __post_init__(self) -> None:
        _finite(self.value, "portfolio equity")
        if self.value < 0:
            raise DomainValidationError("portfolio equity cannot be negative")


@dataclass(frozen=True, slots=True)
class RiskFraction:
    value: Decimal

    def __post_init__(self) -> None:
        _finite(self.value, "risk fraction")
        if not Decimal(0) < self.value <= Decimal(1):
            raise DomainValidationError("risk fraction must be greater than 0 and at most 1")


@dataclass(frozen=True, slots=True)
class RiskAmount:
    value: Money

    def __post_init__(self) -> None:
        if self.value.amount < 0:
            raise DomainValidationError("risk amount cannot be negative")


@dataclass(frozen=True, slots=True)
class StopDistance:
    value: Decimal

    def __post_init__(self) -> None:
        _finite(self.value, "stop distance")
        if self.value <= 0:
            raise DomainValidationError("stop distance must be positive")


@dataclass(frozen=True, slots=True)
class UtcTimestamp:
    value: datetime

    def __post_init__(self) -> None:
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise DomainValidationError("timestamp must be timezone-aware")
        object.__setattr__(self, "value", self.value.astimezone(UTC))

    @classmethod
    def from_datetime(cls, value: datetime) -> Self:
        return cls(value)
