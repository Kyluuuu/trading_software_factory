"""Order types and the narrow risk-approved execution boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from mnq_system.domain.values import (
    ContractQuantity,
    DomainValidationError,
    Price,
    RiskAmount,
    Symbol,
    UtcTimestamp,
)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class UnapprovedOrderError(DomainValidationError):
    """Raised when an order was not issued by the risk approval boundary."""


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A proposal only. It carries no authority to reach an adapter."""

    symbol: Symbol
    side: OrderSide
    quantity: ContractQuantity
    reference_price: Price
    created_at: UtcTimestamp


_AUTHORITY_TOKEN = object()


class RiskApprovalAuthority:
    """Sole issuer of approved orders.

    The constructor is guarded by a module-private token, so the only usable
    instance is the one created in this module. Architecture rules restrict use
    of that instance to the risk package.
    """

    __slots__ = ()

    def __init__(self, token: object) -> None:
        if token is not _AUTHORITY_TOKEN:
            raise UnapprovedOrderError("risk approval authority cannot be constructed directly")

    def issue(self, intent: OrderIntent, approved_risk: RiskAmount) -> "RiskApprovedOrder":
        return RiskApprovedOrder(intent, approved_risk, self)


@dataclass(frozen=True, slots=True, init=False)
class RiskApprovedOrder:
    """An order that passed the risk boundary. Construct it via the authority."""

    intent: OrderIntent
    approved_risk: RiskAmount

    def __init__(
        self,
        intent: OrderIntent,
        approved_risk: RiskAmount,
        authority: RiskApprovalAuthority,
    ) -> None:
        if authority is not RISK_APPROVAL_AUTHORITY:
            raise UnapprovedOrderError(
                "RiskApprovedOrder must be issued by the risk approval boundary"
            )
        if type(intent) is not OrderIntent:
            raise UnapprovedOrderError("a validated order intent is required")
        if type(approved_risk) is not RiskAmount:
            raise UnapprovedOrderError("a validated approved risk amount is required")
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "approved_risk", approved_risk)


RISK_APPROVAL_AUTHORITY = RiskApprovalAuthority(_AUTHORITY_TOKEN)


def require_risk_approved_order(order: object) -> RiskApprovedOrder:
    """Fail closed unless the object is exactly a risk-approved order."""
    if type(order) is not RiskApprovedOrder:
        raise UnapprovedOrderError("only a risk-approved order may cross an execution boundary")
    return order


@runtime_checkable
class ExecutionPort(Protocol):
    """Only validated orders may cross an execution boundary."""

    def accept(self, order: RiskApprovedOrder) -> None: ...
