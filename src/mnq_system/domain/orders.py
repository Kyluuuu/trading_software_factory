"""Order types and narrow execution ports."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from mnq_system.domain.values import ContractQuantity, Price, RiskAmount, Symbol, UtcTimestamp


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: Symbol
    side: OrderSide
    quantity: ContractQuantity
    reference_price: Price
    created_at: UtcTimestamp


class RiskApprovalProof:
    """Capability whose use is statically restricted to the risk package."""


@dataclass(frozen=True, slots=True, init=False)
class RiskApprovedOrder:
    intent: OrderIntent
    approved_risk: RiskAmount

    def __init__(
        self,
        intent: OrderIntent,
        approved_risk: RiskAmount,
        proof: RiskApprovalProof,
    ) -> None:
        if type(proof) is not RiskApprovalProof:
            raise PermissionError("RiskApprovedOrder must be created by the risk approval boundary")
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "approved_risk", approved_risk)


@runtime_checkable
class ExecutionPort(Protocol):
    """Only validated orders may cross an execution boundary."""

    def accept(self, order: RiskApprovedOrder) -> None: ...
