"""Risk approval boundary; no position-sizing behavior is implemented yet."""

from typing import Protocol

from mnq_system.domain.orders import (
    RISK_APPROVAL_AUTHORITY,
    OrderIntent,
    RiskApprovedOrder,
    UnapprovedOrderError,
)
from mnq_system.domain.values import RiskAmount


class RiskPolicy(Protocol):
    def assess(self, intent: OrderIntent) -> RiskAmount: ...


class RiskApprover:
    """The only supported path from an order intent to an executable order."""

    def __init__(self, policy: RiskPolicy) -> None:
        self._policy = policy

    def approve(self, intent: OrderIntent) -> RiskApprovedOrder:
        if type(intent) is not OrderIntent:
            raise UnapprovedOrderError("a validated order intent is required")
        approved_risk = self._policy.assess(intent)
        return RISK_APPROVAL_AUTHORITY.issue(intent, approved_risk)
