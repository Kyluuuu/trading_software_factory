"""Risk approval boundary; no position-sizing behavior is implemented yet."""

from typing import Protocol

from mnq_system.domain.orders import OrderIntent, RiskApprovalProof, RiskApprovedOrder
from mnq_system.domain.values import RiskAmount


class RiskPolicy(Protocol):
    def assess(self, intent: OrderIntent) -> RiskAmount: ...


class RiskApprover:
    def __init__(self, policy: RiskPolicy) -> None:
        self._policy = policy

    def approve(self, intent: OrderIntent) -> RiskApprovedOrder:
        approved_risk = self._policy.assess(intent)
        return RiskApprovedOrder(intent, approved_risk, RiskApprovalProof())
