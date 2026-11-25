"""Permanently disabled broker adapter for the foundation milestone."""

from mnq_system.domain.orders import RiskApprovedOrder, require_risk_approved_order


class BrokerSubmissionDisabled(RuntimeError):
    """Raised because this milestone cannot connect or submit orders."""


class DisabledBrokerAdapter:
    def accept(self, order: RiskApprovedOrder) -> None:
        require_risk_approved_order(order)
        raise BrokerSubmissionDisabled("broker submission is disabled")
