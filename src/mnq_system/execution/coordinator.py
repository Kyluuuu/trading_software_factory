"""Execution coordination accepts only risk-approved orders."""

from mnq_system.domain.orders import (
    ExecutionPort,
    RiskApprovedOrder,
    require_risk_approved_order,
)


class ExecutionCoordinator:
    def __init__(self, adapter: ExecutionPort) -> None:
        self._adapter = adapter

    def execute(self, order: RiskApprovedOrder) -> None:
        """Validate at runtime so dynamic callers cannot bypass the annotation."""
        self._adapter.accept(require_risk_approved_order(order))
