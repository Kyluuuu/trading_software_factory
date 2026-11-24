"""Execution coordination accepts only risk-approved orders."""

from mnq_system.domain.orders import ExecutionPort, RiskApprovedOrder


class ExecutionCoordinator:
    def __init__(self, adapter: ExecutionPort) -> None:
        self._adapter = adapter

    def execute(self, order: RiskApprovedOrder) -> None:
        self._adapter.accept(order)
