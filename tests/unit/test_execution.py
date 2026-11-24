from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest

from mnq_system.domain.orders import (
    ExecutionPort,
    OrderIntent,
    OrderSide,
    RiskApprovalProof,
    RiskApprovedOrder,
)
from mnq_system.domain.values import (
    ContractQuantity,
    Money,
    Price,
    RiskAmount,
    Symbol,
    UtcTimestamp,
)
from mnq_system.execution.coordinator import ExecutionCoordinator
from mnq_system.live.disabled import BrokerSubmissionDisabled, DisabledBrokerAdapter
from mnq_system.risk.approval import RiskApprover


class FixedPolicy:
    def assess(self, intent: OrderIntent) -> RiskAmount:
        del intent
        return RiskAmount(Money(Decimal("10")))


class RecordingAdapter:
    def __init__(self) -> None:
        self.received: RiskApprovedOrder | None = None

    def accept(self, order: RiskApprovedOrder) -> None:
        self.received = order


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol=Symbol("MNQ"),
        side=OrderSide.BUY,
        quantity=ContractQuantity(1),
        reference_price=Price(Decimal("20000")),
        created_at=UtcTimestamp(datetime(2026, 1, 1, tzinfo=UTC)),
    )


def test_risk_boundary_is_supported_construction_path() -> None:
    approved = RiskApprover(FixedPolicy()).approve(_intent())
    adapter = RecordingAdapter()
    ExecutionCoordinator(adapter).execute(approved)
    assert adapter.received is approved
    assert isinstance(adapter, ExecutionPort)


def test_direct_approved_order_construction_is_rejected() -> None:
    with pytest.raises(PermissionError):
        RiskApprovedOrder(
            _intent(),
            RiskAmount(Money(Decimal(1))),
            cast("RiskApprovalProof", object()),  # deliberate invalid runtime boundary
        )


def test_disabled_adapter_never_submits() -> None:
    approved = RiskApprover(FixedPolicy()).approve(_intent())
    with pytest.raises(BrokerSubmissionDisabled, match="disabled"):
        DisabledBrokerAdapter().accept(approved)
