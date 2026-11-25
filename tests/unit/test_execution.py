from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import pytest

from mnq_system.domain.orders import (
    RISK_APPROVAL_AUTHORITY,
    ExecutionPort,
    OrderIntent,
    OrderSide,
    RiskApprovalAuthority,
    RiskApprovedOrder,
    UnapprovedOrderError,
    require_risk_approved_order,
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


def test_order_intent_cannot_reach_an_adapter() -> None:
    adapter = RecordingAdapter()
    coordinator = cast("Any", ExecutionCoordinator(adapter))
    with pytest.raises(UnapprovedOrderError, match="risk-approved"):
        coordinator.execute(_intent())  # dynamic call bypasses static typing
    assert adapter.received is None


def test_disabled_adapter_rejects_unapproved_orders_before_submission() -> None:
    adapter = cast("Any", DisabledBrokerAdapter())
    with pytest.raises(UnapprovedOrderError, match="risk-approved"):
        adapter.accept(_intent())


@pytest.mark.parametrize("candidate", [None, "approved", 1, object()])
def test_execution_guard_rejects_arbitrary_objects(candidate: object) -> None:
    with pytest.raises(UnapprovedOrderError, match="risk-approved"):
        require_risk_approved_order(candidate)


def test_approval_authority_cannot_be_constructed_publicly() -> None:
    with pytest.raises(UnapprovedOrderError, match="cannot be constructed"):
        RiskApprovalAuthority(object())


def test_approved_order_rejects_forged_authorities() -> None:
    class ForgedAuthority(RiskApprovalAuthority):
        def __init__(self) -> None:
            """Deliberately skips the guarded base constructor."""

    forged = cast("RiskApprovalAuthority", ForgedAuthority())
    with pytest.raises(UnapprovedOrderError, match="issued by the risk approval boundary"):
        RiskApprovedOrder(_intent(), RiskAmount(Money(Decimal(1))), forged)
    with pytest.raises(UnapprovedOrderError, match="issued by the risk approval boundary"):
        RiskApprovedOrder(
            _intent(),
            RiskAmount(Money(Decimal(1))),
            cast("RiskApprovalAuthority", object()),
        )


def test_approved_order_rejects_invalid_payloads() -> None:
    with pytest.raises(UnapprovedOrderError, match="order intent"):
        RISK_APPROVAL_AUTHORITY.issue(cast("OrderIntent", object()), RiskAmount(Money(Decimal(1))))
    with pytest.raises(UnapprovedOrderError, match="approved risk"):
        RISK_APPROVAL_AUTHORITY.issue(_intent(), cast("RiskAmount", object()))


def test_only_the_risk_approver_produces_an_executable_order() -> None:
    approver = RiskApprover(FixedPolicy())
    approved = approver.approve(_intent())
    assert require_risk_approved_order(approved) is approved
    assert approved.approved_risk == RiskAmount(Money(Decimal("10")))


def test_risk_approver_rejects_unvalidated_intents() -> None:
    approver = cast("Any", RiskApprover(FixedPolicy()))
    with pytest.raises(UnapprovedOrderError, match="order intent"):
        approver.approve(object())


def test_disabled_adapter_never_submits_approved_orders() -> None:
    approved = RiskApprover(FixedPolicy()).approve(_intent())
    with pytest.raises(BrokerSubmissionDisabled, match="disabled"):
        DisabledBrokerAdapter().accept(approved)
