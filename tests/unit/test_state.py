import pytest

from mnq_system.domain.state import (
    PositionEvent,
    PositionState,
    allowed_transitions,
    transition,
)
from mnq_system.domain.values import DomainValidationError


@pytest.mark.parametrize(
    ("state", "event", "expected"),
    [(state, event, expected) for (state, event), expected in allowed_transitions().items()],
)
def test_every_allowed_transition(
    state: PositionState,
    event: PositionEvent,
    expected: PositionState,
) -> None:
    assert transition(state, event) is expected


_REJECTED = [
    (state, event)
    for state in PositionState
    for event in PositionEvent
    if (state, event) not in allowed_transitions()
]


@pytest.mark.parametrize(("state", "event"), _REJECTED)
def test_every_rejected_transition(state: PositionState, event: PositionEvent) -> None:
    with pytest.raises(DomainValidationError):
        transition(state, event)


def test_transition_table_copy_cannot_mutate_machine() -> None:
    copied = allowed_transitions()
    copied.clear()
    result = transition(PositionState.FLAT, PositionEvent.REQUEST_ENTRY)
    assert result is PositionState.PENDING_ENTRY
