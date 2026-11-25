"""State-machine tests using an expectation defined independently of the implementation."""

import pytest

from mnq_system.domain.state import (
    PositionEvent,
    PositionState,
    allowed_transitions,
    transition,
)
from mnq_system.domain.values import DomainValidationError

EXPECTED_TRANSITIONS: dict[tuple[PositionState, PositionEvent], PositionState] = {
    (PositionState.FLAT, PositionEvent.REQUEST_ENTRY): PositionState.PENDING_ENTRY,
    (PositionState.FLAT, PositionEvent.HALT): PositionState.HALTED,
    (PositionState.PENDING_ENTRY, PositionEvent.FILL_LONG): PositionState.LONG,
    (PositionState.PENDING_ENTRY, PositionEvent.FILL_SHORT): PositionState.SHORT,
    (PositionState.PENDING_ENTRY, PositionEvent.CANCEL): PositionState.FLAT,
    (PositionState.PENDING_ENTRY, PositionEvent.HALT): PositionState.HALTED,
    (PositionState.LONG, PositionEvent.REQUEST_EXIT): PositionState.PENDING_EXIT,
    (PositionState.LONG, PositionEvent.HALT): PositionState.HALTED,
    (PositionState.SHORT, PositionEvent.REQUEST_EXIT): PositionState.PENDING_EXIT,
    (PositionState.SHORT, PositionEvent.HALT): PositionState.HALTED,
    (PositionState.PENDING_EXIT, PositionEvent.FILL_EXIT): PositionState.FLAT,
    (PositionState.PENDING_EXIT, PositionEvent.HALT): PositionState.HALTED,
    (PositionState.HALTED, PositionEvent.ADMIN_RESET): PositionState.FLAT,
}

REJECTED_TRANSITIONS = [
    (state, event)
    for state in PositionState
    for event in PositionEvent
    if (state, event) not in EXPECTED_TRANSITIONS
]


def test_implementation_matches_the_independent_expectation() -> None:
    assert allowed_transitions() == EXPECTED_TRANSITIONS


@pytest.mark.parametrize(
    ("state", "event", "expected"),
    [(state, event, expected) for (state, event), expected in EXPECTED_TRANSITIONS.items()],
)
def test_every_allowed_transition(
    state: PositionState,
    event: PositionEvent,
    expected: PositionState,
) -> None:
    assert transition(state, event) is expected


@pytest.mark.parametrize(("state", "event"), REJECTED_TRANSITIONS)
def test_every_rejected_transition(state: PositionState, event: PositionEvent) -> None:
    with pytest.raises(DomainValidationError):
        transition(state, event)


@pytest.mark.parametrize(
    "state",
    [
        PositionState.PENDING_ENTRY,
        PositionState.LONG,
        PositionState.SHORT,
        PositionState.PENDING_EXIT,
        PositionState.HALTED,
    ],
)
def test_entry_is_only_possible_while_flat(state: PositionState) -> None:
    with pytest.raises(DomainValidationError, match="not allowed"):
        transition(state, PositionEvent.REQUEST_ENTRY)


def test_cancelling_a_pending_exit_cannot_report_a_flat_position() -> None:
    with pytest.raises(DomainValidationError, match="not allowed"):
        transition(PositionState.PENDING_EXIT, PositionEvent.CANCEL)


_NON_RESET_EVENTS = [event for event in PositionEvent if event is not PositionEvent.ADMIN_RESET]


@pytest.mark.parametrize("event", _NON_RESET_EVENTS)
def test_halted_requires_an_administrative_reset(event: PositionEvent) -> None:
    with pytest.raises(DomainValidationError, match="not allowed"):
        transition(PositionState.HALTED, event)


def test_transition_table_copy_cannot_mutate_machine() -> None:
    copied = allowed_transitions()
    copied.clear()
    result = transition(PositionState.FLAT, PositionEvent.REQUEST_ENTRY)
    assert result is PositionState.PENDING_ENTRY
