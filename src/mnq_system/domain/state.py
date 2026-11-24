"""Explicit position state machine."""

from enum import Enum

from mnq_system.domain.values import DomainValidationError


class PositionState(Enum):
    FLAT = "flat"
    PENDING_ENTRY = "pending_entry"
    LONG = "long"
    SHORT = "short"
    PENDING_EXIT = "pending_exit"
    HALTED = "halted"


class PositionEvent(Enum):
    REQUEST_ENTRY = "request_entry"
    FILL_LONG = "fill_long"
    FILL_SHORT = "fill_short"
    REQUEST_EXIT = "request_exit"
    FILL_EXIT = "fill_exit"
    CANCEL = "cancel"
    HALT = "halt"
    ADMIN_RESET = "admin_reset"


_TRANSITIONS: dict[tuple[PositionState, PositionEvent], PositionState] = {
    (PositionState.FLAT, PositionEvent.REQUEST_ENTRY): PositionState.PENDING_ENTRY,
    (PositionState.PENDING_ENTRY, PositionEvent.FILL_LONG): PositionState.LONG,
    (PositionState.PENDING_ENTRY, PositionEvent.FILL_SHORT): PositionState.SHORT,
    (PositionState.PENDING_ENTRY, PositionEvent.CANCEL): PositionState.FLAT,
    (PositionState.LONG, PositionEvent.REQUEST_EXIT): PositionState.PENDING_EXIT,
    (PositionState.SHORT, PositionEvent.REQUEST_EXIT): PositionState.PENDING_EXIT,
    (PositionState.PENDING_EXIT, PositionEvent.FILL_EXIT): PositionState.FLAT,
    (PositionState.PENDING_EXIT, PositionEvent.CANCEL): PositionState.FLAT,
    (PositionState.HALTED, PositionEvent.ADMIN_RESET): PositionState.FLAT,
}
for _state in PositionState:
    if _state is not PositionState.HALTED:
        _TRANSITIONS[(_state, PositionEvent.HALT)] = PositionState.HALTED


def transition(state: PositionState, event: PositionEvent) -> PositionState:
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as error:
        raise DomainValidationError(
            f"transition {event.value} is not allowed from {state.value}"
        ) from error


def allowed_transitions() -> dict[tuple[PositionState, PositionEvent], PositionState]:
    """Return a copy so the central transition table cannot be mutated."""
    return dict(_TRANSITIONS)
