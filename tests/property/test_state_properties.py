import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mnq_system.domain.state import PositionEvent, PositionState, allowed_transitions, transition
from mnq_system.domain.values import DomainValidationError


@settings(derandomize=True)
@given(st.sampled_from(PositionState), st.sampled_from(PositionEvent))
def test_transition_is_deterministic_and_matches_central_table(
    state: PositionState,
    event: PositionEvent,
) -> None:
    expected = allowed_transitions().get((state, event))
    if expected is None:
        with pytest.raises(DomainValidationError):
            transition(state, event)
    else:
        assert transition(state, event) is expected
        assert transition(state, event) is expected
