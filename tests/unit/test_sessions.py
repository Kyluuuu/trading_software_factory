from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pytest

from mnq_system.domain.values import Symbol, UtcTimestamp
from mnq_system.sessions.policy import SessionPolicy, TradingSession

NEW_YORK = ZoneInfo("America/New_York")


class FixedSessionPolicy:
    def __init__(self, session: TradingSession) -> None:
        self._session = session

    def permits(self, symbol: Symbol, timestamp: UtcTimestamp) -> bool:
        del symbol
        local = timestamp.value.astimezone(self._session.timezone).time()
        return self._session.start <= local <= self._session.end


def test_valid_session_is_constructed() -> None:
    session = TradingSession(timezone=NEW_YORK, start=time(9, 30), end=time(16, 0))
    assert session.timezone is NEW_YORK
    assert session.start == time(9, 30)
    assert session.end == time(16, 0)


@pytest.mark.parametrize(
    ("start", "end"),
    [(time(16, 0), time(9, 30)), (time(9, 30), time(9, 30))],
)
def test_invalid_session_times_are_rejected(start: time, end: time) -> None:
    with pytest.raises(ValueError, match="start must be before"):
        TradingSession(timezone=NEW_YORK, start=start, end=end)


def test_session_policy_uses_deterministic_timestamps() -> None:
    session = TradingSession(timezone=NEW_YORK, start=time(9, 30), end=time(16, 0))
    policy: SessionPolicy = FixedSessionPolicy(session)
    symbol = Symbol("MNQ")
    inside = UtcTimestamp(datetime(2026, 1, 5, 15, 0, tzinfo=UTC))
    outside = UtcTimestamp(datetime(2026, 1, 5, 2, 0, tzinfo=UTC))
    assert policy.permits(symbol, inside) is True
    assert policy.permits(symbol, outside) is False
