"""Fail-closed, timezone-aware access to available feature rows at decision time.

Use these joins for decision-time consumers; raw partition reads remain available
for integrity and coverage audits, not for unguarded decision joins.
"""

from bisect import bisect_right
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from mnq_system.features.engine import FeatureRow


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("decision and feature timestamps require timezone offsets")
    return value.astimezone(UTC)


def join_as_of(rows: Sequence[FeatureRow], decisions: Sequence[datetime]) -> tuple[FeatureRow, ...]:
    """For each ordered decision, select latest available row; reject missing history.

    This is not a join on candle timestamp. Each returned row is checked against
    the decision instant, and no consumer may infer current-bar OHLCV at its open.
    """
    previous: datetime | None = None
    times: list[datetime] = []
    for row in rows:
        at, available = _utc(row.timestamp), _utc(row.available_at)
        if (previous is not None and at <= previous) or available != at + timedelta(minutes=1):
            raise ValueError("feature rows must be ordered with valid availability")
        previous = at
        times.append(available)
    results: list[FeatureRow] = []
    previous_decision: datetime | None = None
    for decision in decisions:
        now = _utc(decision)
        if previous_decision is not None and now < previous_decision:
            raise ValueError("decision timestamps must be ordered")
        previous_decision = now
        index = bisect_right(times, now) - 1
        if index < 0:
            raise ValueError("feature not yet available at decision time")
        row = rows[index]
        if _utc(row.available_at) > now:
            raise ValueError("feature not yet available at decision time")
        results.append(row)
    return tuple(results)


def as_of(rows: Sequence[FeatureRow], decision_time: datetime) -> FeatureRow:
    """Get one latest eligible feature vector; never read a future minute."""
    return join_as_of(rows, (decision_time,))[0]
