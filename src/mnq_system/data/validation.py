"""Validate one-symbol 1-minute stock bars before storage and during reads."""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

from mnq_system.backtest.engine import Candle


def validate_bars(bars: tuple[Candle, ...]) -> tuple[Candle, ...]:
    """Reject corrupt rows rather than silently sorting or deduplicating them."""
    if not bars:
        return bars
    symbol = bars[0].symbol
    for bar in bars:
        if bar.symbol != symbol:
            raise ValueError("mixed symbols")
        timestamp = bar.timestamp
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
            or timestamp.second
            or timestamp.microsecond
        ):
            raise ValueError("invalid or missing minute timestamp")
        if type(bar.volume) is not int or bar.volume < 0:
            raise ValueError("invalid volume")
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (
            any(not price.is_finite() or price <= 0 for price in prices)
            or bar.low > min(bar.open, bar.close)
            or bar.high < max(bar.open, bar.close)
            or bar.high < bar.low
        ):
            raise ValueError("invalid OHLC")
    for previous, current in pairwise(bars):
        if previous.timestamp == current.timestamp:
            raise ValueError("duplicate timestamp")
        if previous.timestamp > current.timestamp:
            raise ValueError("out-of-order timestamp")
    return bars


def missing_minutes(
    bars: tuple[Candle, ...], start: datetime, end: datetime
) -> tuple[datetime, ...]:
    """Identify gaps in an expected continuous minute interval (session-aware callers)."""
    observed = {bar.timestamp.astimezone(UTC) for bar in bars}
    cursor = start.astimezone(UTC)
    missing: list[datetime] = []
    while cursor < end:
        if cursor not in observed:
            missing.append(cursor)
        cursor += timedelta(minutes=1)
    return tuple(missing)
