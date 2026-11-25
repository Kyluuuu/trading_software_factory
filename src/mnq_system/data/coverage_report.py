"""Read-only XNYS regular-session coverage of locally stored Alpaca IEX bars."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from mnq_system.data.calendar import xnys
from mnq_system.data.ingest import DEFAULT_ROOT, DEFAULT_SYMBOLS, months
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol


@dataclass(frozen=True, slots=True)
class Coverage:
    symbol: str
    expected_sessions: int
    actual_sessions: int
    total_bars: int
    expected_bars: int
    missing_bars: int
    coverage_percent: Decimal
    missing_dates: tuple[date, ...]
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    issues: tuple[str, ...]


def summarize(store: PartitionStore, symbol: Symbol, start: datetime, end: datetime) -> Coverage:
    """Count only expected regular-session minutes; out-of-session rows are identified."""
    intervals = months(start, end)
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    calendar = xnys()
    first_day, last_day = start.date(), (end - timedelta(microseconds=1)).date()
    sessions = calendar.sessions_in_range(first_day, last_day)
    observed: set[datetime] = set()
    issues: list[str] = []
    all_times: list[datetime] = []
    for year, month, left, right in intervals:
        try:
            interval = store.interval(symbol, year, month)
            if interval is None:
                issues.append(f"missing partition {year:04d}-{month:02d}")
                continue
            if interval != (left, right):
                issues.append(f"interval mismatch {year:04d}-{month:02d}")
                continue
            rows = store.read(symbol, year, month)
        except (ValueError, OSError) as error:
            issues.append(f"invalid partition {year:04d}-{month:02d}: {error}")
            continue
        for row in rows:
            at = row.timestamp.astimezone(UTC)
            if at in observed:
                issues.append(f"duplicate timestamp {at.isoformat()}")
            observed.add(at)
            all_times.append(at)
    expected: set[datetime] = set()
    actual_sessions = 0
    missing_dates: list[date] = []
    for session in sessions:
        minutes = {
            minute.to_pydatetime()
            for minute in calendar.session_minutes(session)
            if start <= minute.to_pydatetime() < end
        }
        if not minutes:
            continue
        expected.update(minutes)
        if observed.intersection(minutes):
            actual_sessions += 1
        else:
            missing_dates.append(session.date())
    unexpected = observed - expected
    if unexpected:
        issues.append(f"{len(unexpected)} out-of-session or outside-range bars")
    absent = expected - observed
    if absent:
        issues.append(f"{len(absent)} missing regular-session timestamps")
    total = len(expected.intersection(observed))
    return Coverage(
        symbol.value,
        sum(
            1
            for session in sessions
            if any(
                start <= minute.to_pydatetime() < end
                for minute in calendar.session_minutes(session)
            )
        ),
        actual_sessions,
        len(all_times),
        len(expected),
        len(absent),
        Decimal(100) * Decimal(total) / Decimal(len(expected)) if expected else Decimal(0),
        tuple(missing_dates),
        min(all_times) if all_times else None,
        max(all_times) if all_times else None,
        tuple(issues),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Alpaca IEX regular-session minute coverage")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        print(f"XNYS regular-session coverage [{start.isoformat()}, {end.isoformat()})")
        for value in args.symbols:
            report = summarize(PartitionStore(args.root), Symbol(value), start, end)
            print(
                f"{report.symbol}: expected sessions={report.expected_sessions}, "
                f"actual sessions={report.actual_sessions}, total bars={report.total_bars}, "
                f"missing bars={report.missing_bars}, coverage={report.coverage_percent:.2f}%"
            )
            print(f"  first={report.first_timestamp} last={report.last_timestamp}")
            print(f"  missing dates={', '.join(map(str, report.missing_dates)) or 'none'}")
            print(f"  data quality issues={'; '.join(report.issues) or 'none'}")
        return 0
    except (ValueError, OSError) as error:
        print(f"coverage failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
