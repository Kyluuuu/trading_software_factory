"""Resumable read-only Alpaca historical ingestion, one symbol-month at a time."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from mnq_system.backtest.engine import Candle
from mnq_system.config.settings import load_settings
from mnq_system.data.alpaca import AlpacaHistoricalBars
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol

DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "META", "TSLA")
DEFAULT_ROOT = Path("data/alpaca/iex/raw/1min")


class BarSource(Protocol):
    def fetch(self, symbol: Symbol, start: datetime, end: datetime) -> tuple[Candle, ...]: ...


def months(start: datetime, end: datetime) -> tuple[tuple[int, int, datetime, datetime], ...]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("ordered timezone-aware range required")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    intervals: list[tuple[int, int, datetime, datetime]] = []
    while cursor < end:
        next_month = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12
            else cursor.replace(month=cursor.month + 1)
        )
        left, right = max(cursor, start), min(next_month, end)
        if left < right:
            intervals.append((cursor.year, cursor.month, left, right))
        cursor = next_month
    return tuple(intervals)


def ingest(
    source: BarSource,
    store: PartitionStore,
    symbols: tuple[Symbol, ...],
    start: datetime,
    end: datetime,
) -> int:
    """Fetch half-open month slices; a completed slice is never re-fetched."""
    intervals = months(start, end)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    completed = 0
    for symbol in symbols:
        for year, month, left, right in intervals:
            try:
                saved = store.interval(symbol, year, month)
                if saved == (left, right):
                    store.read(symbol, year, month)
                    continue
            except (ValueError, OSError):
                pass  # corrupted or interrupted partition is replaced atomically below
            # Alpaca's end is inclusive; subtract one minute to preserve half-open slices.
            # Using right - 1 minute also prevents next month's first bar being stored twice.
            last = right - timedelta(minutes=1)
            if last < left:
                bars: tuple[Candle, ...] = ()
            else:
                bars = source.fetch(symbol, left, last)
            store.write(symbol, year, month, left, right, bars)
            print(f"stored {symbol.value} {year:04d}-{month:02d}: {len(bars)} bars", flush=True)
            completed += 1
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Alpaca IEX 1-minute data ingestion")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        settings = load_settings()
        symbols = tuple(Symbol(value) for value in args.symbols)
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        print(
            f"IEX raw minute bars, range [{start.isoformat()}, {end.isoformat()}); "
            f"root={args.root}",
            flush=True,
        )
        count = ingest(
            AlpacaHistoricalBars(settings), PartitionStore(args.root), symbols, start, end
        )
        print(
            f"Completed {count} new symbol-month partitions; run coverage_report to inspect gaps."
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"ingestion failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
