"""Build local point-in-time features from verified raw monthly partitions."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mnq_system.data.ingest import DEFAULT_ROOT, DEFAULT_SYMBOLS, months
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.engine import generate
from mnq_system.features.lineage import fingerprint
from mnq_system.features.storage import FeatureStore

DEFAULT_FEATURE_ROOT = Path("data/features/iex/raw/1min")


def build(
    raw: PartitionStore,
    output: FeatureStore,
    symbols: tuple[Symbol, ...],
    start: datetime,
    end: datetime,
) -> int:
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    count = 0
    for symbol in symbols:
        for year, month, left, right in months(start, end):
            source = raw.metadata(symbol, year, month)
            if source is None or raw.interval(symbol, year, month) != (left, right):
                raise ValueError(
                    f"missing or mismatched raw partition {symbol.value} {year}-{month:02d}"
                )
            bars = raw.read(symbol, year, month)
            digest = source.get("sha256")
            if not isinstance(digest, str):
                raise ValueError("invalid raw source digest")
            try:
                saved = output.metadata(symbol, year, month)
                if (
                    saved is not None
                    and all(saved.get(key) == value for key, value in fingerprint().items())
                    and saved.get("source_sha256") == digest
                    and saved.get("start") == left.isoformat()
                    and saved.get("end") == right.isoformat()
                ):
                    output.read(symbol, year, month)
                    continue
            except (ValueError, OSError):
                pass
            # Month restarts are explicit. Session features never bridge a month boundary;
            # warm-up nulls preserve point-in-time correctness rather than backfilling.
            rows = generate(bars)
            output.write(symbol, year, month, rows, digest, (left, right))
            print(f"features {symbol.value} {year:04d}-{month:02d}: {len(rows)} rows", flush=True)
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build causal regular-session feature Parquet")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_FEATURE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        count = build(
            PartitionStore(args.raw_root),
            FeatureStore(args.root),
            tuple(Symbol(s) for s in args.symbols),
            start,
            end,
        )
        print(f"Completed {count} new feature partitions.")
        return 0
    except (ValueError, OSError) as error:
        print(f"feature build failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
