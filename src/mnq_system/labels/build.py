"""Build offline supervision from verified raw bars and point-in-time feature rows."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from mnq_system.data.ingest import DEFAULT_ROOT, DEFAULT_SYMBOLS, months
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.build import DEFAULT_FEATURE_ROOT
from mnq_system.features.lineage import fingerprint
from mnq_system.features.storage import FeatureStore
from mnq_system.labels.engine import LabelConfig, generate
from mnq_system.labels.storage import LabelStore

DEFAULT_LABEL_ROOT = Path("data/labels/iex/raw/1min")


def build(
    raw: PartitionStore,
    features: FeatureStore,
    output: LabelStore,
    symbols: tuple[Symbol, ...],
    start: datetime,
    end: datetime,
    config: LabelConfig,
) -> int:
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    count = 0
    intervals = months(start, end)
    for symbol in symbols:
        for year, month, left, right in intervals:
            if raw.interval(symbol, year, month) != (left, right):
                raise ValueError(f"missing raw interval {symbol.value} {year}-{month:02d}")
            raw_bars = raw.read(symbol, year, month)
            raw_meta = raw.metadata(symbol, year, month)
            feature_meta = features.metadata(symbol, year, month)
            if (
                raw_meta is None
                or feature_meta is None
                or any(feature_meta.get(key) != value for key, value in fingerprint().items())
                or feature_meta.get("source_sha256") != raw_meta.get("sha256")
            ):
                raise ValueError(f"missing or stale features {symbol.value} {year}-{month:02d}")
            feature_rows = features.read(symbol, year, month)
            if (
                feature_meta.get("start") != left.isoformat()
                or feature_meta.get("end") != right.isoformat()
            ):
                raise ValueError("feature interval mismatch")
            # The next partition supplies outcomes only. A complete horizon never
            # crosses the session close, but reading ahead permits final-day labels.
            next_month = (
                datetime(year + 1, 1, 1, tzinfo=UTC)
                if month == 12
                else datetime(year, month + 1, 1, tzinfo=UTC)
            )
            following = next((part for part in intervals if part[2] == next_month), None)
            future_bars = ()
            future_hash = "none"
            if following is not None:
                fy, fm, fleft, fright = following
                if raw.interval(symbol, fy, fm) != (fleft, fright):
                    raise ValueError("missing next raw partition")
                future_bars = tuple(
                    bar
                    for bar in raw.read(symbol, fy, fm)
                    if bar.timestamp < right + timedelta(minutes=max(config.horizons))
                )
                future_meta = raw.metadata(symbol, fy, fm)
                if future_meta is None or not isinstance(future_meta.get("sha256"), str):
                    raise ValueError("invalid next raw partition")
                future_hash = str(future_meta["sha256"])
            raw_hash, feature_hash = raw_meta.get("sha256"), feature_meta.get("sha256")
            if not isinstance(raw_hash, str) or not isinstance(feature_hash, str):
                raise ValueError("invalid source hashes")
            sources: dict[str, str] = {
                "raw": raw_hash,
                "features": feature_hash,
                "next_raw": future_hash,
            }
            if output.current(symbol, year, month, sources, config):
                continue
            labels = generate(feature_rows, raw_bars + future_bars, config)
            output.write(symbol, year, month, labels, sources, config)
            print(f"labels {symbol.value} {year:04d}-{month:02d}: {len(labels)} rows", flush=True)
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build matured offline supervision labels")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--direction-threshold", type=Decimal, default=Decimal("0.001"))
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        count = build(
            PartitionStore(args.raw_root),
            FeatureStore(args.feature_root),
            LabelStore(args.root),
            tuple(Symbol(s) for s in args.symbols),
            start,
            end,
            LabelConfig(direction_threshold=args.direction_threshold),
        )
        print(f"Completed {count} new label partitions.")
        return 0
    except (ValueError, OSError) as error:
        print(f"label build failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
