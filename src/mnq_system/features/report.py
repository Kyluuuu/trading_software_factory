"""Read-only validation report for locally generated feature partitions."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mnq_system.data.ingest import DEFAULT_SYMBOLS, months
from mnq_system.domain.values import Symbol
from mnq_system.features.build import DEFAULT_FEATURE_ROOT
from mnq_system.features.metadata import DEFINITIONS, VERSION
from mnq_system.features.storage import FeatureStore


@dataclass(frozen=True, slots=True)
class FeatureCoverage:
    symbol: str
    rows: int
    partitions: int
    missing_partitions: tuple[str, ...]
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    warmup_nulls: int
    invalid: int
    issues: tuple[str, ...]


def summarize(
    store: FeatureStore, symbols: tuple[Symbol, ...], start: datetime, end: datetime
) -> tuple[FeatureCoverage, ...]:
    reports: list[FeatureCoverage] = []
    for symbol in symbols:
        count = partitions = warmups = invalid = 0
        first: datetime | None = None
        last: datetime | None = None
        missing: list[str] = []
        issues: list[str] = []
        for year, month, left, right in months(start, end):
            label = f"{year:04d}-{month:02d}"
            try:
                meta = store.metadata(symbol, year, month)
                if meta is None:
                    missing.append(label)
                    continue
                if meta.get("version") != VERSION or (meta.get("start"), meta.get("end")) != (
                    left.isoformat(),
                    right.isoformat(),
                ):
                    raise ValueError("feature version or interval mismatch")
                rows = store.read(symbol, year, month)
            except (ValueError, OSError) as error:
                invalid += 1
                issues.append(f"{label}: {error}")
                continue
            partitions += 1
            count += len(rows)
            warmups += sum(value is None for row in rows for value in row.values.values())
            if rows:
                first = min(first, rows[0].timestamp) if first else rows[0].timestamp
                last = max(last, rows[-1].timestamp) if last else rows[-1].timestamp
        reports.append(
            FeatureCoverage(
                symbol.value,
                count,
                partitions,
                tuple(missing),
                first,
                last,
                warmups,
                invalid,
                tuple(issues),
            )
        )
    return tuple(reports)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate point-in-time feature coverage")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--root", type=Path, default=DEFAULT_FEATURE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        print(
            f"Feature version={VERSION}, definitions={len(DEFINITIONS)}, "
            f"window=[{start.isoformat()}, {end.isoformat()})"
        )
        for report in summarize(
            FeatureStore(args.root), tuple(Symbol(s) for s in args.symbols), start, end
        ):
            print(
                f"{report.symbol}: rows={report.rows}, partitions={report.partitions}, "
                f"null warm-up values={report.warmup_nulls}, invalid={report.invalid}"
            )
            print(f"  first={report.first_timestamp} last={report.last_timestamp}")
            print(f"  missing partitions={', '.join(report.missing_partitions) or 'none'}")
            print(f"  issues={'; '.join(report.issues) or 'none'}")
        return 0
    except (ValueError, OSError) as error:
        print(f"feature report failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
