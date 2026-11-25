"""Local label maturity, coverage and illustrative purged split report."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from mnq_system.data.ingest import DEFAULT_SYMBOLS, months
from mnq_system.domain.values import Symbol
from mnq_system.labels.build import DEFAULT_LABEL_ROOT
from mnq_system.labels.engine import LabelConfig, LabelRow
from mnq_system.labels.splits import purged_split
from mnq_system.labels.storage import LabelStore


def summarize(
    store: LabelStore, symbol: Symbol, start: datetime, end: datetime, config: LabelConfig
) -> tuple[LabelRow, ...]:
    rows: list[LabelRow] = []
    for year, month, _left, _right in months(start, end):
        if store.metadata(symbol, year, month) is None:
            raise ValueError(f"missing label partition {symbol.value} {year}-{month:02d}")
        rows.extend(store.read(symbol, year, month, config))
    if any(row.maturity_timestamp > end for row in rows):
        raise ValueError("unmatured label exceeds dataset cutoff")
    return tuple(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matured label and purged split readiness")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--direction-threshold", type=Decimal, default=Decimal("0.001"))
    parser.add_argument("--embargo-minutes", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        if args.days < 365 or args.embargo_minutes < 0:
            raise ValueError("days must be at least 365 and embargo nonnegative")
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        config = LabelConfig(direction_threshold=args.direction_threshold)
        barrier_count = len(config.targets) * len(config.stops) * len(config.horizons)
        print(
            f"Matured label report [{start.isoformat()}, {end.isoformat()}); "
            f"horizons={config.horizons}; barriers={barrier_count}"
        )
        for value in args.symbols:
            rows = summarize(LabelStore(args.root), Symbol(value), start, end, config)
            # Calendar dates are illustrative boundaries, not fitted optimizer parameters.
            validation = start + (end - start) * 3 / 4
            test = start + (end - start) * 9 / 10
            split = purged_split(rows, validation, test, timedelta(minutes=args.embargo_minutes))
            print(
                f"{value}: rows={len(rows)}, forward outcomes={len(rows) * len(config.horizons)}, "
                f"triple barriers={len(rows) * barrier_count}; "
                f"first={rows[0].prediction_timestamp if rows else None}; "
                f"last={rows[-1].prediction_timestamp if rows else None}"
            )
            print(
                f"  train={len(split.train)}, validation={len(split.validation)}, "
                f"test={len(split.test)}, purged={len(split.purged)}; "
                f"embargo={args.embargo_minutes}m"
            )
            if rows:
                sample = rows[0]
                outcome = sample.forward[5]
                print(
                    f"  sample prediction={sample.prediction_timestamp}, "
                    f"maturity={sample.maturity_timestamp}, "
                    f"return5={outcome.raw_return}, direction5={outcome.direction}"
                )
        return 0
    except (ValueError, OSError) as error:
        print(f"label report failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
