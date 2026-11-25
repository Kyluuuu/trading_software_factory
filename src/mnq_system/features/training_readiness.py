"""Read-only research readiness and coverage gates; not a training dataset builder."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from mnq_system.config.settings import load_settings
from mnq_system.data.calendar import xnys
from mnq_system.data.ingest import DEFAULT_ROOT, DEFAULT_SYMBOLS, months
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.build import DEFAULT_FEATURE_ROOT
from mnq_system.features.metadata import DEFINITIONS
from mnq_system.features.storage import FeatureStore

# Audit exclusions: bar-based VWAP is not tape VWAP; zero-denominator and
# semantic/convention issues are held back pending independent reference checks.
_BLOCKED: dict[str, str] = {
    "atr": "ATR initialization and gap convention require reference validation",
    "adx": "ADX metadata lookback differs from implementation",
    "ema_distance": "EMA initialization convention requires reference validation",
    "ema_slope": "EMA initialization and slope require reference validation",
    "trend_strength": "absolute EMA distance is not independently validated trend strength",
    "bollinger_distance": "undefined zero-variance ratio is encoded as zero",
    "z_score": "undefined zero-variance ratio is encoded as zero",
    "session_vwap": "observed IEX bar-typical-price VWAP is not complete-market VWAP",
    "vwap_distance": "depends on approximate incomplete-market VWAP",
    "vwap_deviation": "depends on approximate VWAP; zero variance encoded as zero",
    "relative_volume": "not normalized for time of day",
    "volume_spike_score": "zero prior variance encoded as zero",
    "swing_structure": "confirmed at t but refers to center at t-2; backdating leaks",
    "support_distance": "prior rolling low is not validated swing support",
    "resistance_distance": "prior rolling high is not validated swing resistance",
}


@dataclass(frozen=True, slots=True)
class Readiness:
    symbol: str
    status: str
    reasons: tuple[str, ...]
    expected_minutes: int
    rows: int
    gap_minutes: int
    row_coverage: Decimal
    gap_percent: Decimal
    null_percent: Decimal
    warmup_percent: Decimal
    missing_minute_impact: int
    feature_coverage: dict[str, Decimal]
    admissible: tuple[str, ...]
    blocked: dict[str, str]


def _percent(part: int, total: int) -> Decimal:
    return Decimal(100) * Decimal(part) / Decimal(total) if total else Decimal(0)


def assess(
    raw: PartitionStore,
    features: FeatureStore,
    symbol: Symbol,
    start: datetime,
    end: datetime,
    minimum_symbol_coverage: Decimal,
) -> Readiness:
    """Fail closed on inconsistent partitions; no missing minute or null is imputed.

    Null % counts null cells over observed rows * number of definitions. Warmup %
    is the fraction of cells expected null due to insufficient consecutive bars.
    Missing-minute impact counts observed rows in warmup after an intraday gap.
    """
    if not minimum_symbol_coverage.is_finite() or not Decimal(
        0
    ) <= minimum_symbol_coverage <= Decimal(100):
        raise ValueError("minimum symbol coverage must be between 0 and 100")
    expected: set[datetime] = set()
    calendar = xnys()
    for session in calendar.sessions_in_range(
        start.date(), (end - timedelta(microseconds=1)).date()
    ):
        expected.update(
            minute.to_pydatetime()
            for minute in calendar.session_minutes(session)
            if start <= minute.to_pydatetime() < end
        )
    observed: set[datetime] = set()
    raw_times: set[datetime] = set()
    issues: list[str] = []
    populated = nulls = warmups = gap_impact = 0
    non_null: dict[str, int] = dict.fromkeys((d.name for d in DEFINITIONS), 0)
    streak = 0
    recovery = 0
    previous: datetime | None = None
    max_lookback = max(d.lookback for d in DEFINITIONS)
    timezone = ZoneInfo("America/New_York")
    for year, month, left, right in months(start, end):
        try:
            if raw.interval(symbol, year, month) != (left, right):
                raise ValueError("missing or mismatched raw partition")
            raw_rows = raw.read(symbol, year, month)
            if features.metadata(symbol, year, month) is None:
                raise ValueError("missing feature partition")
            feature_rows = features.read(symbol, year, month)
            raw_meta = raw.metadata(symbol, year, month)
            feature_meta = features.metadata(symbol, year, month)
            if (
                raw_meta is None
                or feature_meta is None
                or feature_meta.get("source_sha256") != raw_meta.get("sha256")
            ):
                raise ValueError("feature/raw source checksum mismatch")
        except (ValueError, OSError) as error:
            issues.append(f"{year:04d}-{month:02d}: {error}")
            continue
        raw_times.update(
            bar.timestamp.astimezone(UTC) for bar in raw_rows if bar.timestamp in expected
        )
        for row in feature_rows:
            at = row.timestamp.astimezone(UTC)
            if at not in expected or at in observed:
                issues.append(f"unexpected or duplicate feature timestamp {at.isoformat()}")
                continue
            observed.add(at)
            populated += 1
            local_day = at.astimezone(timezone).date()
            prior_day = previous.astimezone(timezone).date() if previous else None
            gap = (
                previous is not None
                and prior_day == local_day
                and at - previous > timedelta(minutes=1)
            )
            streak = (
                streak + 1 if previous is not None and at - previous == timedelta(minutes=1) else 1
            )
            if gap:
                recovery = max_lookback
            elif prior_day != local_day:
                recovery = 0
            if recovery:
                gap_impact += 1
                recovery -= 1
            previous = at
            for definition in DEFINITIONS:
                value = row.values[definition.name]
                if value is None:
                    nulls += 1
                else:
                    non_null[definition.name] += 1
                if streak < definition.lookback and value is None:
                    warmups += 1
        if {row.timestamp.astimezone(UTC) for row in feature_rows} != {
            bar.timestamp.astimezone(UTC) for bar in raw_rows if bar.timestamp in expected
        }:
            issues.append(f"{year:04d}-{month:02d}: feature/raw timestamp mismatch")
    if observed != raw_times:
        issues.append("feature/raw regular-session timestamps differ")
    missing = len(expected - observed)
    coverage = _percent(len(observed), len(expected))
    if coverage < minimum_symbol_coverage:
        issues.append("symbol coverage")
    if not expected:
        issues.append("no expected sessions")
    # Excluded columns remain visible in feature coverage to explain null prevalence.
    feature_coverage = {name: _percent(count, len(expected)) for name, count in non_null.items()}
    blocked = dict(_BLOCKED)
    for definition in DEFINITIONS:
        name = definition.name
        if name not in blocked and feature_coverage[name] < minimum_symbol_coverage:
            blocked[name] = "feature non-null coverage below minimum symbol coverage"
    admissible = tuple(d.name for d in DEFINITIONS if d.name not in blocked)
    if not admissible:
        issues.append("no admissible features meet coverage threshold")
    return Readiness(
        symbol.value,
        "TRAINABLE (RESEARCH ONLY)" if not issues else "NOT TRAINABLE",
        tuple(issues),
        len(expected),
        populated,
        missing,
        coverage,
        _percent(missing, len(expected)),
        _percent(nulls, populated * len(DEFINITIONS)),
        _percent(warmups, populated * len(DEFINITIONS)),
        gap_impact,
        feature_coverage,
        admissible,
        blocked,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research feature eligibility and coverage gates")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_FEATURE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.days < 365:
            raise ValueError("--days must be at least 365")
        settings = load_settings()
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        print(
            f"Research as-of feature readiness [{start.isoformat()}, {end.isoformat()}); "
            f"minimum_symbol_coverage={settings.minimum_symbol_coverage}%"
        )
        for value in args.symbols:
            result = assess(
                PartitionStore(args.raw_root),
                FeatureStore(args.root),
                Symbol(value),
                start,
                end,
                settings.minimum_symbol_coverage,
            )
            print(
                f"{result.symbol}: {result.status}; rows={result.rows}/{result.expected_minutes}; "
                f"row coverage={result.row_coverage:.2f}%; gap={result.gap_percent:.2f}%; "
                f"null={result.null_percent:.2f}%; warmup={result.warmup_percent:.2f}%; "
                f"missing-minute impact={result.missing_minute_impact}; "
                f"missing bars={result.gap_minutes}"
            )
            print(f"  reasons={'; '.join(result.reasons) or 'none'}")
            print(f"  admissible={', '.join(result.admissible)}")
            blocked = "; ".join(f"{name}: {reason}" for name, reason in result.blocked.items())
            coverage = "; ".join(
                f"{name}:{percent:.2f}%" for name, percent in result.feature_coverage.items()
            )
            print(f"  blocked={blocked}")
            print(f"  feature coverage={coverage}")
        print(
            "Research-only: actual Alpaca receipt timestamps are unavailable; "
            "never assume live availability."
        )
        return 0
    except (ValueError, OSError) as error:
        print(f"readiness failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
