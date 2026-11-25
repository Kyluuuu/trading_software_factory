"""Phase 1 local ingestion and calendar-aware coverage integration tests."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from mnq_system.backtest.engine import Candle
from mnq_system.data.coverage_report import main as coverage_main
from mnq_system.data.coverage_report import summarize
from mnq_system.data.ingest import ingest
from mnq_system.data.storage import PartitionStore
from mnq_system.data.validation import validate_bars
from mnq_system.domain.values import Symbol


def bar(at: datetime, *, volume: int = 1) -> Candle:
    return Candle(
        Symbol("AAPL"), at, Decimal("10"), Decimal("11"), Decimal("9"), Decimal("10"), volume
    )


class Source:
    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []
        self.fail = False

    def fetch(self, symbol: Symbol, start: datetime, end: datetime) -> tuple[Candle, ...]:
        assert symbol.value == "AAPL"
        self.calls.append((start, end))
        if self.fail:
            raise OSError("temporary failure")
        return (bar(start), bar(start + timedelta(minutes=1)))


def test_resume_only_missing_partitions_and_atomic_roundtrip(tmp_path: Path) -> None:
    source = Source()
    store = PartitionStore(tmp_path)
    start = datetime(2026, 1, 30, 15, tzinfo=UTC)
    end = datetime(2026, 2, 2, 16, tzinfo=UTC)
    assert ingest(source, store, (Symbol("AAPL"),), start, end) == 2
    assert len(source.calls) == 2
    assert list(tmp_path.rglob("*.parquet"))
    source.fail = True
    assert ingest(source, store, (Symbol("AAPL"),), start, end) == 0
    assert len(source.calls) == 2
    assert store.read(Symbol("AAPL"), 2026, 1)[0].timestamp == start
    store.partition(Symbol("AAPL"), 2026, 2).unlink()
    with pytest.raises(OSError, match="temporary failure"):
        ingest(source, store, (Symbol("AAPL"),), start, end)
    assert store.read(Symbol("AAPL"), 2026, 1)
    source.fail = False
    assert ingest(source, store, (Symbol("AAPL"),), start, end) == 1
    assert len(source.calls) == 4


def test_validation_rejects_bad_order_duplicate_and_invalid_values() -> None:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    assert validate_bars((bar(start), bar(start + timedelta(minutes=2))))
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars((bar(start), bar(start)))
    with pytest.raises(ValueError, match="out-of-order"):
        validate_bars((bar(start + timedelta(minutes=1)), bar(start)))
    with pytest.raises(ValueError, match="invalid OHLCV"):
        bar(start, volume=-1)
    with pytest.raises(ValueError, match="invalid OHLCV"):
        Candle(Symbol("AAPL"), start, Decimal("10"), Decimal("9"), Decimal("8"), Decimal("10"), 1)


def test_corrupt_partition_is_detected_and_refetched(tmp_path: Path) -> None:
    source = Source()
    store = PartitionStore(tmp_path)
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    end = start + timedelta(days=1)
    ingest(source, store, (Symbol("AAPL"),), start, end)
    store.partition(Symbol("AAPL"), 2026, 1).write_bytes(b"corrupt")
    report = summarize(store, Symbol("AAPL"), start, end)
    assert any("checksum mismatch" in issue for issue in report.issues)
    assert ingest(source, store, (Symbol("AAPL"),), start, end) == 1
    assert len(source.calls) == 2


def test_coverage_command_reports_missing_local_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert coverage_main(["--root", str(tmp_path), "--symbols", "SPY", "--days", "365"]) == 0
    output = capsys.readouterr().out
    assert "SPY: expected sessions=" in output
    assert "missing dates=" in output
    assert "first=None last=None" in output
    assert "missing partition" in output


def test_session_coverage_counts_early_close_and_missing_dates(tmp_path: Path) -> None:
    start = datetime(2025, 11, 27, tzinfo=UTC)
    end = datetime(2025, 11, 30, tzinfo=UTC)
    store = PartitionStore(tmp_path)
    report = summarize(store, Symbol("AAPL"), start, end)
    assert report.expected_sessions == 1
    assert report.actual_sessions == 0
    assert report.expected_bars == 210
    assert report.missing_bars == 210
    assert report.missing_dates == (date(2025, 11, 28),)
    at = datetime(2025, 11, 28, 14, 30, tzinfo=UTC)
    store.write(Symbol("AAPL"), 2025, 11, start, end, (bar(at),))
    report = summarize(store, Symbol("AAPL"), start, end)
    assert report.actual_sessions == 1
    assert report.total_bars == 1
    assert report.missing_bars == 209
    assert report.coverage_percent == Decimal(100) / Decimal(210)
    assert report.missing_dates == ()
