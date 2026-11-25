"""Regression tests for point-in-time access, lineage and research readiness."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnq_system.backtest.engine import Candle
from mnq_system.config.settings import Settings
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.asof import as_of, join_as_of
from mnq_system.features.build import build
from mnq_system.features.engine import generate
from mnq_system.features.lineage import fingerprint
from mnq_system.features.storage import FeatureStore
from mnq_system.features.training_readiness import assess
from mnq_system.features.training_readiness import main as readiness_main


def candle(at: datetime) -> Candle:
    return Candle(
        Symbol("AAPL"), at, Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"), 50
    )


def test_asof_rejects_future_and_naive_decisions_and_keeps_last_known_row() -> None:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    rows = generate((candle(start), candle(start + timedelta(minutes=1))))
    with pytest.raises(ValueError, match="not yet available"):
        as_of(rows, start)
    with pytest.raises(ValueError, match="timezone"):
        as_of(rows, start.replace(tzinfo=None))
    with pytest.raises(ValueError, match="not yet available"):
        join_as_of(rows, (start,))
    assert as_of(rows, start + timedelta(minutes=1)) == rows[0]
    assert as_of(rows, start + timedelta(minutes=2)) == rows[1]
    assert join_as_of(rows, (start + timedelta(minutes=1),)) == (rows[0],)
    with pytest.raises(ValueError, match="ordered"):
        join_as_of(rows[::-1], (start + timedelta(minutes=2),))


def test_partition_lineage_rebuilds_when_formula_or_config_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, output = PartitionStore(tmp_path / "raw"), FeatureStore(tmp_path / "features")
    symbol = Symbol("AAPL")
    start, end = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    raw.write(symbol, 2026, 1, start, end, (candle(start + timedelta(hours=14, minutes=30)),))
    assert build(raw, output, (symbol,), start, end) == 1
    assert build(raw, output, (symbol,), start, end) == 0
    original = fingerprint()
    metadata = output.metadata(symbol, 2026, 1)
    assert metadata is not None
    assert all(metadata[key] == original[key] for key in original)
    minute = start + timedelta(hours=14, minutes=30)
    with pytest.raises(ValueError, match="not yet available"):
        output.at_decision(symbol, 2026, 1, minute)
    assert output.at_decision(symbol, 2026, 1, minute + timedelta(minutes=1)).timestamp == minute
    from mnq_system.features import lineage

    monkeypatch.setattr(
        lineage, "FEATURE_CONFIGURATION", {**lineage.FEATURE_CONFIGURATION, "gap_policy": "changed"}
    )
    with pytest.raises(ValueError, match="lineage"):
        output.read(symbol, 2026, 1)
    assert build(raw, output, (symbol,), start, end) == 1
    monkeypatch.undo()
    monkeypatch.setattr(lineage, "formula_digest", lambda: "changed-formula")
    with pytest.raises(ValueError, match="lineage"):
        output.read(symbol, 2026, 1)
    assert build(raw, output, (symbol,), start, end) == 1


def test_readiness_rejects_stale_raw_source(tmp_path: Path) -> None:
    symbol = Symbol("AAPL")
    raw, output = PartitionStore(tmp_path / "raw"), FeatureStore(tmp_path / "features")
    start, end = datetime(2025, 11, 28, tzinfo=UTC), datetime(2025, 12, 1, tzinfo=UTC)
    at = datetime(2025, 11, 28, 14, 30, tzinfo=UTC)
    raw.write(symbol, 2025, 11, start, end, (candle(at),))
    build(raw, output, (symbol,), start, end)
    raw.write(symbol, 2025, 11, start, end, (candle(at), candle(at + timedelta(minutes=1))))
    result = assess(raw, output, symbol, start, end, Decimal("0"))
    assert result.status == "NOT TRAINABLE"
    assert any("source checksum mismatch" in reason for reason in result.reasons)


def test_readiness_command_reports_status_and_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert readiness_main(["--days", "30"]) == 2
    assert "at least 365" in capsys.readouterr().err
    assert (
        readiness_main(["--symbols", "AAPL", "--raw-root", str(tmp_path), "--root", str(tmp_path)])
        == 0
    )
    output = capsys.readouterr().out
    assert "AAPL: NOT TRAINABLE" in output
    assert "admissible=" in output
    assert "blocked=" in output
    assert "actual Alpaca receipt timestamps are unavailable" in output


def test_threshold_and_missing_minute_readiness(tmp_path: Path) -> None:
    symbol = Symbol("AAPL")
    raw, output = PartitionStore(tmp_path / "raw"), FeatureStore(tmp_path / "features")
    start, end = datetime(2025, 11, 28, tzinfo=UTC), datetime(2025, 12, 1, tzinfo=UTC)
    at = datetime(2025, 11, 28, 14, 30, tzinfo=UTC)
    raw.write(symbol, 2025, 11, start, end, (candle(at), candle(at + timedelta(minutes=2))))
    build(raw, output, (symbol,), start, end)
    report = assess(raw, output, symbol, start, end, Decimal("99.0"))
    assert report.expected_minutes == 210
    assert report.rows == 2
    assert report.gap_minutes == 208
    assert report.row_coverage == Decimal(200) / Decimal(210)
    assert report.gap_percent == Decimal(20800) / Decimal(210)
    assert report.missing_minute_impact > 0
    assert report.null_percent > 0
    assert report.warmup_percent > 0
    assert report.feature_coverage["open"] == report.row_coverage
    assert "open" in report.blocked
    assert "feature non-null coverage" in report.blocked["open"]
    assert report.status == "NOT TRAINABLE"
    assert "symbol coverage" in report.reasons
    with pytest.raises(ValidationError):
        Settings.model_validate({"minimum_symbol_coverage": "101"})
