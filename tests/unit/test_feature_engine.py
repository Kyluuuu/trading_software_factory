"""Point-in-time feature invariants and stored partition integration."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from mnq_system.backtest.engine import Candle
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.build import build
from mnq_system.features.build import main as build_main
from mnq_system.features.engine import generate
from mnq_system.features.metadata import DEFINITIONS
from mnq_system.features.report import main as report_main
from mnq_system.features.report import summarize
from mnq_system.features.storage import FeatureStore


def bars(count: int, start: datetime | None = None) -> tuple[Candle, ...]:
    start = start or datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    return tuple(
        Candle(
            Symbol("AAPL"),
            start + timedelta(minutes=i),
            Decimal(100 + i % 7),
            Decimal(101 + i % 7),
            Decimal(99 + i % 7),
            Decimal(100 + i % 7),
            100 + i,
        )
        for i in range(count)
    )


def test_catalog_defines_all_feature_groups_and_formulas() -> None:
    names = {definition.name for definition in DEFINITIONS}
    assert len(names) >= 38
    assert len(names) == len(DEFINITIONS)
    assert all(
        d.description and d.formula and d.dependencies and d.lookback >= 0 for d in DEFINITIONS
    )
    assert {
        "atr",
        "adx",
        "session_vwap",
        "rsi",
        "donchian_high",
        "swing_structure",
        "session_progress",
        "volume_spike_score",
    } <= names


def test_prefix_invariance_and_minute_availability() -> None:
    source = bars(85)
    earlier = generate(source[:60])
    later = generate(source)
    assert earlier == later[:60]
    assert len(later) == 85
    for candle, row in zip(source, later, strict=True):
        assert row.timestamp == candle.timestamp
        assert row.available_at == candle.timestamp + timedelta(minutes=1)
        assert set(row.values) == {d.name for d in DEFINITIONS}
        assert all(value is None or value.is_finite() for value in row.values.values())
    assert later[0].values["returns"] is None
    assert later[0].values["session_vwap"] is not None
    assert later[25].values["rsi"] is not None
    changed = (
        *source[:55],
        Candle(
            Symbol("AAPL"),
            source[55].timestamp,
            Decimal("500"),
            Decimal("501"),
            Decimal("499"),
            Decimal("500"),
            5000,
        ),
        *source[56:],
    )
    assert generate(changed)[:55] == later[:55]
    assert later[0].values["donchian_high"] is None
    assert later[25].values["donchian_high"] is not None


def test_gaps_and_sessions_reset_rolling_state() -> None:
    first = bars(30)
    next_day = bars(3, datetime(2026, 1, 6, 14, 30, tzinfo=UTC))
    rows = generate(first + next_day)
    assert rows[30].values["returns"] is None
    assert rows[30].values["session_vwap"] is not None
    assert rows[30].values["rsi"] is None
    gapped = generate(first[:2] + first[3:])
    assert gapped[2].values["returns"] is None


def test_feature_commands_report_missing_data_and_reject_short_ranges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_main(["--days", "30"]) == 2
    assert "--days must be at least 365" in capsys.readouterr().err
    assert report_main(["--days", "30"]) == 2
    assert "--days must be at least 365" in capsys.readouterr().err
    assert (
        build_main(
            ["--raw-root", str(tmp_path), "--root", str(tmp_path / "features"), "--symbols", "AAPL"]
        )
        == 2
    )
    assert "missing or mismatched raw partition" in capsys.readouterr().err
    assert report_main(["--root", str(tmp_path), "--symbols", "AAPL"]) == 0
    assert "missing partitions=" in capsys.readouterr().out


def test_partition_build_resume_and_source_change(tmp_path: Path) -> None:
    raw, output = PartitionStore(tmp_path / "raw"), FeatureStore(tmp_path / "features")
    source = bars(65)
    start, end = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    raw.write(Symbol("AAPL"), 2026, 1, start, end, source)
    assert build(raw, output, (Symbol("AAPL"),), start, end) == 1
    assert build(raw, output, (Symbol("AAPL"),), start, end) == 0
    rows = output.read(Symbol("AAPL"), 2026, 1)
    assert len(rows) == 65
    assert rows[0].available_at == source[0].timestamp + timedelta(minutes=1)
    assert rows[0].values["rsi"] is None
    assert rows[30].values["rsi"] is not None
    report = summarize(output, (Symbol("AAPL"),), start, end)
    assert report[0].rows == 65
    assert report[0].invalid == 0
    assert report[0].first_timestamp == source[0].timestamp
    raw.write(Symbol("AAPL"), 2026, 1, start, end, bars(66))
    assert build(raw, output, (Symbol("AAPL"),), start, end) == 1
    assert len(output.read(Symbol("AAPL"), 2026, 1)) == 66
    output.partition(Symbol("AAPL"), 2026, 1).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        output.read(Symbol("AAPL"), 2026, 1)
    assert build(raw, output, (Symbol("AAPL"),), start, end) == 1
