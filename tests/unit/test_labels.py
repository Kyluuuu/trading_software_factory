"""Leakage, barrier, split, and storage regressions for offline labels."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from mnq_system.backtest.engine import Candle
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.engine import FeatureRow
from mnq_system.features.engine import generate as generate_features
from mnq_system.features.storage import FeatureStore
from mnq_system.labels.build import build
from mnq_system.labels.build import main as build_main
from mnq_system.labels.engine import LabelConfig, generate
from mnq_system.labels.report import main as report_main
from mnq_system.labels.report import summarize
from mnq_system.labels.splits import purged_split
from mnq_system.labels.storage import LabelStore

START = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def bar(
    i: int, open_: str = "100", high: str = "100", low: str = "100", close: str = "100"
) -> Candle:
    return Candle(
        Symbol("AAPL"),
        START + timedelta(minutes=i),
        Decimal(open_),
        Decimal(high),
        Decimal(low),
        Decimal(close),
        100,
    )


def feature(
    i: int, *, atr: Decimal | None = Decimal("1"), vol: Decimal | None = Decimal("0.01")
) -> FeatureRow:
    at = START + timedelta(minutes=i)
    return FeatureRow(at, at + timedelta(minutes=1), {"atr": atr, "realized_volatility": vol})


def test_entry_is_after_feature_availability_and_labels_mature_after_exit() -> None:
    bars = (bar(0), *(bar(i) for i in range(1, 63)))
    row = generate((feature(0),), bars, LabelConfig())[0]
    assert row.prediction_timestamp == START + timedelta(minutes=1)
    assert row.available_at == row.prediction_timestamp
    assert row.entry_timestamp == row.prediction_timestamp
    forward = row.forward[5]
    assert forward.label_timestamp == START + timedelta(minutes=5)
    assert forward.maturity_timestamp == START + timedelta(minutes=6)
    assert forward.raw_return == 0
    assert forward.direction == "neutral"
    assert len(row.barriers) == 36
    assert all(
        v.maturity_timestamp >= v.trigger_timestamp + timedelta(minutes=1)
        for v in row.barriers.values()
    )
    assert all(v.maturity_timestamp > row.prediction_timestamp for v in row.barriers.values())


def test_first_hit_both_barriers_and_gap_open_take_stop_conservatively() -> None:
    bars = (bar(0), bar(1, high="102", low="98"), *(bar(i) for i in range(2, 63)))
    row = generate((feature(0),), bars, LabelConfig())[0]
    hit = row.barriers[(Decimal("0.0025"), Decimal("0.0025"), 5)]
    assert hit.barrier == "stop"
    assert hit.trigger_timestamp == START + timedelta(minutes=1)
    assert hit.maturity_timestamp == START + timedelta(minutes=2)
    assert hit.raw_return == Decimal("-0.0025")
    assert hit.holding_minutes == 1
    gapped = (
        bar(0),
        bar(1),
        bar(2, open_="98", high="99", low="97", close="98"),
        *(bar(i) for i in range(3, 63)),
    )
    assert generate((feature(0),), gapped, LabelConfig())[0].barriers[
        (Decimal("0.0025"), Decimal("0.0025"), 5)
    ].raw_return == Decimal("-0.02")


def test_incomplete_horizon_and_missing_minute_are_not_labeled() -> None:
    full = tuple(bar(i) for i in range(60))
    assert generate((feature(0),), full, LabelConfig()) == ()  # needs bar t+60
    missing = (bar(0), *(bar(i) for i in range(1, 62) if i != 4))
    assert generate((feature(0),), missing, LabelConfig()) == ()
    with pytest.raises(ValueError, match="availability"):
        generate((FeatureRow(START, START + timedelta(minutes=2), {}),), full, LabelConfig())


def test_future_changes_do_not_change_matured_short_label() -> None:
    bars = tuple(bar(i) for i in range(62))
    changed = (*bars[:15], bar(15, close="101", high="101"), *bars[16:])
    a = generate((feature(0),), bars, LabelConfig())[0]
    b = generate((feature(0),), changed, LabelConfig())[0]
    assert a.forward[5] == b.forward[5]
    assert a.forward[15] != b.forward[15]


def test_purging_uses_maturity_and_embargo() -> None:
    bars = tuple(bar(i) for i in range(80))
    labels = generate(tuple(feature(i) for i in range(16)), bars, LabelConfig())
    split = purged_split(
        labels, START + timedelta(minutes=8), START + timedelta(minutes=12), timedelta(minutes=2)
    )
    assert all(row.maturity_timestamp <= START + timedelta(minutes=8) for row in split.train)
    assert all(
        row.prediction_timestamp >= START + timedelta(minutes=10) for row in split.validation
    )
    assert all(row.maturity_timestamp <= START + timedelta(minutes=12) for row in split.validation)
    assert all(row.prediction_timestamp >= START + timedelta(minutes=14) for row in split.test)
    assert len(split.purged) > 0


def test_build_and_report_from_verified_partitions(tmp_path: Path) -> None:
    raw = PartitionStore(tmp_path / "raw")
    features = FeatureStore(tmp_path / "features")
    labels = LabelStore(tmp_path / "labels")
    symbol = Symbol("AAPL")
    start, end = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    bars = tuple(bar(i) for i in range(63))
    raw.write(symbol, 2026, 1, start, end, bars)
    source = raw.metadata(symbol, 2026, 1)
    assert source is not None
    features.write(symbol, 2026, 1, generate_features(bars), str(source["sha256"]), (start, end))
    config = LabelConfig()
    assert build(raw, features, labels, (symbol,), start, end, config) == 1
    assert build(raw, features, labels, (symbol,), start, end, config) == 0
    rows = summarize(labels, symbol, start, end, config)
    assert len(rows) == 3
    assert rows[0].maturity_timestamp == START + timedelta(minutes=61)


def test_label_commands_handle_missing_inputs_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_main(["--days", "30"]) == 2
    assert "at least 365" in capsys.readouterr().err
    assert report_main(["--days", "30"]) == 2
    assert "at least 365" in capsys.readouterr().err
    assert (
        build_main(
            [
                "--symbols",
                "AAPL",
                "--raw-root",
                str(tmp_path),
                "--feature-root",
                str(tmp_path),
                "--root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "missing raw interval" in capsys.readouterr().err
    assert report_main(["--symbols", "AAPL", "--root", str(tmp_path)]) == 2
    assert "missing label partition" in capsys.readouterr().err


def test_parquet_lineage_and_resume(tmp_path: Path) -> None:
    rows = generate((feature(0),), tuple(bar(i) for i in range(61)), LabelConfig())
    store = LabelStore(tmp_path)
    store.write(Symbol("AAPL"), 2026, 1, rows, {"raw": "abc", "features": "def"}, LabelConfig())
    assert store.read(Symbol("AAPL"), 2026, 1, LabelConfig()) == rows
    assert store.current(Symbol("AAPL"), 2026, 1, {"raw": "abc", "features": "def"}, LabelConfig())
    assert not store.current(
        Symbol("AAPL"), 2026, 1, {"raw": "changed", "features": "def"}, LabelConfig()
    )
    store.partition(Symbol("AAPL"), 2026, 1).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.read(Symbol("AAPL"), 2026, 1, LabelConfig())
