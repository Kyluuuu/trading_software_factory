"""Offline research command tests."""

from pathlib import Path

import pytest

from mnq_system.backtest.cli import main


def test_demo_prints_labeled_results(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo"]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC" in output
    assert "breakout" in output
    assert "mean_reversion" in output
    assert "Net profit" in output
    assert "NOT live trading" in output


def test_csv_runs_with_provided_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "candles.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-05T15:00:00+00:00,100,100,100,100,200\n"
        "2026-01-05T15:01:00+00:00,100,101,100,101,200\n"
        "2026-01-05T15:02:00+00:00,101,103,101,103,200\n"
        "2026-01-05T15:03:00+00:00,103,105,103,105,200\n",
        encoding="utf-8",
    )
    assert main(["--csv", str(path), "--symbol", "AAPL"]) == 0
    output = capsys.readouterr().out
    assert "CSV" in output
    assert "Trades" in output


def test_alpaca_mode_uses_requested_range_without_orders(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import UTC, datetime

    from mnq_system.backtest.cli import demo_candles
    from mnq_system.backtest.engine import Candle
    from mnq_system.domain.values import Symbol

    calls: list[tuple[str, datetime, datetime]] = []

    def fake_fetch(
        _client: object, symbol: Symbol, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        calls.append((symbol.value, start, end))
        return demo_candles()

    monkeypatch.setattr("mnq_system.backtest.cli.AlpacaHistoricalBars.fetch", fake_fetch)
    assert (
        main(
            [
                "--alpaca",
                "--symbol",
                "AAPL",
                "--start",
                "2026-01-05T15:00:00Z",
                "--end",
                "2026-01-05T16:00:00Z",
            ]
        )
        == 0
    )
    assert calls == [
        ("AAPL", datetime(2026, 1, 5, 15, tzinfo=UTC), datetime(2026, 1, 5, 16, tzinfo=UTC))
    ]
    assert "Alpaca IEX" in capsys.readouterr().out


def test_year_shortcut_requests_365_days(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import datetime, timedelta

    from mnq_system.backtest.cli import demo_candles
    from mnq_system.backtest.engine import Candle
    from mnq_system.domain.values import Symbol

    calls: list[tuple[datetime, datetime]] = []

    def fake_fetch(
        _client: object, _symbol: Symbol, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        calls.append((start, end))
        return demo_candles()

    monkeypatch.setattr("mnq_system.backtest.cli.AlpacaHistoricalBars.fetch", fake_fetch)
    assert main(["--alpaca", "--year"]) == 0
    assert len(calls) == 1
    assert calls[0][1] - calls[0][0] == timedelta(days=365)
    assert "Alpaca IEX" in capsys.readouterr().out
    assert main(["--alpaca", "--year", "--start", "2025-01-01T00:00:00Z"]) == 2


def test_rolling_report_prints_test_performance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo", "--walk-forward", "--train-bars", "6", "--test-bars", "3"]) == 0
    output = capsys.readouterr().out
    assert "Rolling out-of-sample windows" in output
    assert "test net" in output
    assert "Selected out-of-sample" in output
    assert "Retrained model" in output
    assert "ML gate" in output


def test_invalid_csv_fails_with_helpful_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n2026-01-05T15:00:00,100,100,100,100,100\n",
        encoding="utf-8",
    )
    assert main(["--csv", str(path)]) == 2
    assert "timezone" in capsys.readouterr().err
