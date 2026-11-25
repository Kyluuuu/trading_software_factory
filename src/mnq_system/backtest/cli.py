"""Command-line entry point for offline candle research only."""

import argparse
import csv
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from mnq_system.backtest.engine import Candle, ReplayEngine
from mnq_system.backtest.walkforward import evaluate
from mnq_system.config.settings import load_settings
from mnq_system.data.alpaca import AlpacaHistoricalBars
from mnq_system.domain.values import Symbol
from mnq_system.features.strategies import default_strategies

COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def read_csv(path: Path, symbol: Symbol) -> tuple[Candle, ...]:
    """Read one symbol's ordered, timezone-aware candle file without network access."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"CSV columns must be: {', '.join(COLUMNS)}")
        bars: list[Candle] = []
        for number, row in enumerate(reader, start=2):
            try:
                if None in row or any(row[column] is None for column in COLUMNS):
                    raise ValueError("missing or extra fields")
                raw_time = row["timestamp"]
                if raw_time is None:
                    raise ValueError("missing timestamp")
                timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError("timestamp requires timezone offset")
                bars.append(
                    Candle(
                        symbol,
                        timestamp,
                        Decimal(row["open"]),
                        Decimal(row["high"]),
                        Decimal(row["low"]),
                        Decimal(row["close"]),
                        int(row["volume"]),
                    )
                )
            except (ValueError, KeyError, TypeError, InvalidOperation) as error:
                raise ValueError(f"CSV row {number}: {error}") from error
    if len(bars) < 2:
        raise ValueError("CSV requires at least two candles")
    return tuple(bars)


def demo_candles() -> tuple[Candle, ...]:
    """Illustrative values only; intentionally not downloaded market data."""
    prices = ("100", "100", "99", "98", "97", "99", "101", "103", "102", "100", "98", "99")
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    bars: list[Candle] = []
    for i, price in enumerate(prices):
        opening = Decimal(price)
        closing = opening + (Decimal("0.5") if i % 2 else Decimal("-0.5"))
        bars.append(
            Candle(
                Symbol("AAPL"),
                start + timedelta(minutes=i),
                opening,
                max(opening, closing) + Decimal("0.1"),
                min(opening, closing) - Decimal("0.1"),
                closing,
                1000,
            )
        )
    return tuple(bars)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline one-bar strategy comparison; no broker orders"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="run synthetic example data")
    source.add_argument("--csv", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    source.add_argument(
        "--alpaca", action="store_true", help="fetch Alpaca IEX historical minute bars"
    )
    parser.add_argument("--start", help="Alpaca start ISO 8601 timestamp with timezone")
    parser.add_argument("--end", help="Alpaca end ISO 8601 timestamp with timezone (inclusive)")
    parser.add_argument(
        "--year", action="store_true", help="fetch previous 365 days of Alpaca bars"
    )
    parser.add_argument(
        "--walk-forward", action="store_true", help="rank prior windows and score later windows"
    )
    parser.add_argument("--train-bars", type=int, default=7800, help="rolling training candles")
    parser.add_argument("--test-bars", type=int, default=1950, help="rolling evaluation candles")
    parser.add_argument("--symbol", default="AAPL", help="symbol for CSV (default: AAPL)")
    parser.add_argument(
        "--equity", type=Decimal, default=Decimal("10000"), help="starting USD equity"
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        symbol = Symbol(args.symbol)
        if symbol.value not in settings.allowed_symbols:
            raise ValueError(f"symbol {symbol.value} is not in configured allowlist")
        if args.alpaca:
            if args.year and (args.start or args.end):
                raise ValueError("--year cannot be combined with --start or --end")
            if args.year:
                end = datetime.now(UTC) - timedelta(minutes=20)
                start = end - timedelta(days=365)
            else:
                if not args.start or not args.end:
                    raise ValueError("--alpaca requires --year or --start and --end")
                start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
                end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
            bars = AlpacaHistoricalBars(settings).fetch(symbol, start, end)
            print(f"Alpaca IEX minute bars: {symbol.value}, {len(bars)} candles")
        elif args.demo:
            bars = demo_candles()
            print("SYNTHETIC demo data")
        else:
            if args.start or args.end or args.year:
                raise ValueError("--year, --start and --end require --alpaca")
            bars = read_csv(args.csv, symbol)
            print(f"CSV data: {args.csv}")
        print(
            "OFFLINE one-bar candle replay; NOT live trading. Full-range ranking uses hindsight.\n"
        )
        engine = ReplayEngine(settings)
        strategies = default_strategies()
        if args.walk_forward:
            windows = evaluate(
                engine,
                strategies,
                bars,
                args.equity,
                train_bars=args.train_bars,
                test_bars=args.test_bars,
            )
            print(
                f"Rolling out-of-sample windows: {len(windows)}; "
                "supervised outcome model refit each window"
            )
            for i, window in enumerate(windows, start=1):
                leaders = (
                    ", ".join(
                        f"{name} {weight:.0%}" for name, weight in window.allocations if weight > 0
                    )
                    or "none (cash)"
                )
                print(
                    f"  Window {i}: prior-bar ranking {leaders}; "
                    f"test bars {window.test_start}-{window.test_end - 1}"
                )
                for result in window.results:
                    print(
                        f"    {result.name}: test net ${result.metrics.net_profit:.2f}, "
                        f"{result.metrics.number_of_trades} trades"
                    )
                print(
                    f"    Retrained model: {window.model_training_trades} closed training trades; "
                    f"approved patterns: {len(window.model_approved)}"
                )
                for result in window.model_results:
                    print(
                        f"    ML gate {result.name}: test net ${result.metrics.net_profit:.2f}, "
                        f"{result.metrics.number_of_trades} trades"
                    )
                chosen = window.selected_result
                print(
                    f"    Selected out-of-sample: {chosen.name} ${chosen.metrics.net_profit:.2f}"
                    if chosen
                    else "    Selected out-of-sample: cash (none qualified)"
                )
            print("  Each test strategy is independently replayed; allocations are not executed.\n")
        rankings = engine.compare(strategies, bars, args.equity)
        for rank, item in enumerate(rankings, start=1):
            metrics = item.result.metrics
            print(f"#{rank} {item.result.name} | suggested allocation: {item.allocation:.1%}")
            print(f"  Net profit: ${metrics.net_profit:.2f} | Trades: {metrics.number_of_trades}")
            print(f"  Win rate: {metrics.win_rate:.1%} | Profit factor: {metrics.profit_factor}")
            print(
                f"  Sharpe (per trade): {metrics.sharpe_ratio} | "
                f"Max drawdown: {metrics.maximum_drawdown:.1%}"
            )
            print(f"  Average trade return: {metrics.average_trade_return:.3%}")
        return 0
    except (OSError, ValueError) as error:
        print(f"backtest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
