"""Atomic Parquet month partitions with verified completion manifests."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from mnq_system.backtest.engine import Candle
from mnq_system.data.validation import validate_bars
from mnq_system.domain.values import Symbol

_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("open", pa.string()),
        ("high", pa.string()),
        ("low", pa.string()),
        ("close", pa.string()),
        ("volume", pa.int64()),
    ]
)


class PartitionStore:
    """Only a matching, checksum-verified manifest marks a partition complete."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def partition(self, symbol: Symbol, year: int, month: int) -> Path:
        if not 1 <= month <= 12 or not 1900 <= year <= 9999:
            raise ValueError("invalid partition date")
        return (
            self.root
            / f"symbol={symbol.value}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "bars.parquet"
        )

    def manifest(self, symbol: Symbol, year: int, month: int) -> Path:
        return self.partition(symbol, year, month).with_suffix(".json")

    def metadata(self, symbol: Symbol, year: int, month: int) -> dict[str, object] | None:
        path = self.manifest(symbol, year, month)
        if not path.exists():
            return None
        data = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(data, dict):
            raise ValueError(f"invalid manifest: {path}")
        return cast("dict[str, object]", data)

    def interval(self, symbol: Symbol, year: int, month: int) -> tuple[datetime, datetime] | None:
        data = self.metadata(symbol, year, month)
        if data is None:
            return None
        if (
            data.get("version") != 1
            or data.get("symbol") != symbol.value
            or data.get("feed") != "iex"
        ):
            raise ValueError("partition manifest metadata mismatch")
        start, end = data.get("start"), data.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError("invalid manifest interval")
        return datetime.fromisoformat(start), datetime.fromisoformat(end)

    def read(self, symbol: Symbol, year: int, month: int) -> tuple[Candle, ...]:
        path = self.partition(symbol, year, month)
        meta = self.metadata(symbol, year, month)
        if meta is None:
            if path.exists():
                raise ValueError("uncommitted parquet partition")
            return ()
        self.interval(symbol, year, month)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get(
            "sha256"
        ):
            raise ValueError(f"partition missing or checksum mismatch: {path}")
        parquet = pq.ParquetFile(path)
        reader = cast("Callable[[], pa.Table]", vars(type(parquet))["read"].__get__(parquet))
        table = reader()
        if not table.schema.equals(_SCHEMA):
            raise ValueError("unexpected parquet schema")
        columns = table.to_pydict()
        rows = tuple(
            Candle(
                symbol,
                timestamp,
                Decimal(open_),
                Decimal(high),
                Decimal(low),
                Decimal(close),
                volume,
            )
            for timestamp, open_, high, low, close, volume in zip(
                columns["timestamp"],
                columns["open"],
                columns["high"],
                columns["low"],
                columns["close"],
                columns["volume"],
                strict=True,
            )
        )
        validate_bars(rows)
        if meta.get("rows") != len(rows):
            raise ValueError("partition row count mismatch")
        return rows

    def write(
        self,
        symbol: Symbol,
        year: int,
        month: int,
        start: datetime,
        end: datetime,
        bars: tuple[Candle, ...],
    ) -> None:
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("invalid partition interval")
        validate_bars(bars)
        if any(
            bar.symbol != symbol
            or not start <= bar.timestamp < end
            or (bar.timestamp.year, bar.timestamp.month) != (year, month)
            for bar in bars
        ):
            raise ValueError("bars outside partition interval")
        path = self.partition(symbol, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {
                "timestamp": [bar.timestamp.astimezone(UTC) for bar in bars],
                "open": [str(bar.open) for bar in bars],
                "high": [str(bar.high) for bar in bars],
                "low": [str(bar.low) for bar in bars],
                "close": [str(bar.close) for bar in bars],
                "volume": [bar.volume for bar in bars],
            },
            schema=_SCHEMA,
        )
        temp_name: str | None = None
        manifest_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".parquet", delete=False
            ) as handle:
                temp_name = handle.name
            writer = cast("Callable[..., None]", vars(pq)["write_table"])
            writer(table, temp_name, compression="zstd")
            digest = hashlib.sha256(Path(temp_name).read_bytes()).hexdigest()
            metadata = {
                "version": 1,
                "symbol": symbol.value,
                "feed": "iex",
                "adjustment": "raw",
                "start": start.astimezone(UTC).isoformat(),
                "end": end.astimezone(UTC).isoformat(),
                "rows": len(bars),
                "sha256": digest,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, suffix=".json", encoding="utf-8", delete=False
            ) as handle:
                manifest_name = handle.name
                json.dump(metadata, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            temp_name = None
            os.replace(manifest_name, self.manifest(symbol, year, month))
            manifest_name = None
        finally:
            for name in (temp_name, manifest_name):
                if name is not None:
                    Path(name).unlink(missing_ok=True)
