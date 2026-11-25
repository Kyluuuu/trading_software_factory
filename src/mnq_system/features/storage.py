"""Atomic versioned monthly feature Parquet partitions and source fingerprints."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from mnq_system.domain.values import Symbol
from mnq_system.features.asof import as_of
from mnq_system.features.engine import FeatureRow
from mnq_system.features.lineage import fingerprint
from mnq_system.features.metadata import DEFINITIONS, VERSION

_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        *((d.name, pa.float64()) for d in DEFINITIONS),
    ]
)


class FeatureStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def partition(self, symbol: Symbol, year: int, month: int) -> Path:
        if not 1 <= month <= 12 or not 1900 <= year <= 9999:
            raise ValueError("invalid feature partition date")
        return (
            self.root
            / f"symbol={symbol.value}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "features.parquet"
        )

    def manifest(self, symbol: Symbol, year: int, month: int) -> Path:
        return self.partition(symbol, year, month).with_suffix(".json")

    def metadata(self, symbol: Symbol, year: int, month: int) -> dict[str, object] | None:
        manifest = self.manifest(symbol, year, month)
        if not manifest.exists():
            if self.partition(symbol, year, month).exists():
                raise ValueError("uncommitted feature partition")
            return None
        data = cast("object", json.loads(manifest.read_text(encoding="utf-8")))
        if not isinstance(data, dict):
            raise ValueError("invalid feature manifest")
        return cast("dict[str, object]", data)

    def read(self, symbol: Symbol, year: int, month: int) -> tuple[FeatureRow, ...]:
        meta = self.metadata(symbol, year, month)
        if meta is None:
            return ()
        path = self.partition(symbol, year, month)
        if any(meta.get(key) != value for key, value in fingerprint().items()):
            raise ValueError("feature lineage mismatch")
        if (
            meta.get("version") != VERSION
            or meta.get("symbol") != symbol.value
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("sha256")
        ):
            raise ValueError("feature checksum or metadata mismatch")
        parquet = pq.ParquetFile(path)
        reader = cast("Callable[[], pa.Table]", vars(type(parquet))["read"].__get__(parquet))
        table = reader()
        if not table.schema.equals(_SCHEMA):
            raise ValueError("invalid feature schema")
        columns = table.to_pydict()
        rows: list[FeatureRow] = []
        previous: datetime | None = None
        for index in range(table.num_rows):
            at = columns["timestamp"][index]
            available = columns["available_at"][index]
            if (previous is not None and at <= previous) or available != at + timedelta(minutes=1):
                raise ValueError("inconsistent feature timestamps")
            previous = at
            values = {
                d.name: (
                    Decimal(str(columns[d.name][index]))
                    if columns[d.name][index] is not None
                    else None
                )
                for d in DEFINITIONS
            }
            if any(value is not None and not value.is_finite() for value in values.values()):
                raise ValueError("NaN or infinite feature")
            rows.append(FeatureRow(at, available, values))
        if meta.get("rows") != len(rows):
            raise ValueError("feature row count mismatch")
        return tuple(rows)

    def at_decision(
        self, symbol: Symbol, year: int, month: int, decision_time: datetime
    ) -> FeatureRow:
        """Guarded access for decisions; raw read() is for audits only."""
        return as_of(self.read(symbol, year, month), decision_time)

    def write(
        self,
        symbol: Symbol,
        year: int,
        month: int,
        rows: tuple[FeatureRow, ...],
        source_sha256: str,
        interval: tuple[datetime, datetime],
    ) -> None:
        previous: datetime | None = None
        for row in rows:
            if (
                row.timestamp.tzinfo is None
                or row.available_at != row.timestamp + timedelta(minutes=1)
                or (previous is not None and row.timestamp <= previous)
                or (row.timestamp.year, row.timestamp.month) != (year, month)
                or not interval[0] <= row.timestamp < interval[1]
                or set(row.values) != {d.name for d in DEFINITIONS}
                or any(v is not None and not v.is_finite() for v in row.values.values())
            ):
                raise ValueError("invalid point-in-time feature row")
            previous = row.timestamp
        path = self.partition(symbol, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {
                "timestamp": [row.timestamp.astimezone(UTC) for row in rows],
                "available_at": [row.available_at.astimezone(UTC) for row in rows],
                **{
                    d.name: [
                        float(value) if (value := row.values[d.name]) is not None else None
                        for row in rows
                    ]
                    for d in DEFINITIONS
                },
            },
            schema=_SCHEMA,
        )
        tmp: str | None = None
        meta_tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".parquet", delete=False
            ) as file:
                tmp = file.name
            writer = cast("Callable[..., None]", vars(pq)["write_table"])
            writer(table, tmp, compression="zstd")
            digest = hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
            metadata = {
                **fingerprint(),
                "symbol": symbol.value,
                "rows": len(rows),
                "source_sha256": source_sha256,
                "sha256": digest,
                "start": interval[0].astimezone(UTC).isoformat(),
                "end": interval[1].astimezone(UTC).isoformat(),
            }
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, suffix=".json", encoding="utf-8", delete=False
            ) as file:
                meta_tmp = file.name
                json.dump(metadata, file, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, path)
            tmp = None
            os.replace(meta_tmp, self.manifest(symbol, year, month))
            meta_tmp = None
        finally:
            for name in (tmp, meta_tmp):
                if name is not None:
                    Path(name).unlink(missing_ok=True)
