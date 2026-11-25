"""Atomic Parquet outcome partitions with reproducible input and formula lineage."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from mnq_system.domain.values import Symbol
from mnq_system.labels.engine import LabelConfig, LabelRow, Outcome

VERSION = 1
_SCHEMA = pa.schema(
    [
        ("prediction_timestamp", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("maturity_timestamp", pa.timestamp("us", tz="UTC")),
        ("outcomes", pa.string()),
    ]
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lineage(config: LabelConfig) -> dict[str, object]:
    source = Path(__file__).with_name("engine.py").read_bytes()
    return {
        "version": VERSION,
        "formula_hash": _hash(source),
        "configuration_hash": _hash(
            json.dumps(
                {
                    "threshold": str(config.direction_threshold),
                    "targets": list(map(str, config.targets)),
                    "stops": list(map(str, config.stops)),
                    "horizons": config.horizons,
                },
                sort_keys=True,
            ).encode()
        ),
        "generator_version": "labels-1.0",
    }


def _outcome_json(outcome: Outcome) -> dict[str, str | int]:
    return {
        key: value.isoformat()
        if isinstance(value, datetime)
        else str(value)
        if isinstance(value, Decimal)
        else value
        for key, value in asdict(outcome).items()
    }


def _decode(data: dict[str, object]) -> Outcome:
    return Outcome(
        datetime.fromisoformat(str(data["label_timestamp"])),
        datetime.fromisoformat(str(data["maturity_timestamp"])),
        Decimal(str(data["raw_return"])),
        Decimal(str(data["volatility_adjusted_return"]))
        if data["volatility_adjusted_return"] is not None
        else None,
        Decimal(str(data["atr_adjusted_return"]))
        if data["atr_adjusted_return"] is not None
        else None,
        str(data["direction"]),
        str(data["barrier"]),
        datetime.fromisoformat(str(data["trigger_timestamp"])),
        int(str(data["holding_minutes"])),
    )


class LabelStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def partition(self, symbol: Symbol, year: int, month: int) -> Path:
        if not 1 <= month <= 12 or not 1900 <= year <= 9999:
            raise ValueError("invalid label partition date")
        return (
            self.root
            / f"symbol={symbol.value}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "labels.parquet"
        )

    def manifest(self, symbol: Symbol, year: int, month: int) -> Path:
        return self.partition(symbol, year, month).with_suffix(".json")

    def metadata(self, symbol: Symbol, year: int, month: int) -> dict[str, object] | None:
        path = self.manifest(symbol, year, month)
        if not path.exists():
            if self.partition(symbol, year, month).exists():
                raise ValueError("uncommitted label partition")
            return None
        data = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(data, dict):
            raise ValueError("invalid label manifest")
        return cast("dict[str, object]", data)

    def read(
        self, symbol: Symbol, year: int, month: int, config: LabelConfig
    ) -> tuple[LabelRow, ...]:
        meta = self.metadata(symbol, year, month)
        if meta is None:
            return ()
        path = self.partition(symbol, year, month)
        if any(meta.get(k) != v for k, v in _lineage(config).items()):
            raise ValueError("label lineage mismatch")
        if not path.is_file() or _hash(path.read_bytes()) != meta.get("sha256"):
            raise ValueError("label checksum mismatch")
        parquet = pq.ParquetFile(path)
        reader = cast("Callable[[], pa.Table]", vars(type(parquet))["read"].__get__(parquet))
        table = reader()
        if not table.schema.equals(_SCHEMA):
            raise ValueError("invalid label schema")
        columns = table.to_pydict()
        rows: list[LabelRow] = []
        for i in range(table.num_rows):
            payload = json.loads(columns["outcomes"][i])
            forward = {int(k): _decode(v) for k, v in payload["forward"].items()}
            barriers = {
                (Decimal(t), Decimal(s), int(h)): _decode(v) for t, s, h, v in payload["barriers"]
            }
            prediction = columns["prediction_timestamp"][i]
            available = columns["available_at"][i]
            maturity = columns["maturity_timestamp"][i]
            if (
                available != prediction
                or maturity <= available
                or (rows and prediction <= rows[-1].prediction_timestamp)
                or maturity
                != max(v.maturity_timestamp for v in (*forward.values(), *barriers.values()))
            ):
                raise ValueError("invalid label maturity or timestamp")
            rows.append(LabelRow(prediction, available, available, maturity, forward, barriers))
        if meta.get("rows") != len(rows):
            raise ValueError("label row count mismatch")
        return tuple(rows)

    def current(
        self, symbol: Symbol, year: int, month: int, sources: dict[str, str], config: LabelConfig
    ) -> bool:
        try:
            meta = self.metadata(symbol, year, month)
            if meta is None or meta.get("sources") != sources:
                return False
            self.read(symbol, year, month, config)
            return True
        except (ValueError, OSError):
            return False

    def write(
        self,
        symbol: Symbol,
        year: int,
        month: int,
        rows: tuple[LabelRow, ...],
        sources: dict[str, str],
        config: LabelConfig,
    ) -> None:
        previous: datetime | None = None
        for row in rows:
            if (
                row.prediction_timestamp.tzinfo is None
                or row.available_at != row.prediction_timestamp
                or row.maturity_timestamp <= row.available_at
                or (previous is not None and row.prediction_timestamp <= previous)
                or (row.prediction_timestamp.year, row.prediction_timestamp.month) != (year, month)
            ):
                raise ValueError("invalid label row")
            previous = row.prediction_timestamp
        path = self.partition(symbol, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads = [
            json.dumps(
                {
                    "forward": {str(k): _outcome_json(v) for k, v in row.forward.items()},
                    "barriers": [
                        [str(t), str(s), h, _outcome_json(v)]
                        for (t, s, h), v in row.barriers.items()
                    ],
                },
                sort_keys=True,
            )
            for row in rows
        ]
        table = pa.Table.from_pydict(
            {
                "prediction_timestamp": [r.prediction_timestamp for r in rows],
                "available_at": [r.available_at for r in rows],
                "maturity_timestamp": [r.maturity_timestamp for r in rows],
                "outcomes": payloads,
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
            metadata = {
                **_lineage(config),
                "symbol": symbol.value,
                "sources": sources,
                "rows": len(rows),
                "sha256": _hash(Path(tmp).read_bytes()),
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
