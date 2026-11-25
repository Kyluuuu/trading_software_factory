"""Read-only Alpaca stock minute bars for offline research (IEX feed)."""

import json
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mnq_system.backtest.engine import Candle
from mnq_system.config.settings import Settings
from mnq_system.domain.values import Symbol

_URL = "https://data.alpaca.markets/v2/stocks"
_MAX_PAGES = 1000


class AlpacaHistoricalBars:
    """Fetch only historical stock bars, never account or order endpoints."""

    def __init__(self, settings: Settings) -> None:
        key, secret = settings.secrets.alpaca_api_key, settings.secrets.alpaca_api_secret
        if not key or not secret or not key.get_secret_value() or not secret.get_secret_value():
            raise ValueError("Alpaca paper credentials are required in .env")
        self._key = key.get_secret_value()
        self._secret = secret.get_secret_value()
        self._symbols = frozenset(settings.allowed_symbols)

    def fetch(self, symbol: Symbol, start: datetime, end: datetime) -> tuple[Candle, ...]:
        if symbol.value not in self._symbols:
            raise ValueError("symbol is not in the research allowlist")
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or start.utcoffset() is None
            or end.utcoffset() is None
            or start >= end
        ):
            raise ValueError("start and end must be ordered timezone-aware datetimes")
        bars: list[Candle] = []
        seen: set[str] = set()
        token: str | None = None
        for _ in range(_MAX_PAGES):
            params = {
                "timeframe": "1Min",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": "10000",
                "feed": "iex",
                "adjustment": "raw",
            }
            if token is not None:
                params["page_token"] = token
            url = f"{_URL}/{symbol.value}/bars?{urlencode(params)}"
            request = Request(  # noqa: S310 - fixed HTTPS host and allowlisted symbol
                url,
                headers={
                    "APCA-API-KEY-ID": self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                    "Accept": "application/json",
                },
                method="GET",
            )
            payload: object = None
            for attempt in range(4):
                try:
                    with urlopen(request, timeout=10) as response:  # noqa: S310  # nosec B310
                        payload = cast("object", json.load(response))
                    break
                except HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                        raise
                    time.sleep(float(2**attempt))
                except (URLError, TimeoutError):
                    if attempt == 3:
                        raise
                    time.sleep(float(2**attempt))
            if not isinstance(payload, dict):
                raise ValueError("invalid Alpaca bars response")
            data = cast("dict[str, object]", payload)
            raw_bars = data.get("bars")
            if not isinstance(raw_bars, list):
                raise ValueError("invalid Alpaca bars response")
            for item in cast("list[object]", raw_bars):
                if not isinstance(item, dict):
                    raise ValueError("invalid Alpaca bar")
                bar = cast("dict[str, object]", item)
                try:
                    raw_time = bar["t"]
                    if not isinstance(raw_time, str):
                        raise ValueError("invalid timestamp")
                    timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    raw_volume = bar["v"]
                    if type(raw_volume) is not int:
                        raise ValueError("invalid volume")
                    candle = Candle(
                        symbol,
                        timestamp,
                        Decimal(str(bar["o"])),
                        Decimal(str(bar["h"])),
                        Decimal(str(bar["l"])),
                        Decimal(str(bar["c"])),
                        raw_volume,
                    )
                except (KeyError, ValueError, TypeError, InvalidOperation) as error:
                    raise ValueError("invalid Alpaca bar") from error
                if not start <= timestamp <= end or (bars and bars[-1].timestamp >= timestamp):
                    raise ValueError("Alpaca bars outside range or not strictly ordered")
                bars.append(candle)
            next_token = data.get("next_page_token")
            if next_token is None:
                return tuple(bars)
            if not isinstance(next_token, str) or not next_token or next_token in seen:
                raise ValueError("invalid or repeated Alpaca page token")
            seen.add(next_token)
            token = next_token
        raise ValueError("Alpaca pagination exceeded safety limit")
