"""Read-only discovery against Alpaca's paper Trading API.

No network request is performed during construction, and no order endpoint exists.
"""

import json
from typing import cast
from urllib.request import Request, urlopen

from mnq_system.config.settings import Settings

_PAPER_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaReadOnlyClient:
    def __init__(self, settings: Settings) -> None:
        key = settings.secrets.alpaca_api_key
        secret = settings.secrets.alpaca_api_secret
        if not key or not secret or not key.get_secret_value() or not secret.get_secret_value():
            raise ValueError("Alpaca paper credentials are required")
        self._key = key.get_secret_value()
        self._secret = secret.get_secret_value()
        self._symbols = frozenset(settings.allowed_symbols)

    def _get(self, path: str) -> object:
        request = Request(  # noqa: S310 - fixed paper HTTPS endpoint
            f"{_PAPER_URL}{path}",
            headers={
                "APCA-API-KEY-ID": self._key,
                "APCA-API-SECRET-KEY": self._secret,
                "Accept": "application/json",
            },
            method="GET",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310  # nosec B310
            return cast("object", json.load(response))

    def tradable_symbols(self) -> tuple[str, ...]:
        """Return only active, tradable symbols in the configured allowlist."""
        assets = self._get("/assets?status=active")
        if not isinstance(assets, list):
            raise ValueError("invalid Alpaca asset response")
        found: set[str] = set()
        for item in cast("list[object]", assets):
            if not isinstance(item, dict):
                raise ValueError("invalid Alpaca asset response")
            asset = cast("dict[str, object]", item)
            symbol, tradable = asset.get("symbol"), asset.get("tradable")
            if isinstance(symbol, str) and tradable is True and symbol in self._symbols:
                found.add(symbol)
        return tuple(sorted(found))

    def account_equity(self) -> tuple[str, str]:
        """Read display-only equity and previous-close equity (not a risk snapshot)."""
        account = self._get("/account")
        if not isinstance(account, dict):
            raise ValueError("invalid Alpaca account response")
        values = cast("dict[str, object]", account)
        equity, previous = values.get("equity"), values.get("last_equity")
        if not isinstance(equity, str) or not isinstance(previous, str):
            raise ValueError("invalid Alpaca account equity response")
        return equity, previous
