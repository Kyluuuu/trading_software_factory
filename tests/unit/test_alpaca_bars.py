"""Read-only market-data tests with mocked HTTPS responses."""

import json
from datetime import UTC, datetime
from email.message import Message
from io import BytesIO
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from mnq_system.config.settings import Settings
from mnq_system.data.alpaca import AlpacaHistoricalBars
from mnq_system.domain.values import Symbol


class Response(BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def settings() -> Settings:
    return Settings.model_validate(
        {
            "secrets": {
                "alpaca_api_key": "paper-key",  # pragma: allowlist secret
                "alpaca_api_secret": "paper-secret",  # pragma: allowlist secret
            }
        }
    )


def test_fetches_all_pages_of_minute_bars_using_only_get(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[Request] = []

    def fake_open(request: Request, *, timeout: float) -> Response:
        assert timeout == 10
        requests.append(request)
        pages = [
            {
                "bars": [
                    {"t": "2026-01-05T15:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 40}
                ],
                "next_page_token": "next +/=",
            },
            {
                "bars": [
                    {"t": "2026-01-05T16:00:00Z", "o": 101, "h": 102, "l": 100, "c": 101, "v": 50}
                ],
                "next_page_token": None,
            },
        ]
        return Response(json.dumps(pages[len(requests) - 1]).encode())

    monkeypatch.setattr("mnq_system.data.alpaca.urlopen", fake_open)
    bars = AlpacaHistoricalBars(settings()).fetch(
        Symbol("AAPL"),
        datetime(2026, 1, 5, 15, tzinfo=UTC),
        datetime(2026, 1, 5, 16, tzinfo=UTC),
    )
    assert len(bars) == 2
    assert bars[0].volume == 40
    assert bars[1].close == 101
    for request in requests:
        assert request.get_method() == "GET"
        url = request.full_url
        assert isinstance(url, str)
        assert urlsplit(url).netloc == "data.alpaca.markets"
        assert urlsplit(url).path == "/v2/stocks/AAPL/bars"
        assert parse_qs(urlsplit(url).query)["feed"] == ["iex"]
        assert request.get_header("Apca-api-key-id") == "paper-key"
    next_url = requests[1].full_url
    assert isinstance(next_url, str)
    assert parse_qs(urlsplit(next_url).query)["page_token"] == ["next +/="]


@pytest.mark.parametrize(
    "response",
    [
        {"bars": None},
        {"bars": [{"t": "2026-01-05T15:00:00Z", "o": -1, "h": 101, "l": 99, "c": 100, "v": 4}]},
        {
            "bars": [{"t": "2026-01-05T15:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 4}],
            "next_page_token": "loop",
        },
    ],
)
def test_invalid_responses_fail_closed(monkeypatch: pytest.MonkeyPatch, response: object) -> None:
    def fake_open(_request: Request, *, timeout: float) -> Response:
        assert timeout == 10
        return Response(json.dumps(response).encode())

    monkeypatch.setattr("mnq_system.data.alpaca.urlopen", fake_open)
    with pytest.raises(ValueError, match=r"invalid|repeated|not strictly ordered"):
        AlpacaHistoricalBars(settings()).fetch(
            Symbol("AAPL"),
            datetime(2026, 1, 5, 15, tzinfo=UTC),
            datetime(2026, 1, 5, 16, tzinfo=UTC),
        )


def test_transient_rate_limit_retries_same_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.error import HTTPError

    attempts = 0
    slept: list[float] = []

    def fake_open(_request: Request, *, timeout: float) -> Response:
        nonlocal attempts
        attempts += 1
        assert timeout == 10
        if attempts == 1:
            raise HTTPError("https://data.alpaca.markets", 429, "rate limited", Message(), None)
        return Response(json.dumps({"bars": [], "next_page_token": None}).encode())

    monkeypatch.setattr("mnq_system.data.alpaca.urlopen", fake_open)
    monkeypatch.setattr("mnq_system.data.alpaca.time.sleep", slept.append)
    client = AlpacaHistoricalBars(settings())
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    assert client.fetch(Symbol("AAPL"), start, datetime(2026, 1, 5, 16, tzinfo=UTC)) == ()
    assert attempts == 2
    assert slept == [1.0]


def test_year_range_can_paginate_without_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[Request] = []

    def fake_open(request: Request, *, timeout: float) -> Response:
        assert timeout == 10
        requests.append(request)
        minute = "2025-11-25T15:00:00Z" if len(requests) == 1 else "2026-11-24T15:00:00Z"
        return Response(
            json.dumps(
                {
                    "bars": [{"t": minute, "o": 100, "h": 101, "l": 99, "c": 100, "v": 100}],
                    "next_page_token": "page2" if len(requests) == 1 else None,
                }
            ).encode()
        )

    monkeypatch.setattr("mnq_system.data.alpaca.urlopen", fake_open)
    result = AlpacaHistoricalBars(settings()).fetch(
        Symbol("AAPL"), datetime(2025, 11, 25, tzinfo=UTC), datetime(2026, 11, 25, tzinfo=UTC)
    )
    assert len(result) == 2
    assert len(requests) == 2


def test_rejects_invalid_range_and_unsupported_symbol_before_network() -> None:
    client = AlpacaHistoricalBars(settings())
    start = datetime(2026, 1, 5, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match="ordered"):
        client.fetch(Symbol("AAPL"), start, start)
    with pytest.raises(ValueError, match="allowlist"):
        client.fetch(Symbol("MNQ"), start, datetime(2026, 1, 5, 16, tzinfo=UTC))
