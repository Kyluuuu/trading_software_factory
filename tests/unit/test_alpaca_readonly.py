"""Alpaca access must remain read-only and use paper endpoints."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from mnq_system.config.settings import SecretSettings, Settings, load_settings
from mnq_system.reporting.alpaca import AlpacaReadOnlyClient


class Response(BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_read_only_assets_and_account(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[Any] = []

    def fake_open(request: Any, *, timeout: float) -> Response:
        requests.append(request)
        assert timeout == 10
        if request.full_url.endswith("/assets?status=active"):
            return Response(
                json.dumps(
                    [
                        {"symbol": "AAPL", "tradable": True},
                        {"symbol": "SPY", "tradable": False},
                        {"symbol": "MNQ", "tradable": True},
                    ]
                ).encode()
            )
        return Response(b'{"equity":"10000","last_equity":"10100"}')

    monkeypatch.setattr("mnq_system.reporting.alpaca.urlopen", fake_open)
    settings = Settings.model_validate(
        {
            "secrets": {
                "alpaca_api_key": "paper-key",  # pragma: allowlist secret
                "alpaca_api_secret": "paper-secret",  # pragma: allowlist secret
            }
        }
    )
    client = AlpacaReadOnlyClient(settings)
    assert client.tradable_symbols() == ("AAPL",)
    assert client.account_equity() == ("10000", "10100")
    assert len(requests) == 2
    for request in requests:
        assert request.get_method() == "GET"
        assert request.full_url.startswith("https://paper-api.alpaca.markets/v2/")
        assert request.get_header("Apca-api-key-id") == "paper-key"


def test_missing_credentials_fail_closed() -> None:
    empty = Settings.model_construct(secrets=SecretSettings())
    with pytest.raises(ValueError, match="credentials"):
        AlpacaReadOnlyClient(empty)


def test_dotenv_credentials_are_loaded_but_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MNQ_SECRETS__ALPACA_API_KEY=paper-test-key\n"  # pragma: allowlist secret
        "MNQ_SECRETS__ALPACA_API_SECRET=paper-test-secret\n"
    )
    settings = load_settings(Path(__file__).resolve().parents[2] / "config/default.yaml")
    key = settings.secrets.alpaca_api_key
    assert key is not None
    assert key.get_secret_value() == "paper-test-key"
    assert "paper-test-key" not in repr(settings)
    assert "secrets" not in settings.safe_snapshot()
