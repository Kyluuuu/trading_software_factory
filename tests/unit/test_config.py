from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnq_system.config.settings import Environment, Settings, load_settings


def test_configuration_defaults_fail_closed() -> None:
    settings = Settings()
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.live_trading_enabled is False
    assert settings.broker_submission_enabled is False


def test_configuration_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(unknown_setting=True)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    "override",
    [
        {"live_trading_enabled": True},
        {"broker_submission_enabled": True},
        {"environment": "live"},
    ],
)
def test_live_configuration_is_unavailable(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="unavailable"):
        Settings.model_validate(override)


def test_secrets_are_redacted_from_repr_and_snapshots() -> None:
    settings = Settings.model_validate({"secrets": {"provider_token": "test-secret-token"}})
    assert "test-secret-token" not in repr(settings)
    assert "test-secret-token" not in str(settings.safe_snapshot())
    assert "secrets" not in settings.safe_snapshot()


def test_default_yaml_loads() -> None:
    settings = load_settings()
    assert settings.permitted_symbol == "MNQ"


def test_environment_overrides_committed_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNQ_PERMITTED_SYMBOL", "MES")
    assert load_settings().permitted_symbol == "MES"


def test_environment_overrides_custom_configuration_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "permitted_symbol: MNQ\nmaximum_risk_fraction: '0.02'\n",
        encoding="utf-8",
    )
    assert load_settings(custom).maximum_risk_fraction == Decimal("0.02")
    monkeypatch.setenv("MNQ_PERMITTED_SYMBOL", "MES")
    monkeypatch.setenv("MNQ_MAXIMUM_RISK_FRACTION", "0.005")
    settings = load_settings(custom)
    assert settings.permitted_symbol == "MES"
    assert settings.maximum_risk_fraction == Decimal("0.005")


def test_custom_configuration_file_cannot_enable_live_trading(tmp_path: Path) -> None:
    custom = tmp_path / "live.yaml"
    custom.write_text("live_trading_enabled: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unavailable"):
        load_settings(custom)


def test_custom_configuration_file_rejects_unknown_fields(tmp_path: Path) -> None:
    custom = tmp_path / "unknown.yaml"
    custom.write_text("unknown_setting: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_settings(custom)


def test_loader_rejects_missing_and_non_mapping_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_settings(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_settings(invalid)
