"""Typed, fail-closed application configuration."""

from contextvars import ContextVar
from datetime import time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Self, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from mnq_system.domain.values import RiskFraction, Symbol

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

# The YAML path is scoped per load so every caller uses one settings pipeline with
# identical precedence: constructor values, environment, YAML, then file secrets.
_CONFIG_PATH: ContextVar[Path] = ContextVar("mnq_config_path", default=DEFAULT_CONFIG_PATH)


class Environment(Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PAPER = "paper"
    LIVE = "live"


class Provider(Enum):
    DISABLED = "disabled"
    PENDING = "pending"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionSettings(StrictModel):
    timezone: str = "America/New_York"
    start: time = time(9, 30)
    end: time = time(16, 0)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown IANA timezone: {value}") from error
        return value

    @model_validator(mode="after")
    def ordered_times(self) -> Self:
        if self.start >= self.end:
            raise ValueError("session start must be before session end")
        return self


class CostSettings(StrictModel):
    commission_per_contract: Decimal = Field(default=Decimal(0), ge=0)
    slippage_points: Decimal = Field(default=Decimal(0), ge=0)


class SecretSettings(StrictModel):
    provider_token: SecretStr | None = Field(default=None, exclude=True, repr=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MNQ_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del cls, dotenv_settings
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_CONFIG_PATH.get()),
            file_secret_settings,
        )

    environment: Environment = Environment.DEVELOPMENT
    permitted_symbol: str = "MNQ"
    maximum_risk_fraction: Decimal = Decimal("0.01")
    maximum_daily_loss_fraction: Decimal = Decimal("0.03")
    maximum_drawdown_fraction: Decimal = Decimal("0.10")
    session: SessionSettings = SessionSettings()
    costs: CostSettings = CostSettings()
    data_provider: Provider = Provider.DISABLED
    broker: Provider = Provider.DISABLED
    live_trading_enabled: bool = False
    broker_submission_enabled: bool = False
    secrets: SecretSettings = SecretSettings()

    @field_validator(
        "maximum_risk_fraction",
        "maximum_daily_loss_fraction",
        "maximum_drawdown_fraction",
    )
    @classmethod
    def valid_fraction(cls, value: Decimal) -> Decimal:
        return RiskFraction(value).value

    @field_validator("permitted_symbol")
    @classmethod
    def valid_symbol(cls, value: str) -> str:
        return Symbol(value).value

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if self.live_trading_enabled or self.broker_submission_enabled:
            raise ValueError("live trading and broker submission are unavailable in this milestone")
        if self.environment is Environment.LIVE:
            raise ValueError("live environment is unavailable in this milestone")
        return self

    def safe_snapshot(self) -> dict[str, object]:
        """Return serialisable configuration with all secret fields excluded."""
        return self.model_dump(mode="json", exclude={"secrets"})


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load the given YAML file, then apply MNQ_ environment overrides.

    Every path uses the same settings pipeline, so environment variables always
    override file values regardless of which file was supplied.
    """
    if not path.is_file():
        raise ValueError(f"configuration file does not exist: {path}")
    loaded = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    token = _CONFIG_PATH.set(path)
    try:
        return Settings()
    finally:
        _CONFIG_PATH.reset(token)
