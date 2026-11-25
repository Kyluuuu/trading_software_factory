"""Causal prediction joins and matured forward/barrier outcomes for offline supervision."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from math import sqrt
from zoneinfo import ZoneInfo

from mnq_system.backtest.engine import Candle
from mnq_system.data.calendar import xnys
from mnq_system.data.validation import validate_bars
from mnq_system.features.asof import as_of
from mnq_system.features.engine import FeatureRow

HORIZONS = (5, 15, 30, 60)
BARRIERS = (Decimal("0.0025"), Decimal("0.005"), Decimal("0.01"))


@dataclass(frozen=True, slots=True)
class LabelConfig:
    direction_threshold: Decimal = Decimal("0.001")
    targets: tuple[Decimal, ...] = BARRIERS
    stops: tuple[Decimal, ...] = BARRIERS
    horizons: tuple[int, ...] = HORIZONS

    def __post_init__(self) -> None:
        if (
            not self.direction_threshold.is_finite()
            or self.direction_threshold < 0
            or not self.targets
            or not self.stops
            or not self.horizons
            or any(not v.is_finite() or v <= 0 or v >= 1 for v in (*self.targets, *self.stops))
            or any(type(v) is not int or v < 1 for v in self.horizons)
            or len(set(self.horizons)) != len(self.horizons)
        ):
            raise ValueError("invalid label configuration")


@dataclass(frozen=True, slots=True)
class Outcome:
    label_timestamp: datetime
    maturity_timestamp: datetime
    raw_return: Decimal
    volatility_adjusted_return: Decimal | None
    atr_adjusted_return: Decimal | None
    direction: str
    barrier: str
    trigger_timestamp: datetime
    holding_minutes: int


@dataclass(frozen=True, slots=True)
class LabelRow:
    prediction_timestamp: datetime
    available_at: datetime
    entry_timestamp: datetime
    maturity_timestamp: datetime
    forward: dict[int, Outcome]
    barriers: dict[tuple[Decimal, Decimal, int], Outcome]


def _outcome(
    entry: Decimal,
    exit_: Decimal,
    end: datetime,
    holding: int,
    barrier: str,
    atr: Decimal | None,
    volatility: Decimal | None,
    threshold: Decimal,
) -> Outcome:
    result = exit_ / entry - 1
    direction = "up" if result > threshold else "down" if result < -threshold else "neutral"
    return Outcome(
        end,
        end + timedelta(minutes=1),
        result,
        result / (volatility * Decimal(str(sqrt(holding))))
        if volatility is not None and volatility > 0
        else None,
        (exit_ - entry) / atr if atr is not None and atr > 0 else None,
        direction,
        barrier,
        end,
        holding,
    )


def generate(
    features: tuple[FeatureRow, ...], bars: tuple[Candle, ...], config: LabelConfig
) -> tuple[LabelRow, ...]:
    """Use only completed feature bars before entry; future bars are targets, never predictors.

    OHLC cannot resolve same-minute target/stop ordering: stop takes precedence;
    open gaps beyond a stop use the actual adverse open. Never cross sessions,
    missing minutes, or month boundaries. Only complete maximum horizons emit rows.
    """
    validate_bars(bars)
    if not bars or not features:
        return ()
    if any(a.timestamp >= b.timestamp for a, b in pairwise(features)):
        raise ValueError("features must be strictly ordered")
    calendar = xnys()
    tz = ZoneInfo("America/New_York")
    indices = {bar.timestamp.astimezone(UTC): i for i, bar in enumerate(bars)}
    if len(indices) != len(bars):
        raise ValueError("duplicate candle timestamps")
    sessions: dict[date, set[datetime]] = {}
    maximum = max(config.horizons)
    rows: list[LabelRow] = []
    for feature in features:
        now = feature.available_at.astimezone(UTC)
        if as_of((feature,), now) != feature:
            raise ValueError("feature not available at prediction time")
        source = indices.get(feature.timestamp.astimezone(UTC))
        if source is None or now != feature.timestamp + timedelta(minutes=1):
            raise ValueError("feature available_at or raw source mismatch")
        if source + maximum >= len(bars):
            continue
        session = now.astimezone(tz).date()
        if session not in sessions:
            sessions[session] = (
                {minute.to_pydatetime() for minute in calendar.session_minutes(session)}
                if calendar.is_session(session)
                else set()
            )
        expected = sessions[session]
        if any(
            bars[source + i].timestamp != now + timedelta(minutes=i - 1)
            or bars[source + i].timestamp not in expected
            for i in range(1, maximum + 1)
        ):
            continue
        entry = bars[source + 1].open
        atr = feature.values.get("atr")
        volatility = feature.values.get("realized_volatility")
        forward: dict[int, Outcome] = {}
        barriers: dict[tuple[Decimal, Decimal, int], Outcome] = {}
        for horizon in config.horizons:
            final = bars[source + horizon]
            forward[horizon] = _outcome(
                entry,
                final.close,
                final.timestamp,
                horizon,
                "time",
                atr,
                volatility,
                config.direction_threshold,
            )
        for target in config.targets:
            for stop in config.stops:
                stop_price = entry * (1 - stop)
                target_price = entry * (1 + target)
                hit: Outcome | None = None
                for elapsed in range(1, maximum + 1):
                    candle = bars[source + elapsed]
                    if elapsed > 1 and candle.open <= stop_price:
                        price, barrier = candle.open, "stop"
                    elif elapsed > 1 and candle.open >= target_price:
                        price, barrier = candle.open, "target"
                    elif candle.low <= stop_price:
                        price, barrier = stop_price, "stop"
                    elif candle.high >= target_price:
                        price, barrier = target_price, "target"
                    else:
                        continue
                    hit = _outcome(
                        entry,
                        price,
                        candle.timestamp,
                        elapsed,
                        barrier,
                        atr,
                        volatility,
                        config.direction_threshold,
                    )
                    break
                for horizon in config.horizons:
                    barriers[(target, stop, horizon)] = (
                        hit
                        if hit is not None and hit.holding_minutes <= horizon
                        else forward[horizon]
                    )
        maturity = max(
            outcome.maturity_timestamp for outcome in (*forward.values(), *barriers.values())
        )
        rows.append(LabelRow(now, now, now, maturity, forward, barriers))
    return tuple(rows)
