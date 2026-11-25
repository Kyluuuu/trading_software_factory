"""Causal regular-session minute feature matrix; unavailable values are null."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from math import isfinite, log, sqrt
from zoneinfo import ZoneInfo

from mnq_system.backtest.engine import Candle
from mnq_system.data.calendar import xnys
from mnq_system.data.validation import validate_bars
from mnq_system.features.metadata import DEFINITIONS

_NAMES = frozenset(d.name for d in DEFINITIONS)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    timestamp: datetime
    available_at: datetime
    values: dict[str, Decimal | None]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    average = _mean(values)
    return sqrt(max(0.0, sum((x - average) ** 2 for x in values) / len(values)))


def _wilder(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    average = _mean(values[:period])
    for value in values[period:]:
        average = (average * (period - 1) + value) / period
    return average


def _number(value: float | int | None) -> Decimal | None:
    if value is None or not isfinite(value):
        return None
    return Decimal(str(value))


def _features(
    history: list[Candle],
    ema: float | None,
    vwap_sums: tuple[float, float, float],
    true_ranges: list[float],
    dx_values: list[float],
) -> dict[str, Decimal | None]:
    current = history[-1]
    n = len(history)
    closes = [float(c.close) for c in history]
    highs = [float(c.high) for c in history]
    lows = [float(c.low) for c in history]
    volumes = [float(c.volume) for c in history]
    close, high, low, volume = closes[-1], highs[-1], lows[-1], volumes[-1]
    values: dict[str, float | int | None] = dict.fromkeys(_NAMES)
    values.update(open=float(current.open), high=high, low=low, close=close, volume=volume)
    if n >= 2:
        values["returns"] = close / closes[-2] - 1
        values["log_returns"] = log(close / closes[-2])
    if len(true_ranges) >= 14:
        values["atr"] = _wilder(true_ranges, 14)
    if n >= 21:
        values["realized_volatility"] = sqrt(
            sum(log(closes[i] / closes[i - 1]) ** 2 for i in range(n - 20, n))
        )
    if n >= 20:
        window = closes[-20:]
        mean, std = _mean(window), _std(window)
        values.update(
            rolling_std=std,
            bollinger_upper=mean + 2 * std,
            bollinger_lower=mean - 2 * std,
            bollinger_distance=(close - mean) / (2 * std) if std else 0,
            z_score=(close - mean) / std if std else 0,
            distance_from_mean=close / mean - 1,
            volume_percentile=sum(x <= volume for x in volumes[-20:]) / 20,
        )
    if n >= 60:
        ranges = [h - bottom for h, bottom in zip(highs[-60:], lows[-60:], strict=True)]
        values["volatility_percentile"] = sum(x <= ranges[-1] for x in ranges) / 60
    if n >= 15:
        changes = [closes[i] - closes[i - 1] for i in range(1, n)]
        gain = _wilder([max(0.0, x) for x in changes], 14)
        loss = _wilder([max(0.0, -x) for x in changes], 14)
        if gain is not None and loss is not None:
            values["rsi"] = 100 * gain / (gain + loss) if gain + loss else 50
    if n >= 11:
        values["roc"] = close / closes[-11] - 1
        values["trend_acceleration"] = close / closes[-6] - closes[-6] / closes[-11]
    if n >= 6:
        values["momentum_5"] = close / closes[-6] - 1
        values["higher_highs"] = sum(highs[i] > highs[i - 1] for i in range(n - 5, n))
        values["lower_lows"] = sum(lows[i] < lows[i - 1] for i in range(n - 5, n))
    if n >= 21:
        values["momentum_20"] = close / closes[-21] - 1
        prior_high, prior_low = max(highs[-21:-1]), min(lows[-21:-1])
        values.update(
            donchian_high=prior_high,
            donchian_low=prior_low,
            breakout_distance=(
                max(close / prior_high - 1, 0)
                if close > prior_high
                else min(close / prior_low - 1, 0)
                if close < prior_low
                else 0
            ),
            support_distance=close / prior_low - 1,
            resistance_distance=close / prior_high - 1,
        )
        previous_volumes = volumes[-21:-1]
        avg, std = _mean(previous_volumes), _std(previous_volumes)
        values["relative_volume"] = volume / avg if avg else None
        values["volume_spike_score"] = (volume - avg) / std if std else 0
    if n >= 25:
        recent = _mean([highs[i] - lows[i] for i in range(n - 5, n)])
        prior = _mean([highs[i] - lows[i] for i in range(n - 25, n - 5)])
        values["compression"] = recent / prior if prior else None
    if n >= 5:
        center = n - 3
        values["swing_structure"] = (
            1
            if highs[center] > max(highs[center - 2 : center] + highs[center + 1 : center + 3])
            else -1
            if lows[center] < min(lows[center - 2 : center] + lows[center + 1 : center + 3])
            else 0
        )
    if n >= 20 and ema is not None:
        values["ema_distance"] = close / ema - 1
        values["trend_strength"] = abs(close / ema - 1)
    if n >= 21 and ema is not None:
        previous_ema = (ema - (2 / 21) * close) / (1 - 2 / 21)
        values["ema_slope"] = ema / previous_ema - 1
    if len(dx_values) >= 14:
        values["adx"] = _wilder(dx_values, 14)
    weighted, weighted_sq, total_volume = vwap_sums
    if total_volume:
        vwap = weighted / total_volume
        deviation = sqrt(max(0.0, weighted_sq / total_volume - vwap * vwap))
        values["session_vwap"] = vwap
        values["vwap_distance"] = close / vwap - 1
        values["vwap_deviation"] = (close - vwap) / deviation if deviation else 0
    local = current.timestamp.astimezone(ZoneInfo("America/New_York"))
    values.update(
        minute_of_day=local.hour * 60 + local.minute, hour=local.hour, day_of_week=local.weekday()
    )
    return {name: _number(value) for name, value in values.items()}


def generate(bars: tuple[Candle, ...]) -> tuple[FeatureRow, ...]:
    """Produce features known at bar close; ignore extended hours, reset on gaps."""
    validate_bars(bars)
    if not bars:
        return ()
    calendar = xnys()
    session_minutes: dict[date, dict[datetime, int]] = {}
    results: list[FeatureRow] = []
    history: list[Candle] = []
    last: datetime | None = None
    session: date | None = None
    ema: float | None = None
    true_ranges: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    dx_values: list[float] = []
    vwap_sums = (0.0, 0.0, 0.0)
    for candle in bars:
        local_date = candle.timestamp.astimezone(ZoneInfo("America/New_York")).date()
        if local_date not in session_minutes:
            session_minutes[local_date] = (
                {
                    minute.to_pydatetime(): i
                    for i, minute in enumerate(calendar.session_minutes(local_date))
                }
                if calendar.is_session(local_date)
                else {}
            )
        minutes = session_minutes[local_date]
        at = candle.timestamp.astimezone(UTC)
        if at not in minutes:
            continue
        if local_date != session or last is None or at - last != timedelta(minutes=1):
            history, true_ranges, plus_dm, minus_dm, dx_values = [], [], [], [], []
            ema = None
            if local_date != session:
                vwap_sums = (0.0, 0.0, 0.0)
        session, last = local_date, at
        if history:
            previous = history[-1]
            tr = max(
                float(candle.high - candle.low),
                abs(float(candle.high - previous.close)),
                abs(float(candle.low - previous.close)),
            )
            up = float(candle.high - previous.high)
            down = float(previous.low - candle.low)
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            true_ranges.append(tr)
            if len(true_ranges) >= 14:
                atr = _wilder(true_ranges, 14)
                plus = _wilder(plus_dm, 14)
                minus = _wilder(minus_dm, 14)
                if atr is not None and plus is not None and minus is not None and atr:
                    total = plus + minus
                    dx_values.append(100 * abs(plus - minus) / total if total else 0)
        history.append(candle)
        close = float(candle.close)
        ema = close if ema is None else ema + (2 / 21) * (close - ema)
        typical = float((candle.high + candle.low + candle.close) / 3)
        volume = candle.volume
        vwap_sums = (
            vwap_sums[0] + typical * volume,
            vwap_sums[1] + typical * typical * volume,
            vwap_sums[2] + volume,
        )
        values = _features(history, ema, vwap_sums, true_ranges, dx_values)
        values["session_progress"] = Decimal(minutes[at] + 1) / Decimal(len(minutes))
        if set(values) != set(_NAMES) or any(
            v is not None and not v.is_finite() for v in values.values()
        ):
            raise ValueError("invalid feature matrix")
        results.append(FeatureRow(at, at + timedelta(minutes=1), values))
    return tuple(results)
