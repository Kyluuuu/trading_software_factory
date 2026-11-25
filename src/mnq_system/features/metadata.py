"""Versioned feature specification; windows count consecutive observed minute bars."""

from dataclasses import dataclass

VERSION = 1


@dataclass(frozen=True, slots=True)
class Definition:
    name: str
    description: str
    formula: str
    dependencies: tuple[str, ...]
    lookback: int


def _d(name: str, description: str, formula: str, dependencies: str, lookback: int) -> Definition:
    return Definition(name, description, formula, tuple(dependencies.split()), lookback)


DEFINITIONS: tuple[Definition, ...] = (
    _d("open", "Minute opening price", "open_t", "open", 1),
    _d("high", "Minute maximum price", "high_t", "high", 1),
    _d("low", "Minute minimum price", "low_t", "low", 1),
    _d("close", "Minute closing price", "close_t", "close", 1),
    _d("volume", "Minute share volume", "volume_t", "volume", 1),
    _d("returns", "One-minute simple close return", "close_t/close_(t-1)-1", "close", 2),
    _d("log_returns", "One-minute log close return", "ln(close_t/close_(t-1))", "close", 2),
    _d(
        "atr",
        "14-bar Wilder average true range",
        "RMA14(max(high-low,abs(high-prev_close),abs(low-prev_close)))",
        "high low close",
        15,
    ),
    _d(
        "realized_volatility",
        "20-return realized volatility (unannualized)",
        "sqrt(sum(log_returns^2,20))",
        "close",
        21,
    ),
    _d("rolling_std", "20-close population standard deviation", "std_pop(close,20)", "close", 20),
    _d(
        "volatility_percentile",
        "Percent rank of current range in last 60 ranges",
        "count(range_i<=range_t,60)/60",
        "high low",
        60,
    ),
    _d("rsi", "14-change Wilder RSI; flat=50", "100-100/(1+RMA14(gain)/RMA14(loss))", "close", 15),
    _d("bollinger_upper", "Upper Bollinger band", "SMA20(close)+2*std_pop20(close)", "close", 20),
    _d("bollinger_lower", "Lower Bollinger band", "SMA20(close)-2*std_pop20(close)", "close", 20),
    _d(
        "bollinger_distance",
        "Normalized distance to SMA20 in band widths",
        "(close-SMA20)/(2*std_pop20); 0 if flat",
        "close",
        20,
    ),
    _d(
        "z_score",
        "20-close standardized distance; 0 if flat",
        "(close-SMA20)/std_pop20",
        "close",
        20,
    ),
    _d("distance_from_mean", "Relative SMA20 distance", "close/SMA20-1", "close", 20),
    _d("roc", "10-bar close rate of change", "close_t/close_(t-10)-1", "close", 11),
    _d("momentum_5", "Five-bar close return", "close_t/close_(t-5)-1", "close", 6),
    _d("momentum_20", "Twenty-bar close return", "close_t/close_(t-20)-1", "close", 21),
    _d(
        "trend_acceleration",
        "Change in five-bar momentum",
        "momentum5_t-momentum5_(t-5)",
        "close",
        11,
    ),
    _d("ema_distance", "Relative distance to EMA20", "close/EMA20-1", "close", 20),
    _d("ema_slope", "Relative one-bar EMA20 slope", "EMA20_t/EMA20_(t-1)-1", "close", 21),
    _d(
        "adx",
        "14-bar Wilder ADX",
        "RMA14(100*abs(+DI14--DI14)/(+DI14+-DI14))",
        "high low close",
        29,
    ),
    _d("trend_strength", "Absolute EMA20 distance", "abs(close/EMA20-1)", "close", 20),
    _d(
        "session_vwap",
        "Session cumulative typical-price VWAP",
        "sum(((high+low+close)/3)*volume)/sum(volume)",
        "high low close volume",
        1,
    ),
    _d(
        "vwap_distance",
        "Relative session VWAP distance",
        "close/session_vwap-1",
        "close session_vwap",
        1,
    ),
    _d(
        "vwap_deviation",
        "Session volume-weighted typical-price deviation",
        "(close-session_vwap)/sqrt(sum(volume*(typical-session_vwap)^2)/sum(volume))",
        "high low close volume",
        1,
    ),
    _d(
        "relative_volume",
        "Current vs prior 20-minute volume",
        "volume_t/mean(volume_(t-20:t-1))",
        "volume",
        21,
    ),
    _d(
        "volume_percentile",
        "Current volume percentile of last 20",
        "count(volume_i<=volume_t,20)/20",
        "volume",
        20,
    ),
    _d(
        "volume_spike_score",
        "Current volume versus prior mean in std units",
        "(volume_t-mean(prior20))/std_pop(prior20)",
        "volume",
        21,
    ),
    _d(
        "donchian_high",
        "Prior 20-bar high, excluding current bar",
        "max(high_(t-20:t-1))",
        "high",
        21,
    ),
    _d("donchian_low", "Prior 20-bar low, excluding current bar", "min(low_(t-20:t-1))", "low", 21),
    _d(
        "compression",
        "Recent five ranges versus previous 20",
        "mean(range_(t-4:t))/mean(range_(t-24:t-5))",
        "high low",
        25,
    ),
    _d(
        "breakout_distance",
        "Signed distance past prior Donchian channel; 0 inside",
        "max(close/donchian_high-1,0)+min(close/donchian_low-1,0)",
        "close high low",
        21,
    ),
    _d(
        "higher_highs",
        "Count of rising consecutive highs in last five comparisons",
        "sum(high_i>high_(i-1),5)",
        "high",
        6,
    ),
    _d(
        "lower_lows",
        "Count of falling consecutive lows in last five comparisons",
        "sum(low_i<low_(i-1),5)",
        "low",
        6,
    ),
    _d(
        "swing_structure",
        "Confirmed two-neighbor swing direction",
        "+1 if high_(t-2)>neighbors else -1 if low_(t-2)<neighbors else 0",
        "high low",
        5,
    ),
    _d(
        "support_distance",
        "Relative distance to prior 20-bar support",
        "close/min(low_(t-20:t-1))-1",
        "close low",
        21,
    ),
    _d(
        "resistance_distance",
        "Relative distance to prior 20-bar resistance",
        "close/max(high_(t-20:t-1))-1",
        "close high",
        21,
    ),
    _d(
        "minute_of_day",
        "XNYS local minute index since midnight",
        "hour_NY*60+minute_NY",
        "timestamp",
        1,
    ),
    _d("hour", "XNYS local hour", "hour_NY", "timestamp", 1),
    _d("day_of_week", "XNYS local weekday (Monday=0)", "weekday_NY", "timestamp", 1),
    _d(
        "session_progress",
        "Fraction of XNYS session elapsed at bar close",
        "(minute_index+1)/session_minutes",
        "timestamp calendar",
        1,
    ),
)
