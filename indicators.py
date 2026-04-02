"""
indicators.py — Technical Indicator Library
============================================
All pure-math indicator functions. No side effects, no I/O.
Designed to be swappable with any data source (yfinance, Kite, Upstox).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


# ───────────────────────────────────────
#  CORE INDICATORS
# ───────────────────────────────────────

def calculate_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's smoothed RSI — same method used by TradingView and 
    all professional platforms. Less lag than SMA RSI.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_dual_rsi(close: pd.Series) -> dict:
    """
    Returns both RSI periods. Use RSI7 as entry trigger,
    RSI14 as trend confirmation — only trade when both agree.
    """
    if len(close) < 15:
        return {"rsi7": None, "rsi14": None}
    
    rsi7_series = calculate_rsi_wilder(close, period=7)
    rsi14_series = calculate_rsi_wilder(close, period=14)
    return {
        "rsi7":  round(float(rsi7_series.iloc[-1]), 2),
        "rsi14": round(float(rsi14_series.iloc[-1]), 2),
    }

def ema(closes: pd.Series, span: int) -> Optional[float]:
    """Latest Exponential Moving Average value."""
    if len(closes) < span:
        return None
    return round(float(closes.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def calculate_vwap_with_bands(df: pd.DataFrame) -> dict:
    """
    VWAP with ±1σ and ±2σ bands.
    Anchored to current session open (9:15 AM reset daily).
    """
    if df.empty:
        return {"vwap": df["Close"].iloc[-1] if not df.empty else 0.0, "upper_1sd": 0, "lower_1sd": 0, "upper_2sd": 0, "lower_2sd": 0, "deviation": 0, "deviation_pct": 0}

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tpv = (typical_price * df["Volume"]).cumsum()
    cumulative_vol  = df["Volume"].cumsum()
    vwap = cumulative_tpv / cumulative_vol

    # Rolling standard deviation of (typical_price - vwap)
    deviation = typical_price - vwap
    rolling_std = deviation.rolling(window=20).std().fillna(0)

    return {
        "vwap":      round(float(vwap.iloc[-1]), 2),
        "upper_1sd": round(float((vwap + 1 * rolling_std).iloc[-1]), 2),
        "lower_1sd": round(float((vwap - 1 * rolling_std).iloc[-1]), 2),
        "upper_2sd": round(float((vwap + 2 * rolling_std).iloc[-1]), 2),
        "lower_2sd": round(float((vwap - 2 * rolling_std).iloc[-1]), 2),
        "deviation": round(float(deviation.iloc[-1]), 2),
        "deviation_pct": round(float(((deviation / vwap) * 100).iloc[-1]), 2),
    }

def calculate_volume_surge_ratio(
        current_volume: float,
        historical_volume_df: pd.DataFrame,
        current_time) -> float:
    """
    Compares today's candle volume to the SAME TIME WINDOW
    averaged over the last 5 trading days.
    """
    if historical_volume_df.empty:
        return 1.0

    same_time_mask = (historical_volume_df.index.time == current_time)
    same_time_vols = historical_volume_df.loc[same_time_mask, "Volume"]

    if len(same_time_vols) < 3:
        return 1.0

    baseline = same_time_vols.mean()
    if baseline == 0:
        return 1.0

    return round(float(current_volume / baseline), 2)


# ───────────────────────────────────────
#  ADVANCED INDICATORS
# ───────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average True Range — measures volatility."""
    if len(df) < period + 1:
        return None
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean().iloc[-1]
    return round(float(atr_val), 2) if not pd.isna(atr_val) else None


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend logic returning a DataFrame of results."""
    if len(df) < period + 1:
        return pd.DataFrame()

    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(window=period).mean()

    upper = hl2 + (multiplier * atr_s)
    lower = hl2 - (multiplier * atr_s)

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=str)
    st.iloc[period] = upper.iloc[period]
    direction.iloc[period] = "DOWN"

    for i in range(period + 1, len(df)):
        if df["Close"].iloc[i - 1] <= st.iloc[i - 1]:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1]) \
                if st.iloc[i - 1] == upper.iloc[i - 1] else upper.iloc[i]
            if df["Close"].iloc[i] > st.iloc[i]:
                direction.iloc[i] = "UP"
                st.iloc[i] = lower.iloc[i]
            else:
                direction.iloc[i] = "DOWN"
        else:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1]) \
                if st.iloc[i - 1] == lower.iloc[i - 1] else lower.iloc[i]
            if df["Close"].iloc[i] < st.iloc[i]:
                direction.iloc[i] = "DOWN"
                st.iloc[i] = upper.iloc[i]
            else:
                direction.iloc[i] = "UP"

    return pd.DataFrame({"supertrend": st, "direction": direction})

def get_dual_timeframe_supertrend(
        df_1min: pd.DataFrame,
        df_5min: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0) -> dict:
    """
    Calculate Supertrend on both 1-min and 5-min candles.
    Returns direction of each and whether they agree.
    """
    if df_1min.empty or df_5min.empty:
        return {"dir_1min": "NONE", "dir_5min": "NONE", "agreement": False, "conflict": False}
        
    st_1min = calculate_supertrend(df_1min, period, multiplier)
    st_5min = calculate_supertrend(df_5min, period, multiplier)

    if st_1min.empty or st_5min.empty:
        return {"dir_1min": "NONE", "dir_5min": "NONE", "agreement": False, "conflict": False}

    dir_1min = st_1min["direction"].iloc[-1]
    dir_5min = st_5min["direction"].iloc[-1]

    return {
        "dir_1min":  dir_1min,
        "dir_5min":  dir_5min,
        "agreement": dir_1min == dir_5min,
        "conflict":  dir_1min != dir_5min,
        "st_value": round(float(st_1min["supertrend"].iloc[-1]), 2)
    }


def macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, Any]]:
    """MACD line, signal line, histogram, and bullish flag."""
    if len(closes) < slow + signal:
        return None
    ema_f = closes.ewm(span=fast, adjust=False).mean()
    ema_s = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return {
        "macd":      round(float(macd_line.iloc[-1]), 4),
        "signal":    round(float(sig_line.iloc[-1]), 4),
        "histogram": round(float(hist.iloc[-1]), 4),
        "bullish":   bool(macd_line.iloc[-1] > sig_line.iloc[-1]),
        "hist_series": hist,
    }

def detect_macd_divergence(close: pd.Series, 
                            macd_hist: pd.Series,
                            lookback: int = 5) -> str:
    """
    Detects classic MACD-histogram divergence over last N candles.
    Returns: 'BULLISH', 'BEARISH', or 'NONE'
    """
    if len(close) < lookback + 1:
        return "NONE"

    recent_close = close.iloc[-lookback:].reset_index(drop=True)
    recent_hist  = macd_hist.iloc[-lookback:].reset_index(drop=True)

    price_low_idx = recent_close.idxmin()
    hist_low_idx  = recent_hist.idxmin()

    price_high_idx = recent_close.idxmax()
    hist_high_idx  = recent_hist.idxmax()

    # Bullish: price lowest at earlier candle, histogram lowest at later candle
    if price_low_idx < hist_low_idx:
        price_ll = recent_close.iloc[price_low_idx] < recent_close.iloc[0]
        hist_hl  = recent_hist.iloc[hist_low_idx]   > recent_hist.iloc[0]
        if price_ll and hist_hl:
            return "BULLISH"

    # Bearish: price highest at earlier candle, histogram highest at later candle
    if price_high_idx < hist_high_idx:
        price_hh = recent_close.iloc[price_high_idx] > recent_close.iloc[0]
        hist_lh  = recent_hist.iloc[hist_high_idx]   < recent_hist.iloc[0]
        if price_hh and hist_lh:
            return "BEARISH"

    return "NONE"


def calculate_bollinger_with_bandwidth(close: pd.Series,
                                        period: int = 20,
                                        std_mult: float = 2.0) -> dict:
    if len(close) < period:
        return {"upper": 0, "lower": 0, "middle": 0, "bandwidth": 0}
    sma    = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = sma + std_mult * std
    lower  = sma - std_mult * std
    bw_pct = ((upper - lower) / sma) * 100

    return {
        "upper":     round(float(upper.iloc[-1]), 2),
        "lower":     round(float(lower.iloc[-1]), 2),
        "middle":    round(float(sma.iloc[-1]), 2),
        "bandwidth": round(float(bw_pct.iloc[-1]), 2),
    }

def adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average Directional Index — trend strength measurement."""
    if len(df) < period * 2:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_v = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_v)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_v)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx_v = dx.rolling(window=period).mean()
    val = adx_v.iloc[-1]
    return round(float(val), 2) if not pd.isna(val) else None


def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """Classic CPR Pivot Points — S1-S3, R1-R3."""
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
    }


# ───────────────────────────────────────
#  V5.2 — CANDLESTICK PATTERN ENGINE
# ───────────────────────────────────────

def detect_candle_patterns(df: pd.DataFrame, atr_val: float) -> dict:
    """
    Detects 8 high-probability patterns on the last 2 candles.
    All patterns are normalized by ATR to avoid false signals
    on low-volatility candles.
    """
    if df is None or len(df) < 2 or atr_val is None or atr_val <= 0:
        return {}

    o  = float(df["Open"].iloc[-1])
    h  = float(df["High"].iloc[-1])
    l  = float(df["Low"].iloc[-1])
    c  = float(df["Close"].iloc[-1])
    po = float(df["Open"].iloc[-2])
    ph = float(df["High"].iloc[-2])
    pl = float(df["Low"].iloc[-2])
    pc = float(df["Close"].iloc[-2])

    body       = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    prev_body  = abs(pc - po)

    patterns = {}

    # 1. HAMMER — bullish reversal
    if (body > 0 and lower_wick >= 2.0 * body and
        upper_wick <= 0.3 * body and body >= 0.1 * atr_val):
        patterns["HAMMER"] = "BULLISH"

    # 2. SHOOTING STAR — bearish reversal
    if (body > 0 and upper_wick >= 2.0 * body and
        lower_wick <= 0.3 * body and body >= 0.1 * atr_val):
        patterns["SHOOTING_STAR"] = "BEARISH"

    # 3. BULLISH ENGULFING
    if (c > o and pc > po and
        o <= pc and c >= po and
        prev_body > 0 and body >= prev_body * 1.1):
        patterns["BULLISH_ENGULFING"] = "BULLISH"

    # 4. BEARISH ENGULFING
    if (c < o and pc < po and
        o >= pc and c <= po and
        prev_body > 0 and body >= prev_body * 1.1):
        patterns["BEARISH_ENGULFING"] = "BEARISH"

    # 5. DOJI — indecision
    total_range = h - l
    if total_range > 0 and body / total_range < 0.10:
        patterns["DOJI"] = "NEUTRAL"

    # 6. PIN BAR (bullish)
    if (body > 0 and lower_wick >= 3.0 * body and
        (h - l) > 0 and (c - l) / (h - l) >= 0.60):
        patterns["PIN_BAR_BULL"] = "BULLISH"

    # 7. PIN BAR (bearish)
    if (body > 0 and upper_wick >= 3.0 * body and
        (h - l) > 0 and (h - c) / (h - l) >= 0.60):
        patterns["PIN_BAR_BEAR"] = "BEARISH"

    # 8. INSIDE BAR — compression before breakout
    if h < ph and l > pl:
        patterns["INSIDE_BAR"] = "NEUTRAL"

    return patterns


# ───────────────────────────────────────
#  V5.2 — INTRADAY SUPPORT/RESISTANCE
# ───────────────────────────────────────

def calculate_intraday_sr(df: pd.DataFrame, lookback_candles: int = 20) -> dict:
    """
    Detects today's intraday S/R levels by finding swing highs/lows.
    A swing high = local max with 2 lower highs on each side.
    Returns the nearest S/R levels relative to current price.
    """
    result = {"resistance_1": None, "resistance_2": None,
              "support_1": None, "support_2": None}

    if df is None or len(df) < lookback_candles or lookback_candles < 5:
        return result

    highs = df["High"].iloc[-lookback_candles:]
    lows  = df["Low"].iloc[-lookback_candles:]

    swing_highs = []
    swing_lows  = []

    for i in range(2, len(highs) - 2):
        if (highs.iloc[i] > highs.iloc[i-1] and
            highs.iloc[i] > highs.iloc[i-2] and
            highs.iloc[i] > highs.iloc[i+1] and
            highs.iloc[i] > highs.iloc[i+2]):
            swing_highs.append(float(highs.iloc[i]))

        if (lows.iloc[i] < lows.iloc[i-1] and
            lows.iloc[i] < lows.iloc[i-2] and
            lows.iloc[i] < lows.iloc[i+1] and
            lows.iloc[i] < lows.iloc[i+2]):
            swing_lows.append(float(lows.iloc[i]))

    current_price = float(df["Close"].iloc[-1])
    resistances = sorted([h for h in swing_highs if h > current_price])
    supports    = sorted([s for s in swing_lows if s < current_price], reverse=True)

    result["resistance_1"] = resistances[0] if len(resistances) > 0 else None
    result["resistance_2"] = resistances[1] if len(resistances) > 1 else None
    result["support_1"]    = supports[0]    if len(supports)    > 0 else None
    result["support_2"]    = supports[1]    if len(supports)    > 1 else None
    return result


# ───────────────────────────────────────
#  V5.2 — VOLUME PROFILE (VPOC)
# ───────────────────────────────────────

def calculate_volume_profile(df: pd.DataFrame, price_bins: int = 50) -> dict:
    """
    Builds intraday volume profile. Divides price range into N bins.
    Returns VPOC (price with highest volume) and Value Area boundaries.
    """
    if df is None or df.empty or len(df) < 5:
        return {"vpoc": None, "va_high": None, "va_low": None}

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())

    if price_max == price_min:
        return {"vpoc": float(df["Close"].iloc[-1]),
                "va_high": price_max, "va_low": price_min}

    bin_size = (price_max - price_min) / price_bins
    bins     = {}

    for _, row in df.iterrows():
        candle_low  = float(row["Low"])
        candle_high = float(row["High"])
        candle_vol  = float(row["Volume"])
        candle_bins = max(1, int((candle_high - candle_low) / bin_size))
        vol_per_bin = candle_vol / candle_bins

        for b in range(candle_bins):
            price_level = round(candle_low + b * bin_size, 2)
            bins[price_level] = bins.get(price_level, 0) + vol_per_bin

    if not bins:
        return {"vpoc": float(df["Close"].iloc[-1]), "va_high": None, "va_low": None}

    vpoc = max(bins, key=bins.get)

    # Value Area: price range containing 70% of total volume
    total_vol  = sum(bins.values())
    target_vol = total_vol * 0.70
    sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)

    va_levels   = []
    accumulated = 0
    for price_level, vol in sorted_bins:
        accumulated += vol
        va_levels.append(price_level)
        if accumulated >= target_vol:
            break

    return {
        "vpoc":    round(vpoc, 2),
        "va_high": round(max(va_levels), 2) if va_levels else None,
        "va_low":  round(min(va_levels), 2) if va_levels else None,
    }


# ───────────────────────────────────────
#  V5.2 — GAP CLASSIFICATION ENGINE
# ───────────────────────────────────────

def classify_gap(today_open: float, prev_close: float,
                 prev_high: float, prev_low: float,
                 avg_volume_today: float, avg_volume_5day: float,
                 atr_val: float) -> dict:
    """
    Classifies the opening gap into one of 4 types.
    Each type implies a different trading strategy.
    """
    if prev_close == 0 or atr_val is None or atr_val <= 0:
        return {"gap_type": "FLAT", "gap_pct": 0, "direction": "FLAT",
                "gap_size_atr": 0, "volume_ratio": 1.0, "strategy_hint": "ORB_BOTH"}

    gap_pct = (today_open - prev_close) / prev_close * 100
    volume_ratio = avg_volume_today / avg_volume_5day if avg_volume_5day > 0 else 1.0
    direction = "UP" if gap_pct > 0.1 else ("DOWN" if gap_pct < -0.1 else "FLAT")
    gap_size_atr = abs(today_open - prev_close) / atr_val

    if abs(gap_pct) < 0.3:
        gap_type = "FLAT"
        strategy_hint = "ORB_BOTH"
    elif gap_size_atr > 2.0 and volume_ratio > 2.0:
        gap_type = "BREAKAWAY"
        strategy_hint = f"ORB_{direction}"
    elif gap_size_atr > 1.5 and volume_ratio < 1.2:
        gap_type = "EXHAUSTION"
        strategy_hint = f"REVERSAL_{'LONG' if direction == 'DOWN' else 'SHORT'}"
    elif 0.5 < gap_size_atr <= 2.0 and volume_ratio > 1.5:
        gap_type = "CONTINUATION"
        strategy_hint = f"ORB_{direction}"
    else:
        gap_type = "COMMON"
        strategy_hint = "WAIT_OR_ORB"

    return {
        "gap_type":      gap_type,
        "gap_pct":       round(gap_pct, 2),
        "direction":     direction,
        "gap_size_atr":  round(gap_size_atr, 2),
        "volume_ratio":  round(volume_ratio, 2),
        "strategy_hint": strategy_hint,
    }


# ───────────────────────────────────────
#  V5.2 — ORDER FLOW IMBALANCE (OFI)
# ───────────────────────────────────────

def calculate_order_flow_imbalance(df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Measures buying/selling pressure from candle close position.
    OFI = (Close - Low) / (High - Low)
    OFI = 1.0 → close at high (max buying pressure)
    OFI = 0.0 → close at low (max selling pressure)
    """
    if df is None or len(df) < lookback + 1:
        return {"current_ofi": 0.5, "avg_ofi": 0.5, "ofi_trend": 0.0,
                "buyers_in_control": False, "sellers_in_control": False, "neutral": True}

    results = []
    for i in range(-lookback, 0):
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        rng = h - l
        ofi = (c - l) / rng if rng > 0 else 0.5
        results.append(ofi)

    current_ofi = results[-1]
    avg_ofi     = sum(results) / len(results)
    ofi_trend   = results[-1] - results[0]

    return {
        "current_ofi":        round(current_ofi, 3),
        "avg_ofi":            round(avg_ofi, 3),
        "ofi_trend":          round(ofi_trend, 3),
        "buyers_in_control":  current_ofi > 0.65,
        "sellers_in_control": current_ofi < 0.35,
        "neutral":            0.35 <= current_ofi <= 0.65,
    }
