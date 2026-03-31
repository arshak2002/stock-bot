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
