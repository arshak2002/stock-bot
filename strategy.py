import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, time as dtime

# REGIME MAPPING
REGIME_STRATEGY_MAP = {
    "STRONG_TREND_UP":   ["ORB_LONG", "EMA_PULLBACK_LONG"],
    "WEAK_TREND_UP":     ["ORB_LONG", "EMA_PULLBACK_LONG", "REVERSAL_LONG"],
    "RANGE":             ["REVERSAL_LONG", "REVERSAL_SHORT"],
    "WEAK_TREND_DOWN":   ["ORB_SHORT", "EMA_PULLBACK_SHORT", "REVERSAL_SHORT"],
    "STRONG_TREND_DOWN": ["ORB_SHORT", "EMA_PULLBACK_SHORT"],
}

REGIME_SIZE_MULTIPLIER = {
    "STRONG_TREND_UP":   1.0,
    "WEAK_TREND_UP":     0.7,
    "RANGE":             0.5,
    "WEAK_TREND_DOWN":   0.7,
    "STRONG_TREND_DOWN": 1.0,
}

def get_allowed_strategies(regime: str) -> list:
    return REGIME_STRATEGY_MAP.get(regime, [])

def get_position_size_multiplier(regime: str) -> float:
    return REGIME_SIZE_MULTIPLIER.get(regime, 0.5)

def get_min_score(current_time: dtime, cfg: Dict) -> int:
    hour, minute = current_time.hour, current_time.minute
    if hour == 9 or (hour == 10 and minute == 0):
        return cfg["min_score_orb_window"]   
    elif hour < 12:
        return cfg["min_score_morning"]      
    else:
        return cfg["min_score_afternoon"]    


# ═══════════════════════════════════════════════════════════
# POINT SCORERS
# ═══════════════════════════════════════════════════════════

def score_rsi(rsi7: float, rsi14: float, strategy_type: str) -> float:
    """
    Entry trigger = RSI7, confirmation = RSI14.
    Both must agree for full point. If only RSI14 agrees, 0.5 pts.
    """
    if rsi7 is None or rsi14 is None:
        return 0.0

    trigger = confirm = False
    if strategy_type == "ORB_LONG":
        trigger = 45 < rsi7  < 75
        confirm = 45 < rsi14 < 75
    elif strategy_type == "ORB_SHORT":
        trigger = 25 < rsi7  < 55
        confirm = 25 < rsi14 < 55
    elif strategy_type == "REVERSAL_LONG":
        trigger = rsi7  < 38   
        confirm = rsi14 < 42
    elif strategy_type == "REVERSAL_SHORT":
        trigger = rsi7  > 62
        confirm = rsi14 > 58

    if trigger and confirm:
        return 1.0    
    elif confirm:
        return 0.5    
    return 0.0


def score_macd(macd_div: str, strategy_type: str) -> float:
    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        return 0.5 if macd_div == "BULLISH" else 0.0
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        return 0.5 if macd_div == "BEARISH" else 0.0
    return 0.0


def score_bollinger(close: pd.Series, bb_upper: float, bb_lower: float,
                    bb_bandwidth: float, strategy_type: str) -> float:
    """
    For ORB strategies: reward SQUEEZE BREAKOUT (expansion)
    For REVERSAL strategies: reward BAND TOUCH (mean reversion)
    """
    if close.empty or bb_upper is None or bb_lower is None or bb_bandwidth is None:
        return 0.0

    current_price = close.iloc[-1]
    prev_bw       = bb_bandwidth  

    if strategy_type in ("ORB_LONG", "ORB_SHORT"):
        squeeze_breakout = prev_bw > 1.0   
        price_outside    = (current_price > bb_upper) or (current_price < bb_lower)
        return 1.0 if (squeeze_breakout and price_outside) else 0.0
    elif strategy_type == "REVERSAL_LONG":
        return 1.0 if current_price <= bb_lower * 1.002 else 0.0
    elif strategy_type == "REVERSAL_SHORT":
        return 1.0 if current_price >= bb_upper * 0.998 else 0.0
    return 0.0


def score_adx(adx: float, strategy_type: str, cfg: Dict) -> float:
    if adx is None:
        return 0.0
    ind = cfg["indicators"]
    orb_min = ind.get("adx_orb_min", 25)
    orb_strong = ind.get("adx_orb_strong", 30)
    rev_max = ind.get("adx_reversal_max", 18)
    rev_partial = ind.get("adx_reversal_partial", 22)

    if strategy_type in ("ORB_LONG", "ORB_SHORT"):
        if adx >= orb_strong: return 1.0    
        elif adx >= orb_min: return 0.5    
        else: return 0.0    

    elif strategy_type in ("REVERSAL_LONG", "REVERSAL_SHORT"):
        if adx < rev_max: return 1.0    
        elif adx < rev_partial: return 0.5    
        else: return 0.0    
    return 0.0


def score_volume(surge_ratio: float) -> float:
    if surge_ratio >= 2.5: return 2.0    
    elif surge_ratio >= 1.8: return 1.5    
    elif surge_ratio >= 1.2: return 1.0    
    elif surge_ratio >= 0.9: return 0.0    
    else: return -0.5   


def score_supertrend(st_data: dict, strategy_type: str) -> float:
    dir_1 = st_data.get("dir_1min", "NONE")
    dir_5 = st_data.get("dir_5min", "NONE")

    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if dir_1 == "UP" and dir_5 == "UP": return 2.0    
        elif dir_5 == "UP" and dir_1 == "DOWN": return 1.0    
        else: return 0.0
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if dir_1 == "DOWN" and dir_5 == "DOWN": return 2.0
        elif dir_5 == "DOWN" and dir_1 == "UP": return 1.0
        else: return 0.0
    return 0.0


def score_vwap_bands(price: float, vwap_data: dict, pivot_levels: dict, strategy_type: str, cfg: Dict) -> float:
    vwap      = vwap_data.get("vwap")
    upper_1sd = vwap_data.get("upper_1sd")
    lower_1sd = vwap_data.get("lower_1sd")
    upper_2sd = vwap_data.get("upper_2sd")
    lower_2sd = vwap_data.get("lower_2sd")
    
    if vwap is None or upper_1sd is None:
        return 0.0

    cpr_band  = cfg["scoring"]["cpr_proximity_pct"] / 100

    near_pivot = False
    if pivot_levels:
        near_pivot = any(abs(price - lvl) / price < cpr_band for lvl in pivot_levels.values())

    if strategy_type == "ORB_LONG":
        if lower_1sd <= price <= upper_1sd and price > vwap: return 1.0   
        elif price > upper_1sd: return 0.5   
        elif near_pivot: return 0.5
    elif strategy_type == "ORB_SHORT":
        if lower_1sd <= price <= upper_1sd and price < vwap: return 1.0
        elif price < lower_1sd: return 0.5
        elif near_pivot: return 0.5
    elif strategy_type == "REVERSAL_LONG":
        if price <= lower_2sd: return 1.0   
        elif price <= lower_1sd: return 0.5   
        elif near_pivot: return 0.5
    elif strategy_type == "REVERSAL_SHORT":
        if price >= upper_2sd: return 1.0
        elif price >= upper_1sd: return 0.5
        elif near_pivot: return 0.5

    return 0.0


def score_ema_cross(ema9: pd.Series, ema21: pd.Series, strategy_type: str, lookback: int = 3) -> float:
    if len(ema9) < lookback + 1 or len(ema21) < lookback + 1:
        return 0.0
        
    recent_9  = ema9.iloc[-lookback-1:].reset_index(drop=True)
    recent_21 = ema21.iloc[-lookback-1:].reset_index(drop=True)

    cross_up   = any(recent_9.iloc[i-1] <= recent_21.iloc[i-1] and recent_9.iloc[i] > recent_21.iloc[i] for i in range(1, len(recent_9)))
    cross_down = any(recent_9.iloc[i-1] >= recent_21.iloc[i-1] and recent_9.iloc[i] < recent_21.iloc[i] for i in range(1, len(recent_9)))

    currently_above = ema9.iloc[-1] > ema21.iloc[-1]
    currently_below = ema9.iloc[-1] < ema21.iloc[-1]

    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if cross_up: return 1.0              
        elif currently_above: return 0.5              
        else: return 0.0
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if cross_down: return 1.0
        elif currently_below: return 0.5
        else: return 0.0

    return 0.0


def pcr_direction_ok(pcr: float, strategy_type: str) -> bool:
    """
    Blocks trades going AGAINST strong PCR signal.
    """
    if pcr is None: return True
    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if pcr < 0.55: return False    # only block on extreme bearish options positioning
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if pcr > 1.50: return False    # only block on extreme bullish options positioning
    return True


# ═══════════════════════════════════════════════════════════
# V5.2 NEW SCORERS
# ═══════════════════════════════════════════════════════════

def score_candle_pattern(patterns: dict, strategy_type: str) -> float:
    """Adds up to +1.0 based on candlestick pattern alignment."""
    if not patterns:
        return 0.0

    bullish_strong = {"BULLISH_ENGULFING", "PIN_BAR_BULL"}
    bullish_medium = {"HAMMER"}
    bearish_strong = {"BEARISH_ENGULFING", "PIN_BAR_BEAR"}
    bearish_medium = {"SHOOTING_STAR"}
    neutral        = {"DOJI", "INSIDE_BAR"}

    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if any(p in bullish_strong for p in patterns):
            return 1.0
        elif any(p in bullish_medium for p in patterns):
            return 0.75
        elif any(p in neutral for p in patterns):
            return -0.5
        elif any(p in bearish_strong for p in patterns):
            return -1.0
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if any(p in bearish_strong for p in patterns):
            return 1.0
        elif any(p in bearish_medium for p in patterns):
            return 0.75
        elif any(p in neutral for p in patterns):
            return -0.5
        elif any(p in bullish_strong for p in patterns):
            return -1.0
    return 0.0


def score_intraday_sr(price: float, sr_levels: dict, strategy_type: str) -> float:
    """Scores +1.0 if price is at a key intraday S/R level."""
    if not sr_levels:
        return 0.0
    tolerance = 0.003

    if strategy_type in ("ORB_LONG",):
        r1 = sr_levels.get("resistance_1")
        if r1 and abs(price - r1) / r1 < tolerance:
            return 1.0
        s1 = sr_levels.get("support_1")
        if s1 and abs(price - s1) / s1 < tolerance:
            return 0.75
    elif strategy_type in ("ORB_SHORT",):
        s1 = sr_levels.get("support_1")
        if s1 and abs(price - s1) / s1 < tolerance:
            return 1.0
        r1 = sr_levels.get("resistance_1")
        if r1 and abs(price - r1) / r1 < tolerance:
            return 0.75
    elif strategy_type == "REVERSAL_LONG":
        s1 = sr_levels.get("support_1")
        s2 = sr_levels.get("support_2")
        if s1 and abs(price - s1) / s1 < tolerance:
            return 1.0
        if s2 and abs(price - s2) / s2 < tolerance:
            return 0.75
    elif strategy_type == "REVERSAL_SHORT":
        r1 = sr_levels.get("resistance_1")
        if r1 and abs(price - r1) / r1 < tolerance:
            return 1.0
    return 0.0


def score_vpoc(price: float, vp: dict, strategy_type: str) -> float:
    """Scores based on price position relative to VPOC and Value Area."""
    if not vp:
        return 0.0
    vpoc    = vp.get("vpoc")
    va_high = vp.get("va_high")
    va_low  = vp.get("va_low")
    if not vpoc:
        return 0.0
    tolerance = 0.003

    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if abs(price - vpoc) / vpoc < tolerance:
            return 0.75
        if va_low and price < va_low * 0.998:
            return 1.0
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if abs(price - vpoc) / vpoc < tolerance:
            return 0.75
        if va_high and price > va_high * 1.002:
            return 1.0
    return 0.0


def score_ofi(ofi_data: dict, strategy_type: str) -> float:
    """Scores based on order flow imbalance direction."""
    if not ofi_data:
        return 0.0
    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if ofi_data.get("buyers_in_control") and ofi_data.get("ofi_trend", 0) > 0.1:
            return 0.5
        elif ofi_data.get("sellers_in_control"):
            return -0.5
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if ofi_data.get("sellers_in_control") and ofi_data.get("ofi_trend", 0) < -0.1:
            return 0.5
        elif ofi_data.get("buyers_in_control"):
            return -0.5
    return 0.0


def apply_fii_bias(score: float, fii_data: dict, strategy_type: str) -> float:
    """Adjusts score by ±0.5 based on FII/DII flow direction."""
    if not fii_data:
        return score
    bias = fii_data.get("bias", "NEUTRAL")
    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        if bias == "BULLISH":  return score + 0.5
        elif bias == "BEARISH": return score - 0.5
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if bias == "BEARISH":  return score + 0.5
        elif bias == "BULLISH": return score - 0.5
    return score


def is_strategy_gap_aligned(strategy_type: str, gap: dict) -> bool:
    """Blocks trades that go against the gap classification."""
    if not gap:
        return True
    gtype = gap.get("gap_type", "")
    if gtype == "BREAKAWAY":
        if "LONG" in strategy_type and gap.get("direction") == "DOWN":
            return False
        if "SHORT" in strategy_type and gap.get("direction") == "UP":
            return False
    if gtype == "EXHAUSTION":
        if "ORB" in strategy_type:
            return False
    return True


# ═══════════════════════════════════════════════════════════
# MAIN STRATEGY CLASSES
# ═══════════════════════════════════════════════════════════

class MarketRegime:
    @staticmethod
    def detect(data: Dict[str, Any], cfg: Dict) -> Dict[str, Any]:
        rc = cfg["regime"]
        adx_val  = data.get("adx")
        boll     = data.get("bollinger")
        current  = data["current_price"]
        vwap_val = data["vwap"]

        if adx_val is None:
            regime = "RANGE"
            return {
                "regime": regime, 
                "confidence": 30, 
                "size_mult": get_position_size_multiplier(regime),
                "allowed": get_allowed_strategies(regime)
            }

        vwap_dist_pct = ((current - vwap_val) / vwap_val) * 100
        bw = boll["bandwidth"] if boll else 0

        regime = "RANGE"
        confidence = 50

        # Strong trend: ADX alone is enough — don't require BB expansion (it lags)
        if adx_val > rc["strong_trend_adx"] and vwap_dist_pct > rc["vwap_strong_dist_pct"]:
            regime = "STRONG_TREND_UP"
            confidence = min(40 + int(adx_val) + int(bw * 5), 100)
        elif adx_val > rc["strong_trend_adx"] and vwap_dist_pct < -rc["vwap_strong_dist_pct"]:
            regime = "STRONG_TREND_DOWN"
            confidence = min(40 + int(adx_val) + int(bw * 5), 100)
        # Strong ADX but price near VWAP — still trending, use VWAP side to classify
        elif adx_val > rc["strong_trend_adx"]:
            regime = "STRONG_TREND_UP" if vwap_dist_pct >= 0 else "STRONG_TREND_DOWN"
            confidence = min(35 + int(adx_val), 90)
        elif rc["weak_trend_adx_min"] <= adx_val <= rc["weak_trend_adx_max"] and vwap_dist_pct > 0:
            regime = "WEAK_TREND_UP"
            confidence = min(30 + int(adx_val), 80)
        elif rc["weak_trend_adx_min"] <= adx_val <= rc["weak_trend_adx_max"] and vwap_dist_pct < 0:
            regime = "WEAK_TREND_DOWN"
            confidence = min(30 + int(adx_val), 80)
        elif adx_val < rc["range_adx"] and abs(vwap_dist_pct) < rc["vwap_range_dist_pct"]:
            regime = "RANGE"
            confidence = min(50 + int(20 - adx_val), 90)

        return {
            "regime": regime,
            "confidence": confidence,
            "size_mult": get_position_size_multiplier(regime),
            "allowed": get_allowed_strategies(regime),
        }

class OpeningRangeBreakout:
    def __init__(self):
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.captured = False
        self.capture_date: Optional[str] = None

    def capture_range(self, df_1m) -> None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.capture_date != today_str:
            self.captured = False
            self.orb_high = None
            self.orb_low = None
            self.capture_date = today_str

        if self.captured or df_1m is None or df_1m.empty:
            return
        if datetime.now().time() < dtime(9, 30):
            return
        
        # Only use today's data for opening range
        df_today = df_1m[df_1m.index.date == df_1m.index[-1].date()]
        if df_today.empty:
            return
        
        try:
            opening = df_today.between_time("09:15", "09:29")
            if not opening.empty:
                self.orb_high = round(float(opening["High"].max()), 2)
                self.orb_low = round(float(opening["Low"].min()), 2)
                self.captured = True
        except Exception:
            pass

    def evaluate(self, current_price: float, vol_ratio: Optional[float]) -> Dict[str, Any]:
        if not self.captured or self.orb_high is None:
            return {"orb_signal": "NONE", "orb_high": None, "orb_low": None}
        has_volume = vol_ratio is not None and vol_ratio >= 1.5
        if current_price > self.orb_high and has_volume:
            return {"orb_signal": "BUY_BREAKOUT", "orb_high": self.orb_high, "orb_low": self.orb_low}
        elif current_price < self.orb_low and has_volume:
            return {"orb_signal": "SELL_BREAKOUT", "orb_high": self.orb_high, "orb_low": self.orb_low}
        return {"orb_signal": "NONE", "orb_high": self.orb_high, "orb_low": self.orb_low}

class LiquiditySweepFilter:
    _wait_until_candle: int = 0
    @classmethod
    def detect(cls, df_1m, current_price: float, atr_val: float, cfg: Dict) -> Dict[str, Any]:
        lc = cfg["liquidity"]
        result = {"is_liquidity_trap": False, "trap_type": "NONE", "sweep_logged": False}

        if df_1m is None or len(df_1m) < 30 or atr_val is None or atr_val == 0:
            return result

        candle_count = len(df_1m)
        if candle_count < cls._wait_until_candle:
            result["is_liquidity_trap"] = True
            result["trap_type"] = "WAIT_AFTER_SWEEP"
            return result

        recent = df_1m.iloc[-30:]
        prev_high = float(recent["High"].iloc[:-1].max())
        prev_low = float(recent["Low"].iloc[:-1].min())

        last = df_1m.iloc[-1]
        c_high  = float(last["High"])
        c_low   = float(last["Low"])
        c_close = float(last["Close"])
        c_open  = float(last["Open"])
        c_vol   = float(last["Volume"])

        prev_vol = float(df_1m["Volume"].iloc[-2]) if len(df_1m) >= 2 else 0

        upper_wick = (c_high - max(c_close, c_open)) / atr_val
        lower_wick = (min(c_close, c_open) - c_low) / atr_val

        vol_confirmed = prev_vol > 0 and (c_vol / prev_vol) >= lc["sweep_volume_ratio"]

        if c_high > prev_high and c_close < prev_high and upper_wick > lc["wick_atr_threshold"] and vol_confirmed:
            result["is_liquidity_trap"] = True
            result["trap_type"] = "BEAR_TRAP"
            result["sweep_logged"] = True
            cls._wait_until_candle = candle_count + lc["re_entry_wait_candles"]
        elif c_low < prev_low and c_close > prev_low and lower_wick > lc["wick_atr_threshold"] and vol_confirmed:
            result["is_liquidity_trap"] = True
            result["trap_type"] = "BULL_TRAP"
            result["sweep_logged"] = True
            cls._wait_until_candle = candle_count + lc["re_entry_wait_candles"]

        return result

class CorrelationGuard:
    _active_symbols: List[str] = []

    @classmethod
    def add_position(cls, symbol: str):
        if symbol not in cls._active_symbols:
            cls._active_symbols.append(symbol)

    @classmethod
    def remove_position(cls, symbol: str):
        if symbol in cls._active_symbols:
            cls._active_symbols.remove(symbol)

    @classmethod
    def is_allowed(cls, new_symbol: str, cfg: Dict) -> bool:
        if not cls._active_symbols:
            return True
        max_corr = cfg["correlation"]["max_correlation"]
        lookback = cfg["correlation"]["lookback_days"]
        try:
            import yfinance as yf
            new_hist = yf.Ticker(new_symbol).history(period=f"{lookback + 5}d")["Close"]
            if len(new_hist) < lookback: return True
            new_returns = new_hist.pct_change().dropna().tail(lookback)

            for existing in cls._active_symbols:
                ex_hist = yf.Ticker(existing).history(period=f"{lookback + 5}d")["Close"]
                if len(ex_hist) < lookback: continue
                ex_returns = ex_hist.pct_change().dropna().tail(lookback)

                combined = pd.concat([new_returns, ex_returns], axis=1).dropna()
                if len(combined) < 10: continue
                corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                if abs(corr) > max_corr: return False
        except Exception:
            pass
        return True

    @classmethod
    def status(cls) -> str:
        if cls._active_symbols:
            return f"Active: {', '.join(cls._active_symbols)}"
        return "No positions"

# ═══════════════════════════════════════════════════════════
# V5.2 MASTER SCORER — 14 POINTS + MODIFIERS
# ═══════════════════════════════════════════════════════════
# Criterion          Max    Function                 Type
# ────────────────────────────────────────────────────────────
# Volume surge       2.0    score_volume()           CORE
# Supertrend (dual)  2.0    score_supertrend()       CORE
# Candle pattern     1.0    score_candle_pattern()   NEW V5.2
# Intraday S/R       1.0    score_intraday_sr()      NEW V5.2
# VWAP bands         1.0    score_vwap_bands()       UPGRADED
# EMA cross          1.0    score_ema_cross()        UPGRADED
# Dual RSI           1.0    score_rsi()              UPGRADED
# ADX (split)        1.0    score_adx()              UPGRADED
# Bollinger          1.0    score_bollinger()        UPGRADED
# Price move         1.0    (unchanged)              CORE
# VPOC proximity     1.0    score_vpoc()             NEW V5.2
# MACD divergence    0.5    score_macd()             UPGRADED
# ORB bonus          0.5    (unchanged)              CORE
# OFI imbalance      0.5    score_ofi()              NEW V5.2
# ────────────────────────────────────────────────────────────
# BASE TOTAL        14.0
# MODIFIERS (additive):
#   Sector bonus    ±0.5    get_sector_score_bonus()
#   FII/DII bias    ±0.5    apply_fii_bias()
# EFFECTIVE MAX     15.0
# MIN TO FIRE        7.0    morning / 8.0 afternoon
# PRE-FILTERS:
#   PCR direction   hard block
#   Gap alignment   hard block if gap conflicts
#   VIX extreme     hard block if VIX > 30
# ═══════════════════════════════════════════════════════════

class TradeScorer:
    @staticmethod
    def score_trade(data: Dict, pct_from_open: float, strategy_type: str, orb_signal: str, cfg: Dict) -> Dict[str, Any]:
        sc = cfg["scoring"]
        score = 0.0
        reasons: List[str] = []
        breakdown: List[str] = []

        current   = data["current_price"]
        vwap_data = data["vwap_data"]
        rsi_dual  = data["rsi_dual"]
        vol_r     = data["volume_ratio"]
        adx_val   = data["adx"]
        st_data   = data["supertrend"]
        boll      = data["bollinger"]
        pivots    = data.get("pivots", {})
        ema9_s    = data["ema9_series"]
        ema21_s   = data["ema21_series"]
        closes_s  = data["closes_series"]
        symbol    = data.get("symbol", "")

        # V5.2 new data fields
        candle_patterns = data.get("candle_patterns", {})
        intraday_sr     = data.get("intraday_sr", {})
        vpoc_data       = data.get("vpoc", {})
        ofi_data        = data.get("ofi", {})
        fii_data        = data.get("fii_dii", {})

        # V5.3 new data fields
        stoch_rsi   = data.get("stoch_rsi", {})
        cmf_val     = data.get("cmf", 0.0)
        ichimoku    = data.get("ichimoku", {})
        williams_r  = data.get("williams_r", None)
        global_bias = data.get("global_bias", {"bias": "NEUTRAL", "score_mod": 0.0})
        roc_val     = data.get("roc", None)

        # ── V5.1 Core Criteria ──

        # Price Move
        if strategy_type in ("ORB_LONG", "REVERSAL_SHORT"): 
            if pct_from_open >= sc["move_threshold_pct"]:
                score += sc["weight_move"]
                reasons.append(f"↑{pct_from_open:.2f}%")
                breakdown.append(f"move:+{sc['weight_move']}")
        elif strategy_type in ("ORB_SHORT", "REVERSAL_LONG"):
            if pct_from_open <= -sc["move_threshold_pct"]:
                score += sc["weight_move"]
                reasons.append(f"↓{abs(pct_from_open):.2f}%")
                breakdown.append(f"move:+{sc['weight_move']}")

        # VWAP Bands
        vwap_sc = score_vwap_bands(current, vwap_data, pivots, strategy_type, cfg)
        if vwap_sc > 0:
            val = sc["weight_vwap"] * vwap_sc
            score += val
            reasons.append("VWAP Bands")
            breakdown.append(f"vwap:+{val}")

        # EMA Cross
        ema_sc = score_ema_cross(ema9_s, ema21_s, strategy_type, 3)
        if ema_sc > 0:
            val = sc["weight_candle"] * ema_sc
            score += val
            reasons.append(f"EMA Cross {ema_sc}")
            breakdown.append(f"ema:+{val}")

        # Dual RSI
        rsi_sc = score_rsi(rsi_dual["rsi7"], rsi_dual["rsi14"], strategy_type)
        if rsi_sc > 0:
            val = sc["weight_rsi"] * rsi_sc
            score += val
            reasons.append(f"RSI aligned")
            breakdown.append(f"rsi:+{val}")

        # Volume Surge
        vol_sc = score_volume(vol_r)
        if vol_sc != 0:
            score += vol_sc
            reasons.append(f"Vol {vol_r}x")
            breakdown.append(f"vol:{'+' if vol_sc>0 else ''}{vol_sc}")

        # Supertrend
        st_sc = score_supertrend(st_data, strategy_type)
        if st_sc > 0:
            score += st_sc
            reasons.append(f"ST {st_sc}")
            breakdown.append(f"st:+{st_sc}")

        # ROC — replaces MACD as momentum filter (less lag on 1m charts)
        if roc_val is not None:
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                if roc_val > 0.4:
                    score += 1.0; reasons.append(f"ROC+{roc_val:.2f}"); breakdown.append("roc:+1.0")
                elif roc_val > 0.1:
                    score += 0.5; reasons.append(f"ROC+{roc_val:.2f}"); breakdown.append("roc:+0.5")
                elif roc_val < -0.3:
                    score -= 0.5; breakdown.append("roc:-0.5")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                if roc_val < -0.4:
                    score += 1.0; reasons.append(f"ROC{roc_val:.2f}"); breakdown.append("roc:+1.0")
                elif roc_val < -0.1:
                    score += 0.5; reasons.append(f"ROC{roc_val:.2f}"); breakdown.append("roc:+0.5")
                elif roc_val > 0.3:
                    score -= 0.5; breakdown.append("roc:-0.5")

        # Bollinger Bands
        bb_sc = score_bollinger(closes_s, boll["upper"], boll["lower"], boll["bandwidth"], strategy_type)
        if bb_sc > 0:
            val = sc["weight_bollinger"] * bb_sc
            score += val
            reasons.append(f"BB Squeeze/Touch")
            breakdown.append(f"bb:+{val}")

        # ADX Strength
        adx_sc = score_adx(adx_val, strategy_type, cfg)
        if adx_sc > 0:
            val = sc["weight_adx"] * adx_sc
            score += val
            reasons.append(f"ADX {adx_val}")
            breakdown.append(f"adx:+{val}")

        # ORB Bonus
        orb_sc = 0.5 if (orb_signal != "NONE" and vol_r >= 1.5 and
                        ((strategy_type == "ORB_LONG" and orb_signal == "BUY_BREAKOUT") or 
                         (strategy_type == "ORB_SHORT" and orb_signal == "SELL_BREAKOUT"))) else 0.0
        if orb_sc > 0:
            score += orb_sc
            reasons.append("ORB breakout")
            breakdown.append(f"orb:+{orb_sc}")

        # ── V5.2 New Criteria ──

        # Candle Pattern (max +1.0)
        cp_sc = score_candle_pattern(candle_patterns, strategy_type)
        if cp_sc != 0:
            score += cp_sc
            pnames = ','.join(candle_patterns.keys()) if candle_patterns else 'NONE'
            reasons.append(f"Pattern:{pnames}")
            breakdown.append(f"cp:{'+' if cp_sc>0 else ''}{cp_sc}")

        # Intraday S/R (max +1.0)
        sr_sc = score_intraday_sr(current, intraday_sr, strategy_type)
        if sr_sc > 0:
            score += sr_sc
            reasons.append("S/R Level")
            breakdown.append(f"sr:+{sr_sc}")

        # VPOC (max +1.0)
        vp_sc = score_vpoc(current, vpoc_data, strategy_type)
        if vp_sc > 0:
            score += vp_sc
            reasons.append("VPOC")
            breakdown.append(f"vpoc:+{vp_sc}")

        # OFI (max +0.5)
        ofi_sc = score_ofi(ofi_data, strategy_type)
        if ofi_sc != 0:
            score += ofi_sc
            reasons.append(f"OFI")
            breakdown.append(f"ofi:{'+' if ofi_sc>0 else ''}{ofi_sc}")

        # ── V5.3 New Indicator Scores ──

        # Stochastic RSI (max +1.0)
        # Oversold %K crossing above %D = bullish; overbought crossing below = bearish
        if stoch_rsi and stoch_rsi.get("k") is not None:
            k = stoch_rsi["k"]
            k_above_d = stoch_rsi.get("k_above_d")
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                if stoch_rsi.get("oversold") and k_above_d:
                    score += 1.0; reasons.append("StochRSI OB↑"); breakdown.append("srsi:+1.0")
                elif k < 50 and k_above_d:
                    score += 0.5; reasons.append("StochRSI↑"); breakdown.append("srsi:+0.5")
                elif stoch_rsi.get("overbought"):
                    score -= 0.5; breakdown.append("srsi:-0.5")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                if stoch_rsi.get("overbought") and not k_above_d:
                    score += 1.0; reasons.append("StochRSI OS↓"); breakdown.append("srsi:+1.0")
                elif k > 50 and not k_above_d:
                    score += 0.5; reasons.append("StochRSI↓"); breakdown.append("srsi:+0.5")
                elif stoch_rsi.get("oversold"):
                    score -= 0.5; breakdown.append("srsi:-0.5")

        # Chaikin Money Flow (max +1.0)
        # CMF > +0.1 = sustained buying pressure; < -0.1 = sustained selling
        if cmf_val is not None:
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                if cmf_val > 0.2:
                    score += 1.0; reasons.append(f"CMF+{cmf_val:.2f}"); breakdown.append("cmf:+1.0")
                elif cmf_val > 0.1:
                    score += 0.5; reasons.append(f"CMF+{cmf_val:.2f}"); breakdown.append("cmf:+0.5")
                elif cmf_val < -0.1:
                    score -= 0.5; breakdown.append(f"cmf:-0.5")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                if cmf_val < -0.2:
                    score += 1.0; reasons.append(f"CMF{cmf_val:.2f}"); breakdown.append("cmf:+1.0")
                elif cmf_val < -0.1:
                    score += 0.5; reasons.append(f"CMF{cmf_val:.2f}"); breakdown.append("cmf:+0.5")
                elif cmf_val > 0.1:
                    score -= 0.5; breakdown.append("cmf:-0.5")

        # Ichimoku Cloud (max +1.0)
        # Price above cloud + Tenkan > Kijun = strong bullish confluence
        if ichimoku and ichimoku.get("above_cloud") is not None:
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                if ichimoku.get("above_cloud") and ichimoku.get("tenkan_above_kijun") and ichimoku.get("cloud_bullish"):
                    score += 1.0; reasons.append("Ichimoku↑↑"); breakdown.append("ichi:+1.0")
                elif ichimoku.get("above_cloud") and ichimoku.get("tenkan_above_kijun"):
                    score += 0.5; reasons.append("Ichimoku↑"); breakdown.append("ichi:+0.5")
                elif ichimoku.get("below_cloud"):
                    score -= 0.5; breakdown.append("ichi:-0.5")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                if ichimoku.get("below_cloud") and not ichimoku.get("tenkan_above_kijun") and ichimoku.get("cloud_bullish") is False:
                    score += 1.0; reasons.append("Ichimoku↓↓"); breakdown.append("ichi:+1.0")
                elif ichimoku.get("below_cloud") and not ichimoku.get("tenkan_above_kijun"):
                    score += 0.5; reasons.append("Ichimoku↓"); breakdown.append("ichi:+0.5")
                elif ichimoku.get("above_cloud"):
                    score -= 0.5; breakdown.append("ichi:-0.5")

        # Williams %R (max +0.5)
        # Confirms overbought/oversold — used as additional momentum filter
        if williams_r is not None:
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                if williams_r < -80:   # oversold — good for reversal long
                    score += 0.5; reasons.append(f"W%R{williams_r:.0f}"); breakdown.append("wr:+0.5")
                elif williams_r > -20: # overbought — bad for longs
                    score -= 0.25; breakdown.append("wr:-0.25")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                if williams_r > -20:   # overbought — good for reversal short
                    score += 0.5; reasons.append(f"W%R{williams_r:.0f}"); breakdown.append("wr:+0.5")
                elif williams_r < -80: # oversold — bad for shorts
                    score -= 0.25; breakdown.append("wr:-0.25")

        # ── V5.2 Modifiers (push above/below base) ──

        # Sector Bonus (±0.5)
        try:
            from execution import get_sector_score_bonus
            sect_sc = get_sector_score_bonus(symbol, strategy_type)
            if sect_sc != 0:
                score += sect_sc
                reasons.append(f"Sector:{'+' if sect_sc>0 else ''}{sect_sc}")
                breakdown.append(f"sect:{'+' if sect_sc>0 else ''}{sect_sc}")
        except Exception:
            pass

        # FII/DII Bias (±0.5)
        score = apply_fii_bias(score, fii_data, strategy_type)

        # Global Market Bias (±0.5) — S&P500, DXY, US VIX overnight direction
        g_mod = global_bias.get("score_mod", 0.0)
        g_bias = global_bias.get("bias", "NEUTRAL")
        if g_mod != 0.0:
            if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
                score += g_mod
                reasons.append(f"Global:{g_bias}")
                breakdown.append(f"glbl:{'+' if g_mod>0 else ''}{g_mod}")
            elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
                score -= g_mod   # bearish global = good for shorts
                reasons.append(f"Global:{g_bias}")
                breakdown.append(f"glbl:{'+' if -g_mod>0 else ''}{-g_mod}")

        return {"score": round(score, 1), "reasons": reasons, "breakdown": "|".join(breakdown)}


class MasterStrategy:
    @staticmethod
    def evaluate(
        data: Dict, regime_info: Dict, orb_result: Dict,
        liquidity: Dict, cfg: Dict,
    ) -> Dict[str, Any]:
        current = data["current_price"]
        open_p  = data["open_price"]
        atr_val = data.get("atr")
        st      = data["supertrend"]
        rm      = cfg["risk_management"]

        pct_from_open  = round(((current - open_p) / open_p) * 100, 2)
        intraday_range = round(data["high"] - data["low"], 2)
        now_time       = datetime.now().time()
        no_rev_before  = dtime(*map(int, cfg["session"]["no_reversal_before"].split(":")))
        close_time     = dtime(*map(int, cfg["session"]["safe_close"].split(":")))

        regime     = regime_info["regime"]
        confidence = regime_info["confidence"]
        
        allowed    = get_allowed_strategies(regime)

        if now_time > close_time:
            return _result("NO TRADE", 0, pct_from_open, intraday_range, "⏳ Past trading window", "")

        # Data quality gate — reject obviously corrupted candles
        vol_ratio = data.get("volume_ratio", 1.0)
        atr_check = data.get("atr")
        if vol_ratio is not None and vol_ratio > 15.0:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"BAD_DATA: vol_ratio={vol_ratio}x (capped at 15x)", "")
        if atr_check is not None and current > 0 and (atr_check / current) > 0.05:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"BAD_DATA: ATR={atr_check} is >5% of price (data anomaly)", "")

        # PCR Filter pre-check
        pcr = data.get("pcr", 1.0)
        if allowed and not pcr_direction_ok(pcr, allowed[0]):
            return _result("NO TRADE", 0, pct_from_open, intraday_range, f"PCR_BLOCK={pcr}", "")

        # V5.2: Gap alignment filter
        gap_data = data.get("gap_classification", {})
        if allowed and gap_data:
            if not is_strategy_gap_aligned(allowed[0], gap_data):
                return _result("NO TRADE", 0, pct_from_open, intraday_range,
                               f"GAP_BLOCK={gap_data.get('gap_type','?')}", "")

        # V5.2: VIX extreme block
        vix = data.get("india_vix", 15.0)
        if vix > 30:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"VIX_EXTREME={vix}", "")

        is_early = now_time < no_rev_before

        if confidence < cfg["regime"]["min_confidence"]:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"Regime confidence too low ({confidence}%)", "")

        if liquidity["is_liquidity_trap"]:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"🔒 {liquidity['trap_type']}", "")

        adx_val = data.get("adx")
        if adx_val is not None and adx_val < 12:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"Choppy (ADX={adx_val})", "")

        min_score = get_min_score(now_time, cfg["scoring"])
        orb_sig = orb_result["orb_signal"]

        signal = "NO TRADE"
        score = 0.0
        entry = sl = tgt = 0.0
        reasons = "Conditions not met. Waiting..."
        breakdown = ""

        if not allowed:
            return _result("NO TRADE", 0, pct_from_open, intraday_range, "No strategies allowed in this regime", "")

        # ── Breakout trades (ORB) ──
        if orb_sig != "NONE":
            if orb_sig == "BUY_BREAKOUT" and "ORB_LONG" in allowed:
                buy = TradeScorer.score_trade(data, pct_from_open, "ORB_LONG", orb_sig, cfg)
                if buy["score"] >= min_score:
                    signal = "BUY"
                    score = buy["score"]
                    entry = current
                    sl, tgt = _dynamic_sl_tgt(entry, "BUY", atr_val, rm)
                    reasons = " | ".join(buy["reasons"])
                    breakdown = buy["breakdown"]

            elif orb_sig == "SELL_BREAKOUT" and "ORB_SHORT" in allowed:
                sell = TradeScorer.score_trade(data, pct_from_open, "ORB_SHORT", orb_sig, cfg)
                if sell["score"] >= min_score:
                    signal = "SELL"
                    score = sell["score"]
                    entry = current
                    sl, tgt = _dynamic_sl_tgt(entry, "SELL", atr_val, rm)
                    reasons = " | ".join(sell["reasons"])
                    breakdown = sell["breakdown"]

        # ── Reversal trades ──
        if signal == "NO TRADE" and not is_early:
            move_thr = cfg["scoring"]["move_threshold_pct"]

            if "REVERSAL_LONG" in allowed and pct_from_open <= -move_thr:
                buy = TradeScorer.score_trade(data, pct_from_open, "REVERSAL_LONG", orb_sig, cfg)
                if buy["score"] >= min_score:
                    signal = "BUY"
                    score = buy["score"]
                    entry = current
                    sl, tgt = _dynamic_sl_tgt(entry, "BUY", atr_val, rm)
                    reasons = " | ".join(buy["reasons"])
                    breakdown = buy["breakdown"]

            if signal == "NO TRADE" and "REVERSAL_SHORT" in allowed and pct_from_open >= move_thr:
                sell = TradeScorer.score_trade(data, pct_from_open, "REVERSAL_SHORT", orb_sig, cfg)
                if sell["score"] >= min_score:
                    signal = "SELL"
                    score = sell["score"]
                    entry = current
                    sl, tgt = _dynamic_sl_tgt(entry, "SELL", atr_val, rm)
                    reasons = " | ".join(sell["reasons"])
                    breakdown = sell["breakdown"]

        # ── EMA9 Pullback trades (trend continuation) ──
        if signal == "NO TRADE":
            ema9_series  = data.get("ema9_series")
            ema21_series = data.get("ema21_series")
            if ema9_series is not None and len(ema9_series) >= 1 and ema21_series is not None and len(ema21_series) >= 1:
                ema9_val  = float(ema9_series.iloc[-1])
                ema21_val = float(ema21_series.iloc[-1])
                pullback_thr   = cfg["scoring"].get("ema9_pullback_pct", 0.5) / 100
                dist_from_ema9 = (current - ema9_val) / ema9_val

                if "EMA_PULLBACK_LONG" in allowed and ema9_val > ema21_val and 0 <= dist_from_ema9 <= pullback_thr:
                    pull = TradeScorer.score_trade(data, pct_from_open, "ORB_LONG", orb_sig, cfg)
                    if pull["score"] >= min_score:
                        signal = "BUY"; score = pull["score"]; entry = current
                        sl, tgt = _dynamic_sl_tgt(entry, "BUY", atr_val, rm)
                        reasons = "EMA9 Pullback | " + " | ".join(pull["reasons"])
                        breakdown = "ema9pull|" + pull["breakdown"]

                if signal == "NO TRADE" and "EMA_PULLBACK_SHORT" in allowed and ema9_val < ema21_val and -pullback_thr <= dist_from_ema9 <= 0:
                    pull = TradeScorer.score_trade(data, pct_from_open, "ORB_SHORT", orb_sig, cfg)
                    if pull["score"] >= min_score:
                        signal = "SELL"; score = pull["score"]; entry = current
                        sl, tgt = _dynamic_sl_tgt(entry, "SELL", atr_val, rm)
                        reasons = "EMA9 Pullback | " + " | ".join(pull["reasons"])
                        breakdown = "ema9pull|" + pull["breakdown"]

        # ── VWAP Bounce trades (most reliable intraday setup) ──
        # Price touches VWAP from the trend side then reclaims it with volume = institutional re-entry
        if signal == "NO TRADE" and not is_early:
            vwap_val  = data.get("vwap", 0)
            vol_ratio = data.get("volume_ratio", 0)
            if vwap_val and vwap_val > 0:
                vwap_bounce_dist = cfg["scoring"].get("vwap_bounce_dist_pct", 0.25) / 100
                dist_from_vwap   = (current - vwap_val) / vwap_val  # signed

                # VWAP Bounce LONG: price is just above VWAP (within 0.25%), vol elevated, trending up
                if (0 <= dist_from_vwap <= vwap_bounce_dist and vol_ratio >= 1.3
                        and regime in ("STRONG_TREND_UP", "WEAK_TREND_UP")):
                    bounce = TradeScorer.score_trade(data, pct_from_open, "ORB_LONG", orb_sig, cfg)
                    if bounce["score"] >= min_score:
                        signal = "BUY"; score = bounce["score"]; entry = current
                        sl, tgt = _dynamic_sl_tgt(entry, "BUY", atr_val, rm)
                        reasons = "VWAP Bounce | " + " | ".join(bounce["reasons"])
                        breakdown = "vwapbounce|" + bounce["breakdown"]

                # VWAP Bounce SHORT: price is just below VWAP, vol elevated, trending down
                if (signal == "NO TRADE" and -vwap_bounce_dist <= dist_from_vwap <= 0
                        and vol_ratio >= 1.3
                        and regime in ("STRONG_TREND_DOWN", "WEAK_TREND_DOWN")):
                    bounce = TradeScorer.score_trade(data, pct_from_open, "ORB_SHORT", orb_sig, cfg)
                    if bounce["score"] >= min_score:
                        signal = "SELL"; score = bounce["score"]; entry = current
                        sl, tgt = _dynamic_sl_tgt(entry, "SELL", atr_val, rm)
                        reasons = "VWAP Bounce | " + " | ".join(bounce["reasons"])
                        breakdown = "vwapbounce|" + bounce["breakdown"]

        grade = "A+" if score >= cfg["scoring"]["grade_a_plus"] else ("A" if score >= cfg["scoring"]["grade_a"] else "-")
        stars = f"{'★' * min(int(score), 14)}{'☆' * max(14 - int(score), 0)}  {score}/14" if score > 0 else "—"

        return {
            "signal": signal, "score": score, "grade": grade,
            "score_display": stars, "entry": entry,
            "stop_loss": sl, "target": tgt,
            "reason": reasons, "breakdown": breakdown,
            "pct_from_open": pct_from_open, "intraday_range": intraday_range,
        }

def _dynamic_sl_tgt(entry: float, direction: str, atr_val, rm: Dict):
    if atr_val and atr_val > 0:
        sl_off = atr_val * rm["sl_atr_multiplier"]
        if "fixed_target_pct" in rm and rm["fixed_target_pct"] > 0:
            tgt_off = entry * (rm["fixed_target_pct"] / 100)
        else:
            tgt_off = atr_val * rm["target_atr_multiplier"]
    else:
        sl_off = entry * 0.005
        if "fixed_target_pct" in rm and rm["fixed_target_pct"] > 0:
            tgt_off = entry * (rm["fixed_target_pct"] / 100)
        else:
            tgt_off = entry * 0.015
    if direction == "BUY":
        return round(entry - sl_off, 2), round(entry + tgt_off, 2)
    else:
        return round(entry + sl_off, 2), round(entry - tgt_off, 2)

def _result(signal, score, pct, rng, reason, breakdown):
    return {
        "signal": signal, "score": score, "grade": "-",
        "score_display": "—", "entry": 0, "stop_loss": 0, "target": 0,
        "reason": reason, "breakdown": breakdown,
        "pct_from_open": pct, "intraday_range": rng,
    }
