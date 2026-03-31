import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, time as dtime

# REGIME MAPPING
REGIME_STRATEGY_MAP = {
    "STRONG_TREND_UP":   ["ORB_LONG"],
    "WEAK_TREND_UP":     ["ORB_LONG"],
    "RANGE":             ["REVERSAL_LONG", "REVERSAL_SHORT"],
    "WEAK_TREND_DOWN":   ["ORB_SHORT"],
    "STRONG_TREND_DOWN": ["ORB_SHORT"],
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
        if pcr < 0.75: return False   
    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        if pcr > 1.25: return False   
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

        if adx_val > rc["strong_trend_adx"] and vwap_dist_pct > rc["vwap_strong_dist_pct"] and bw > rc["bb_strong_bw"]:
            regime = "STRONG_TREND_UP"
            confidence = min(40 + int(adx_val) + int(bw * 10), 100)
        elif adx_val > rc["strong_trend_adx"] and vwap_dist_pct < -rc["vwap_strong_dist_pct"] and bw > rc["bb_strong_bw"]:
            regime = "STRONG_TREND_DOWN"
            confidence = min(40 + int(adx_val) + int(bw * 10), 100)
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
        has_volume = vol_ratio is not None and vol_ratio >= 2.0  
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
# UPGRADED 11-POINT SCORER — V5.1
# ═══════════════════════════════════════════════════════════
# Criterion         Weight   Function              Notes
# ─────────────────────────────────────────────────────────
# Volume surge      2.0      score_volume()        Same-time 5-day baseline
# Supertrend        2.0      score_supertrend()    1-min + 5-min agreement
# VWAP bands        1.0      score_vwap_bands()    ±1σ/±2σ positioning
# EMA cross         1.0      score_ema_cross()     Replaces candle type
# RSI (dual)        1.0      score_rsi()           RSI7 trigger + RSI14 confirm
# ADX (split)       1.0      score_adx()           Direction-aware threshold
# Bollinger         1.0      score_bollinger()     Strategy-aware (squeeze vs touch)
# Price move        1.0      (unchanged)           ≥ 0.7% from open
# MACD divergence   0.5      score_macd()          Divergence only, not cross
# ORB bonus         0.5      (unchanged)           Volume-confirmed ORB only
# ─────────────────────────────────────────────────────────
# MAX TOTAL         11.0
# MIN TO FIRE        7.0     (morning) / 8.0 (afternoon)
# PCR FILTER        pre-check — blocks counter-trend trades silently
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
        macd_div  = data["macd_div"]
        boll      = data["bollinger"]
        pivots    = data.get("pivots", {})
        ema9_s    = data["ema9_series"]
        ema21_s   = data["ema21_series"]
        closes_s  = data["closes_series"]

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

        # MACD Divergence
        macd_sc = score_macd(macd_div, strategy_type)
        if macd_sc > 0:
            score += macd_sc
            reasons.append(f"MACD Div")
            breakdown.append(f"macd:+{macd_sc}")

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
        orb_sc = 0.5 if (orb_signal != "NONE" and vol_r >= 2.0 and 
                        ((strategy_type == "ORB_LONG" and orb_signal == "BUY_BREAKOUT") or 
                         (strategy_type == "ORB_SHORT" and orb_signal == "SELL_BREAKOUT"))) else 0.0
        if orb_sc > 0:
            val = sc["weight_orb"] * orb_sc
            score += orb_sc
            reasons.append("ORB breakout")
            breakdown.append(f"orb:+{orb_sc}")

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

        # PCR Filter pre-check
        pcr = data.get("pcr", 1.0)
        if allowed and not pcr_direction_ok(pcr, allowed[0]):
            return _result("NO TRADE", 0, pct_from_open, intraday_range, f"PCR_BLOCK={pcr}", "")

        is_early = now_time < no_rev_before

        if confidence < cfg["regime"]["min_confidence"]:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"Regime confidence too low ({confidence}%)", "")

        if liquidity["is_liquidity_trap"]:
            return _result("NO TRADE", 0, pct_from_open, intraday_range,
                           f"🔒 {liquidity['trap_type']}", "")

        adx_val = data.get("adx")
        if adx_val is not None and adx_val < 15:
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

        grade = "A+" if score >= cfg["scoring"]["grade_a_plus"] else ("A" if score >= cfg["scoring"]["grade_a"] else "-")
        stars = f"{'★' * int(score)}{'☆' * (11 - int(score))}  {score}/11" if score > 0 else "—"

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
