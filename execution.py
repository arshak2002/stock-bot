"""
execution.py — V5.0 Data Provider, Slippage, News, Telegram, Logger
=====================================================================
All I/O lives here. Strategy and risk modules stay pure logic.
"""

import os
import csv
import json
import hashlib
import subprocess
import time as _time
import yaml
from collections import deque
from datetime import datetime, time as dtime, timedelta
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
import yfinance as yf

from indicators import (
    get_dual_rsi, ema, calculate_vwap_with_bands, calculate_volume_surge_ratio, atr,
    get_dual_timeframe_supertrend, macd, detect_macd_divergence,
    calculate_bollinger_with_bandwidth, adx, pivot_points,
    calculate_stoch_rsi, calculate_cmf, calculate_ichimoku, calculate_williams_r,
    calculate_roc,
)


# ══════════════════════════════════════════════
#  CONFIG LOADER (hot-reload)
# ══════════════════════════════════════════════

_CONFIG_CACHE: Dict[str, Any] = {}
_CONFIG_MTIME: float = 0.0

def load_config(path: str = None) -> Dict[str, Any]:
    """Load config.yaml with hot-reload on file change."""
    global _CONFIG_CACHE, _CONFIG_MTIME
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        mtime = os.path.getmtime(path)
        if mtime != _CONFIG_MTIME or not _CONFIG_CACHE:
            with open(path, "r") as f:
                _CONFIG_CACHE = yaml.safe_load(f)
            _CONFIG_MTIME = mtime
    except Exception as e:
        if not _CONFIG_CACHE:
            raise RuntimeError(f"Cannot load config: {e}")
    return _CONFIG_CACHE


# ══════════════════════════════════════════════
#  SESSION FILTER
# ══════════════════════════════════════════════

class SessionFilter:
    NSE_HOLIDAYS_STR = {
        "2026-01-26", "2026-03-03", "2026-03-20", "2026-03-31", "2026-04-03", "2026-04-14",
        "2026-05-01", "2026-08-15", "2026-10-02", "2026-11-09", "2026-11-24", "2026-12-25"
    }

    @classmethod
    def is_holiday(cls) -> bool:
        today = datetime.now()
        if today.weekday() >= 5: # Saturday = 5, Sunday = 6
            return True
        if today.strftime("%Y-%m-%d") in cls.NSE_HOLIDAYS_STR:
            return True
        return False

    @staticmethod
    def is_trading_window(cfg: Dict) -> bool:
        if SessionFilter.is_holiday():
            return False
        s = cfg["session"]
        o = dtime(*map(int, s["safe_open"].split(":")))
        c = dtime(*map(int, s["safe_close"].split(":")))
        return o <= datetime.now().time() <= c

    @staticmethod
    def status(cfg: Dict) -> str:
        if SessionFilter.is_holiday():
            return "HOLIDAY (CLOSED)"
            
        now = datetime.now().time()
        s = cfg["session"]
        o = dtime(*map(int, s["safe_open"].split(":")))
        c = dtime(*map(int, s["safe_close"].split(":")))
        if now < dtime(9, 15):
            return "PRE-MARKET"
        elif now < o:
            return "⚠️  OPENING (9:15–9:30)"
        elif now <= c:
            return "✅ ACTIVE"
        elif now <= dtime(15, 30):
            return "⚠️  CLOSING ZONE"
        else:
            return "MARKET CLOSED"


# ══════════════════════════════════════════════
#  F1: DATA PROVIDER — with validation & caching
# ══════════════════════════════════════════════

class DataCache:
    """Fixed-size deque cache for last N candles per symbol."""
    def __init__(self, maxlen: int = 75):
        self._cache: Dict[str, deque] = {}
        self.maxlen = maxlen

    def update(self, symbol: str, df: pd.DataFrame):
        if symbol not in self._cache:
            self._cache[symbol] = deque(maxlen=self.maxlen)
        for _, row in df.iterrows():
            self._cache[symbol].append(row.to_dict())

    def get_df(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._cache and len(self._cache[symbol]) > 0:
            return pd.DataFrame(list(self._cache[symbol]))
        return None


def _validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Reject candles with bad data: volume=0, OHLC anomalies."""
    if df.empty:
        return df
    # Remove volume=0 candles
    df = df[df["Volume"] > 0].copy()
    # Remove OHLC anomalies
    df = df[df["High"] >= df["Low"]].copy()
    df = df[df["Close"] <= df["High"]].copy()
    df = df[df["Close"] >= df["Low"]].copy()
    return df


# ══════════════════════════════════════════════
#  NSE INDIA OFFICIAL API (0-Delay Live Feed)
# ══════════════════════════════════════════════

class NSEIndiaLiveFeed:
    """
    Scrapes the official NSE India website for absolute real-time quotes.
    Bypasses yfinance's occasional 15-min free tier delay.
    Maintains a session to reuse cookies for bot protection bypass.
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._init_cookies()

    def _init_cookies(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            print(f"[WARN] NSE init failed: {e}")

    def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean_sym = symbol.replace(".NS", "").upper()
        url = f"https://www.nseindia.com/api/quote-equity?symbol={clean_sym}"
        try:
            res = self.session.get(url, timeout=5)
            if res.status_code == 401:
                self._init_cookies()  # Refresh cookies and retry
                res = self.session.get(url, timeout=5)
            
            data = res.json()["priceInfo"]
            return {
                "last_price": float(data["lastPrice"]),
                "open": float(data["open"]),
                "day_high": float(data["intraDayHighLow"]["max"]),
                "day_low": float(data["intraDayHighLow"]["min"]),
                "prev_close": float(data["previousClose"])
            }
        except Exception as e:
            print(f"[WARN] NSE API fetch failed: {e}. Falling back to yfinance.")
            return None

class YFinanceProvider:
    """Enhanced data provider with validation, caching, and fallback."""

    def __init__(self, symbol: str):
        if "." not in symbol:
            symbol = f"{symbol}.NS"
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self.cache = DataCache(maxlen=75)
        self._fail_count = 0
        self.nse_live = NSEIndiaLiveFeed()

    def _rebuild_ticker(self):
        """Fallback: recreate ticker object on repeated failures."""
        self.ticker = yf.Ticker(self.symbol)
        self._fail_count = 0

    def fetch(self, cfg: Dict) -> Optional[Dict[str, Any]]:
        ind = cfg["indicators"]
        try:
            # First try official NSE India for 0-delay current price
            nse_quote = self.nse_live.fetch_quote(self.symbol)
            info = self.ticker.fast_info
            
            if nse_quote:
                current_price = nse_quote["last_price"]
                open_price    = nse_quote["open"]
                day_high      = nse_quote["day_high"]
                day_low       = nse_quote["day_low"]
                prev_close    = nse_quote["prev_close"]
            else:
                # Fallback to yfinance
                current_price = float(info.last_price)
                open_price    = float(info.open)
                day_high      = float(info.day_high)
                day_low       = float(info.day_low)
                prev_close    = float(info.previous_close)
                
            volume_val = int(info.last_volume)

            # 1-min candles — validated
            df_1m_raw = self.ticker.history(period="5d", interval="1m")
            df_1m = _validate_candles(df_1m_raw)
            if df_1m.empty or len(df_1m) < 5:
                raise ValueError("Insufficient valid 1m candles")

            # Cache for later use
            self.cache.update(self.symbol, df_1m)

            closes_1m  = df_1m["Close"]
            volumes_1m = df_1m["Volume"]

            # 5-min candles (multi-TF)
            df_5m = self.ticker.history(period="5d", interval="5m")
            df_5m = _validate_candles(df_5m)
            closes_5m = df_5m["Close"] if not df_5m.empty else pd.Series(dtype=float)

            # Daily candles (CPR)
            df_daily = self.ticker.history(period="5d", interval="1d")

            # Prepare df_today since vwap resets daily
            df_1m_today = df_1m[df_1m.index.date == df_1m.index[-1].date()].copy()
            if df_1m_today.empty:
                df_1m_today = df_1m.copy()

            # Compute all indicators
            vwap_data  = calculate_vwap_with_bands(df_1m_today)
            rsi_dual   = get_dual_rsi(closes_1m)
            ema9_val   = ema(closes_1m, ind["ema_fast"])
            ema21_val  = ema(closes_1m, ind["ema_slow"])

            # EMA Series for EMA Cross + Pullback strategy
            ema9_series  = closes_1m.ewm(span=ind["ema_fast"], adjust=False).mean()
            ema21_series = closes_1m.ewm(span=ind["ema_slow"], adjust=False).mean()

            # Use last completed 1m candle volume — more reliable than live tick volume
            last_candle_vol = int(df_1m["Volume"].iloc[-1]) if not df_1m.empty else volume_val
            vol_r_raw  = calculate_volume_surge_ratio(last_candle_vol, df_1m_raw, df_1m_raw.index[-1].time())
            vol_r      = min(vol_r_raw, 20.0)  # cap: ratios >20x are data artifacts, not real surges
            atr_val    = atr(df_1m, ind["atr_period"])
            st_data    = get_dual_timeframe_supertrend(df_1m, df_5m, ind["supertrend_period"], ind["supertrend_multiplier"])
            macd_data  = macd(closes_1m, ind["macd_fast"], ind["macd_slow"], ind["macd_signal"])
            macd_div   = detect_macd_divergence(closes_1m, macd_data["hist_series"], 5) if macd_data else "NONE"
            boll_data  = calculate_bollinger_with_bandwidth(closes_1m, ind["bollinger_period"], ind["bollinger_std"])
            adx_val    = adx(df_1m, ind["adx_period"])

            # ── New V5.3 Indicators ──
            stoch_rsi_data  = calculate_stoch_rsi(closes_1m)
            cmf_val         = calculate_cmf(df_1m)
            ichimoku_data   = calculate_ichimoku(df_1m)
            williams_r_val  = calculate_williams_r(df_1m)
            roc_val         = calculate_roc(closes_1m, period=10)  # 10-bar momentum on 1m = 10-min ROC

            pivots_data = None
            if len(df_daily) >= 2:
                prev_day = df_daily.iloc[-2]
                pivots_data = pivot_points(
                    float(prev_day["High"]), float(prev_day["Low"]), float(prev_day["Close"]),
                )

            gap_pct = round(((open_price - prev_close) / prev_close) * 100, 2)
            prev_candle = float(closes_1m.iloc[-2]) if len(closes_1m) > 1 else current_price
            pcr_val = fetch_pcr("NIFTY")

            self._fail_count = 0
            return {
                "symbol": self.symbol,
                "current_price": round(current_price, 2),
                "open_price": round(open_price, 2),
                "high": round(day_high, 2),
                "low": round(day_low, 2),
                "volume": volume_val,
                "prev_close": round(prev_close, 2),
                "vwap_data": vwap_data,
                "vwap": round(vwap_data.get("vwap", current_price) or current_price, 2),
                "prev_candle_close": round(prev_candle, 2),
                "rsi_dual": rsi_dual,
                "ema9": ema9_val, "ema21": ema21_val,
                "ema9_series": ema9_series, "ema21_series": ema21_series,
                "volume_ratio": vol_r, "atr": atr_val,
                "supertrend": st_data, "macd_div": macd_div,
                "bollinger": boll_data, "adx": adx_val,
                "pivots": pivots_data, "gap_pct": gap_pct,
                "df_1m": df_1m, "pcr": pcr_val,
                "closes_series": closes_1m,
                # V5.3 new indicators
                "stoch_rsi": stoch_rsi_data,
                "cmf": cmf_val,
                "ichimoku": ichimoku_data,
                "williams_r": williams_r_val,
                "roc": roc_val,
            }
        except Exception as e:
            self._fail_count += 1
            print(f"[ERROR] Fetch failed ({self._fail_count}): {e}")
            if self._fail_count >= 3:
                print("[WARN] 3 failures — rebuilding ticker (fallback)")
                self._rebuild_ticker()
            return None


# ══════════════════════════════════════════════
#  F2: SLIPPAGE & SPREAD MODEL
# ══════════════════════════════════════════════

# Nifty 50 large-cap symbols (subset for classification)
NIFTY50 = {
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL", "TATAPOWER", "GAIL",
    "MARUTI", "TMCV", "TMPV", "M&M", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO",
    "ASHOKLEY", "TVSMOTOR",
    "JSWSTEEL", "TATASTEEL", "HINDALCO", "JINDALSTEL", "SAIL",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "LUPIN", "ALKEM",
    "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "MARICO",
    "COLPAL", "GODREJCP",
    "ASIANPAINT", "BERGEPAINT", "KANSAINER",
    "PIDILITIND", "ASTRAL", "FINPIPE",
    "ULTRACEMCO", "AMBUJACEM", "SHREECEM", "DALBHARAT",
    "ABB", "SIEMENS", "CUMMINSIND", "HAVELLS", "POLYCAB", "KEI",
    "DIXON", "AMBER", "SYRMA",
    "DEEPAKNTR", "AARTIIND", "NAVINFLUOR", "SRF",
    "ADANIPORTS", "CONCOR"

}

class SlippageModel:
    """Estimates slippage by cap tier. Computes true edge after costs."""

    @staticmethod
    def get_slippage_pct(symbol: str, cfg: Dict) -> float:
        sl = cfg["slippage"]
        clean = symbol.replace(".NS", "").replace(".BO", "").upper()
        if clean in NIFTY50:
            return sl["large_cap_pct"]
        # Simple heuristic: if not in Nifty50, assume mid-cap
        return sl["mid_cap_pct"]

    @staticmethod
    def apply_slippage(price: float, direction: str, slippage_pct: float) -> float:
        """Adjust price for slippage (worse fill)."""
        if direction == "BUY":
            return round(price * (1 + slippage_pct / 100), 2)
        else:
            return round(price * (1 - slippage_pct / 100), 2)

    @staticmethod
    def compute_true_edge(
        entry: float, target: float, stop_loss: float,
        slippage_pct: float, direction: str,
        win_rate: float = 0.5,
    ) -> float:
        """
        True edge = expected value using actual historical win rate.
        EV = (win_rate × reward) - (loss_rate × risk) - round_trip_slippage
        With no history, defaults to 50% win rate.
        """
        total_cost_pct = slippage_pct * 2
        if direction == "BUY":
            reward_pct = ((target - entry) / entry) * 100
            risk_pct   = ((entry - stop_loss) / entry) * 100
        else:
            reward_pct = ((entry - target) / entry) * 100
            risk_pct   = ((stop_loss - entry) / entry) * 100

        wr = max(min(win_rate, 0.80), 0.35)   # clamp 35–80% to avoid extremes on small samples
        raw_edge = (wr * reward_pct) - ((1 - wr) * risk_pct)
        return round(raw_edge - total_cost_pct, 4)

    @staticmethod
    def get_true_edge_threshold(symbol: str, cfg: Dict) -> float:
        rm = cfg["risk_management"]
        clean = symbol.replace(".NS", "").replace(".BO", "").upper()
        if clean in NIFTY50:
            return rm["true_edge_min_pct"]
        return rm.get("true_edge_min_midcap", 0.15)

    @staticmethod
    def passes_edge_filter(true_edge: float, symbol: str, cfg: Dict) -> bool:
        return true_edge >= SlippageModel.get_true_edge_threshold(symbol, cfg)



# ══════════════════════════════════════════════
#  F13: PUT-CALL RATIO (PCR)
# ══════════════════════════════════════════════
import requests, json
from datetime import datetime, date

_pcr_cache = {"value": None, "fetched_at": None}

def fetch_pcr(symbol: str = "NIFTY") -> float:
    """
    Fetches Put-Call Ratio from NSE options chain.
    Free, official, updates every few minutes during market hours.
    Returns PCR float or 1.0 (neutral) on failure.
    """
    global _pcr_cache

    if _pcr_cache["fetched_at"]:
        age = (datetime.now() - _pcr_cache["fetched_at"]).seconds
        if age < 900 and _pcr_cache["value"]:
            return _pcr_cache["value"]

    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # Get cookies first
        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        headers["Accept"] = "application/json"
        headers["Referer"] = f"https://www.nseindia.com/get-quotes/derivatives?symbol={symbol}"
        
        resp = session.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return 1.0
            
        data = resp.json()
        if "records" not in data or "data" not in data["records"]:
            return 1.0
            
        total_ce_oi = sum(
            r["CE"]["openInterest"]
            for r in data["records"]["data"]
            if "CE" in r and r["CE"].get("openInterest")
        )
        total_pe_oi = sum(
            r["PE"]["openInterest"]
            for r in data["records"]["data"]
            if "PE" in r and r["PE"].get("openInterest")
        )

        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 1.0
        _pcr_cache = {"value": pcr, "fetched_at": datetime.now()}
        return pcr

    except Exception as e:
        return 1.0


# ══════════════════════════════════════════════
#  F7: NEWS FILTER — RSS
# ══════════════════════════════════════════════

class NewsFilter:
    """
    Checks NSE corporate announcements and MoneyControl RSS for
    high-impact events. Blocks trading if found.
    """

    _cache: Dict[str, Any] = {}
    _cache_time: Dict[str, float] = {}

    BLOCK_KEYWORDS = [
        "board meeting", "earnings", "quarterly results", "dividend",
        "bonus", "stock split", "buyback", "promoter", "sebi",
        "trading halt", "circuit", "insider trading", "stake",
        "acquisition", "merger", "demerger", "rights issue",
    ]

    RSS_FEEDS = [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://feeds.reuters.com/reuters/INbusinessNews",
    ]

    GLOBAL_BLOCK_KEYWORDS = [
        "federal reserve", "fed rate", "interest rate hike", "rate cut",
        "us recession", "trade war", "tariff", "crude oil shock",
        "rbi rate", "repo rate", "dollar surge", "rupee crash",
        "global sell-off", "market crash", "circuit breaker",
    ]

    @classmethod
    def is_safe_to_trade(cls, symbol: str, cfg: Dict) -> bool:
        if not cfg["news"]["enabled"]:
            return True

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper()
        cache_key = clean_symbol
        ttl = cfg["news"]["cache_ttl_minutes"] * 60

        # Check cache
        if cache_key in cls._cache_time:
            if _time.time() - cls._cache_time[cache_key] < ttl:
                return cls._cache.get(cache_key, True)

        # Fetch and check
        try:
            import feedparser
            is_safe = True
            for feed_url in cls.RSS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:40]:
                        title = entry.get("title", "").lower()
                        summary = entry.get("summary", "").lower()
                        text = title + " " + summary

                        # Symbol-specific block only — global macro is handled via score modifier
                        if clean_symbol.lower() in text:
                            for kw in cls.BLOCK_KEYWORDS:
                                if kw in text:
                                    is_safe = False
                                    break

                        if not is_safe:
                            break
                except Exception:
                    pass
                if not is_safe:
                    break

            cls._cache[cache_key] = is_safe
            cls._cache_time[cache_key] = _time.time()
            return is_safe

        except Exception:
            # Fail-safe: block if fetch fails
            if cfg["news"]["block_on_failure"]:
                return False
            return True

    @classmethod
    def status(cls, symbol: str, cfg: Dict) -> str:
        if not cfg["news"]["enabled"]:
            return "Disabled"
        clean = symbol.replace(".NS", "").replace(".BO", "").upper()
        if clean in cls._cache:
            return "✅ Clear" if cls._cache[clean] else "🚫 BLOCKED (news event)"
        return "Pending check"


# ══════════════════════════════════════════════
#  F11: TELEGRAM ALERTS
# ══════════════════════════════════════════════

class TelegramAlerter:
    """Sends alerts via Telegram Bot API with per-symbol throttle."""

    _last_alert: Dict[str, float] = {}

    @classmethod
    def send(cls, message: str, symbol: str, cfg: Dict):
        tc = cfg["telegram"]
        if not tc["enabled"] or not tc["bot_token"] or not tc["chat_id"]:
            return

        # Throttle: max 1 alert per symbol per N seconds
        key = symbol
        now = _time.time()
        if key in cls._last_alert:
            if now - cls._last_alert[key] < tc["throttle_seconds"]:
                return
        cls._last_alert[key] = now

        try:
            import urllib.request
            url = f"https://api.telegram.org/bot{tc['bot_token']}/sendMessage"
            payload = json.dumps({
                "chat_id": tc["chat_id"],
                "text": message,
                "parse_mode": "Markdown",
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[WARN] Telegram alert failed: {e}")

    @classmethod
    def format_signal(cls, data: Dict, result: Dict, regime: str) -> str:
        sig = result["signal"]
        return (
            f"🔔 *{sig}* — {data['symbol']}\n"
            f"Price: ₹{data['current_price']} | Score: {result['score']}/11\n"
            f"Entry: ₹{result['entry']} | SL: ₹{result['stop_loss']} | T: ₹{result['target']}\n"
            f"Regime: {regime} | Grade: {result['grade']}\n"
            f"Reason: {result['reason']}"
        )

    @classmethod
    def format_alert(cls, msg: str) -> str:
        return f"⚠️ *ALERT*\n{msg}"


# ══════════════════════════════════════════════
#  F12: TRADE LOGGER — Enhanced
# ══════════════════════════════════════════════

class TradeLogger:
    HEADERS = [
        "timestamp", "symbol", "price", "signal", "score", "grade",
        "market_regime", "regime_confidence",
        "rsi", "adx", "supertrend_dir", "atr", "macd_hist", "volume_ratio",
        "orb_signal", "slippage_pct", "true_edge",
        "position_size_units", "risk_amount",
        "entry", "stop_loss", "target", "reason",
        "news_status", "circuit_breaker_active", "tranche_id", "tranche_status"
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._trades_today: List[Dict] = []
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADERS)

    def log(self, data: Dict, result: Dict, extras: Dict):
        self._log_row(data, result, extras)

    def log_fill(self, symbol: str, fill: Dict, avg_entry: float, status: str):
        """Log tranche fills to a separate file — not the main trade log."""
        fill_log = self.filepath.replace("trade_log.csv", "tranche_fills.csv")
        write_header = not os.path.exists(fill_log)
        with open(fill_log, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "symbol", "tranche_id", "price", "qty", "avg_entry", "sl", "target", "status"])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, fill["id"], fill["price"], fill["qty"],
                avg_entry, fill["sl"], fill["target"], status,
            ])

    def _log_row(self, data: Dict, result: Dict, extras: Dict):
        if result["signal"] == "NO TRADE":
            return
        st_dir = data.get("supertrend", {}).get("dir_1min", "")
        macd_h = data.get("macd_div", "")
        rsi_val = f"{data.get('rsi_dual', {}).get('rsi7','')}/{data.get('rsi_dual', {}).get('rsi14','')}"
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["symbol"], data["current_price"],
            result["signal"], result["score"], result["grade"],
            extras.get("regime", ""), extras.get("regime_confidence", ""),
            rsi_val, data.get("adx", ""),
            st_dir, data.get("atr", ""), macd_h, data.get("volume_ratio", ""),
            extras.get("orb_signal", ""),
            extras.get("slippage_pct", ""), extras.get("true_edge", ""),
            extras.get("position_size", ""), extras.get("risk_amount", ""),
            result["entry"], result["stop_loss"], result["target"],
            result["reason"],
            extras.get("news_status", ""), extras.get("circuit_active", ""),
            "", "" # tranche_id, tranche_status
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._trades_today.append({
            "time": datetime.now().isoformat(),
            "symbol": data["symbol"],
            "signal": result["signal"],
            "entry": result["entry"],
            "score": result["score"],
        })
        print(f"  📝 Logged to {os.path.basename(self.filepath)}")

    def export_eod_json(self):
        """Export today's trades as JSON summary."""
        if not self._trades_today:
            return
        fp = self.filepath.replace(".csv", f"_eod_{datetime.now().strftime('%Y%m%d')}.json")
        with open(fp, "w") as f:
            json.dump({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_trades": len(self._trades_today),
                "trades": self._trades_today,
            }, f, indent=2)
        print(f"  📊 EOD summary: {os.path.basename(fp)}")


# ══════════════════════════════════════════════
#  SOUND ALERT
# ══════════════════════════════════════════════

def play_alert():
    try:
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════
#  V5.2 — SECTOR ROTATION FILTER
# ══════════════════════════════════════════════

SECTOR_INDEX_MAP = {
    "IT":      "^CNXIT",
    "BANK":    "^NSEBANK",
    "AUTO":    "^CNXAUTO",
    "PHARMA":  "^CNXPHARMA",
    "METAL":   "^CNXMETAL",
    "ENERGY":  "^CNXENERGY",
    "FMCG":    "^CNXFMCG",
    "REALTY":  "^CNXREALTY",
}

STOCK_SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT",
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK",
    "KOTAKBANK": "BANK", "AXISBANK": "BANK", "INDUSINDBK": "BANK",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO", "BAJAJ-AUTO": "AUTO",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "COALINDIA": "METAL", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "NTPC": "ENERGY", "TATAPOWER": "ENERGY",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "RELIANCE": "ENERGY", "ADANIPORTS": "ENERGY",
}

_sector_strength_cache = {}

def fetch_sector_strength() -> dict:
    """
    Calculates today's intraday return for each sector index.
    Returns sector -> strength label.
    """
    global _sector_strength_cache
    strengths = {}

    for sector, ticker in SECTOR_INDEX_MAP.items():
        try:
            import yfinance as yf
            data = yf.download(ticker, period="1d", interval="5m", progress=False)
            if data.empty:
                strengths[sector] = "NEUTRAL"
                continue
            open_col = data["Open"]
            if isinstance(open_col, pd.DataFrame):
                open_col = open_col.iloc[:, 0]
            close_col = data["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            open_price = float(open_col.iloc[0])
            last_price = float(close_col.iloc[-1])
            change_pct = (last_price - open_price) / open_price * 100

            if change_pct > 0.5:
                strengths[sector] = "STRONG_BULL"
            elif change_pct > 0.2:
                strengths[sector] = "MILD_BULL"
            elif change_pct < -0.5:
                strengths[sector] = "STRONG_BEAR"
            elif change_pct < -0.2:
                strengths[sector] = "MILD_BEAR"
            else:
                strengths[sector] = "NEUTRAL"
        except Exception:
            strengths[sector] = "NEUTRAL"

    _sector_strength_cache = strengths
    return strengths


def get_sector_score_bonus(symbol: str, strategy_type: str) -> float:
    """
    Returns a scoring bonus based on sector momentum alignment.
    """
    clean = symbol.replace(".NS", "").replace(".BO", "").upper()
    sector = STOCK_SECTOR_MAP.get(clean)
    if not sector:
        return 0.0

    strength = _sector_strength_cache.get(sector, "NEUTRAL")

    if strategy_type in ("ORB_LONG", "REVERSAL_LONG"):
        return {"STRONG_BULL": 0.5, "MILD_BULL": 0.25,
                "NEUTRAL": 0.0, "MILD_BEAR": -0.25,
                "STRONG_BEAR": -0.5}.get(strength, 0.0)

    elif strategy_type in ("ORB_SHORT", "REVERSAL_SHORT"):
        return {"STRONG_BEAR": 0.5, "MILD_BEAR": 0.25,
                "NEUTRAL": 0.0, "MILD_BULL": -0.25,
                "STRONG_BULL": -0.5}.get(strength, 0.0)

    return 0.0


# ══════════════════════════════════════════════
#  V5.2 — FII/DII FLOW FILTER
# ══════════════════════════════════════════════

_fii_dii_cache = {"data": None, "fetched_at": None}

def fetch_fii_dii_data() -> dict:
    """
    Fetches FII and DII provisional trading data from NSE.
    Returns net flows and directional bias.
    """
    global _fii_dii_cache

    if _fii_dii_cache["fetched_at"]:
        age = (datetime.now() - _fii_dii_cache["fetched_at"]).seconds
        if age < 3600 and _fii_dii_cache["data"]:
            return _fii_dii_cache["data"]

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=5)

        url  = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer":    "https://www.nseindia.com",
            "Accept":     "application/json",
        }, timeout=8)

        data = resp.json()
        if not data or not isinstance(data, list):
            raise ValueError("Invalid FII/DII response")

        today = data[0]
        fii_net = float(today.get("fii_index_stats", {}).get("netAmount", 0))
        dii_net = float(today.get("dii_index_stats", {}).get("netAmount", 0))
        combined = fii_net + dii_net

        if fii_net < -2000:
            bias = "BEARISH"
        elif fii_net > 2000:
            bias = "BULLISH"
        elif combined < -1000:
            bias = "BEARISH"
        elif combined > 1000:
            bias = "BULLISH"
        else:
            bias = "NEUTRAL"

        result = {"fii_net": fii_net, "dii_net": dii_net,
                  "combined": combined, "bias": bias}
        _fii_dii_cache = {"data": result, "fetched_at": datetime.now()}
        return result

    except Exception as e:
        print(f"[FII/DII] Fetch failed: {e} — neutral bias")
        return {"fii_net": 0, "dii_net": 0,
                "combined": 0, "bias": "NEUTRAL"}


# ══════════════════════════════════════════════
#  V5.2 — INDIA VIX ADAPTIVE SIZING
# ══════════════════════════════════════════════

_vix_cache = {"value": 15.0, "fetched_at": None}

def fetch_india_vix() -> float:
    """
    Fetches India VIX — 30-day implied volatility of Nifty options.
    """
    global _vix_cache

    if _vix_cache["fetched_at"]:
        age = (datetime.now() - _vix_cache["fetched_at"]).seconds
        if age < 1800 and _vix_cache["value"]:
            return _vix_cache["value"]

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        url  = "https://www.nseindia.com/api/allIndices"
        resp = session.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer":    "https://www.nseindia.com",
        }, timeout=8)
        indices = resp.json()["data"]
        vix_row = next((i for i in indices
                        if i.get("index") == "INDIA VIX"), None)
        vix = float(vix_row["last"]) if vix_row else 15.0
        _vix_cache = {"value": vix, "fetched_at": datetime.now()}
        return vix
    except Exception:
        return _vix_cache.get("value", 15.0)


_global_bias_cache = {"data": None, "fetched_at": None}

def fetch_global_market_bias() -> dict:
    """
    Fetches overnight US market and Dollar Index direction as a macro filter.
    Uses:
      - S&P 500 (^GSPC): overall risk sentiment
      - Nasdaq (^IXIC): tech/risk appetite
      - DXY (DX-Y.NYB): strong dollar = headwind for Indian markets
      - US VIX (^VIX): fear gauge — high VIX = risk-off globally

    Returns a bias label: BULLISH / BEARISH / NEUTRAL
    and a score modifier: +0.5 / -0.5 / 0.0
    Cached for 30 minutes (re-fetch on market open).
    """
    global _global_bias_cache

    if _global_bias_cache["fetched_at"]:
        age = (datetime.now() - _global_bias_cache["fetched_at"]).seconds
        if age < 1800 and _global_bias_cache["data"]:
            return _global_bias_cache["data"]

    try:
        result = {"bias": "NEUTRAL", "score_mod": 0.0, "spx_chg": 0.0,
                  "dxy_chg": 0.0, "us_vix": 20.0, "detail": ""}

        # S&P 500 — last 2 daily closes
        spx = yf.download("^GSPC", period="5d", interval="1d", progress=False)
        if not spx.empty and len(spx) >= 2:
            close_col = spx["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            spx_prev  = float(close_col.iloc[-2])
            spx_last  = float(close_col.iloc[-1])
            result["spx_chg"] = round((spx_last - spx_prev) / spx_prev * 100, 2)

        # DXY (US Dollar Index) — strong dollar is negative for INR/Indian stocks
        dxy = yf.download("DX-Y.NYB", period="5d", interval="1d", progress=False)
        if not dxy.empty and len(dxy) >= 2:
            close_col = dxy["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            dxy_prev  = float(close_col.iloc[-2])
            dxy_last  = float(close_col.iloc[-1])
            result["dxy_chg"] = round((dxy_last - dxy_prev) / dxy_prev * 100, 2)

        # US VIX
        uvix = yf.download("^VIX", period="2d", interval="1d", progress=False)
        if not uvix.empty:
            close_col = uvix["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            result["us_vix"] = round(float(close_col.iloc[-1]), 2)

        # Bias logic:
        # SPX up > 0.5% AND DXY neutral/down AND US VIX < 20 → BULLISH for India
        # SPX down > 0.5% OR DXY up > 0.3% OR US VIX > 25 → BEARISH for India
        spx_bull = result["spx_chg"] > 0.5
        spx_bear = result["spx_chg"] < -0.5
        dxy_bear = result["dxy_chg"] > 0.3   # strong dollar = negative for India
        vix_fear = result["us_vix"] > 25

        if spx_bull and not dxy_bear and not vix_fear:
            result["bias"] = "BULLISH"
            result["score_mod"] = 0.5
            result["detail"] = f"SPX+{result['spx_chg']}% DXY{result['dxy_chg']:+.2f}% UVIX{result['us_vix']}"
        elif spx_bear or dxy_bear or vix_fear:
            result["bias"] = "BEARISH"
            result["score_mod"] = -0.5
            result["detail"] = f"SPX{result['spx_chg']:+.2f}% DXY{result['dxy_chg']:+.2f}% UVIX{result['us_vix']}"
        else:
            result["bias"] = "NEUTRAL"
            result["score_mod"] = 0.0
            result["detail"] = f"SPX{result['spx_chg']:+.2f}% DXY{result['dxy_chg']:+.2f}% UVIX{result['us_vix']}"

        _global_bias_cache = {"data": result, "fetched_at": datetime.now()}
        return result

    except Exception as e:
        print(f"[GLOBAL BIAS] Fetch failed: {e}")
        return {"bias": "NEUTRAL", "score_mod": 0.0, "spx_chg": 0.0,
                "dxy_chg": 0.0, "us_vix": 20.0, "detail": "fetch_error"}


def get_vix_adjusted_params(vix: float) -> dict:
    """
    Returns ATR multiplier and size multiplier based on VIX regime.
    """
    if vix < 13:
        return {"sl_atr_mult": 1.0, "target_atr_mult": 1.8,
                "size_mult": 1.1,  "label": "LOW_VOL"}
    elif vix < 18:
        return {"sl_atr_mult": 1.2, "target_atr_mult": 2.0,
                "size_mult": 1.0,  "label": "NORMAL"}
    elif vix < 25:
        return {"sl_atr_mult": 1.6, "target_atr_mult": 2.5,
                "size_mult": 0.7,  "label": "ELEVATED"}
    else:
        return {"sl_atr_mult": 2.0, "target_atr_mult": 3.0,
                "size_mult": 0.5,  "label": "EXTREME"}
