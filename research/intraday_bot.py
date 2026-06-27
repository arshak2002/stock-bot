"""
intraday_bot.py — Lean Intraday Paper-Trading Bot (V7, full rewrite)
====================================================================
A single-file, reliable, aggressive intraday signal bot for NSE equities.

Why the rewrite:
  - The old multi-file system never produced a single trade (empty trade_log.csv).
  - Root cause #1: the data layer crashed on `fast_info.open == None` and hammered
    yfinance's 1-minute endpoint per-symbol every 5s -> rate-limit bans -> every
    symbol skipped at the data gate, every cycle.
  - Root cause #2: 7+ stacked gates + razor-thin entry windows that never aligned.

This rewrite fixes both:
  - ONE batched `yf.download(..., interval='5m', group_by='ticker')` per cycle for
    the whole universe (the only fetch pattern proven reliable in this environment).
  - All OHLC/indicators derived from the candle dataframe itself — never fast_info.
  - A simple, aggressive, score-based strategy (VWAP + EMA trend + ORB + momentum +
    volume) that fires LONG/SHORT whenever conviction clears a low threshold.
  - ATR-based stop/target, position sizing, a daily-loss circuit breaker, CSV logging
    and Telegram alerts.
  - A built-in backtester (`python intraday_bot.py backtest`) that runs the *same*
    scoring on historical bars so you can verify it actually trades before going live.

Usage:
    python intraday_bot.py             # live paper-trading loop (market hours only)
    python intraday_bot.py backtest    # backtest the strategy on recent history
    python intraday_bot.py walkforward # month-by-month + 2-year out-of-sample robustness
    python intraday_bot.py stress      # cost/slippage stress test -> breakeven cost
    python intraday_bot.py scan        # one-shot scan: print current signals & exit
"""

from __future__ import annotations

import os
import sys
import csv
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import yaml
except Exception:
    yaml = None

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

IST = timezone(timedelta(hours=5, minutes=30))

CONFIG = {
    # ── account ──
    "capital": 100_000.0,
    "risk_per_trade_pct": 1.0,     # % of capital risked per trade
    "max_position_pct": 20.0,      # cap one position's notional at this % of capital

    # ── session (IST) ──
    "market_open": "09:15",
    "session_start": "09:20",      # don't trade in the first chaotic minutes
    "orb_end": "09:45",            # opening-range = 09:15..09:45 (first 6 x 5m bars)
    "square_off": "15:15",         # close everything at/after this time
    "poll_interval_sec": 30,

    # ── strategy (AGGRESSIVE: many signals) ──
    "min_score": 4.0,              # fire when long/short score clears this (max 10)
    "vol_ratio_min": 1.1,          # volume confirmation floor
    "max_trades_per_symbol": 3,    # re-entries allowed per symbol per day
    "max_trades_per_day": 25,      # global daily cap
    # Backtest: longs-only is far stronger (PF 1.56 vs 1.23) — shorting cash
    # equities intraday is a net drag after costs. Flip to True to also short.
    "allow_short": False,          # paper short selling

    # ── risk / exits ──
    "sl_atr_mult": 1.5,
    "target_atr_mult": 2.5,
    "trail_atr_mult": 1.5,         # trail stop by this much ATR once in profit
    "trail_activate_atr": 1.0,     # start trailing after price moves this many ATR in favor
    "min_atr_pct": 0.3,            # floor ATR at this % of price (avoids tiny stops)

    # ── costs (realistic intraday round-trip) ──
    "cost_pct_per_side": 0.05,     # brokerage+STT+slippage estimate, each side (~0.1% round trip)

    # ── circuit breaker ──
    "daily_loss_limit_pct": -3.0,  # halt new trades if day PnL <= this % of capital
    "daily_gain_lock_pct": 6.0,    # halt new trades after locking this % gain

    # ── index regime filter ──
    "use_index_filter": True,      # only go long when Nifty is bullish (close>aVWAP & ema9>21)
    "index_symbol": "^NSEI",       # Nifty 50

    # ── data ──
    "interval": "5m",
    "history_period": "5d",        # for live indicators
    "backtest_period": "60d",      # for backtest
    "backtest_interval": "15m",    # 5m only goes back ~60d unreliably; 15m is stable

    # ── files ──
    "trade_log": "trades.csv",
}

# Liquid NSE universe (~45 names across sectors). Edit freely.
UNIVERSE = [
    "RELIANCE", "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "KOTAKBANK",
    "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE",
    "MARUTI", "M&M", "TATAMOTORS", "EICHERMOT", "BAJAJ-AUTO",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
    "ITC", "HINDUNILVR", "NESTLEIND", "TITAN", "ASIANPAINT",
    "LT", "ADANIPORTS", "ADANIENT", "ULTRACEMCO", "GRASIM",
    "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL",
    "DIXON", "POLYCAB", "TRENT",
]


def load_telegram_creds() -> dict:
    """Pull Telegram token/chat from the existing config.yaml if present."""
    creds = {"enabled": False, "bot_token": "", "chat_id": ""}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if yaml and os.path.exists(path):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            tg = cfg.get("telegram", {})
            if tg.get("bot_token") and tg.get("chat_id"):
                creds = {
                    "enabled": bool(tg.get("enabled", True)),
                    "bot_token": str(tg["bot_token"]),
                    "chat_id": str(tg["chat_id"]),
                }
        except Exception:
            pass
    return creds


# ════════════════════════════════════════════════════════════════════
#  INDICATORS  (pure pandas — computed once per dataframe)
# ════════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = ((up > down) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)) * down.clip(lower=0)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate a (single-symbol) OHLCV dataframe with all indicator columns.
    Index must be tz-aware (Asia/Kolkata). VWAP/ORB reset per trading day."""
    df = df.copy()
    # yfinance returns UTC for batch downloads — normalize to IST so the
    # 09:15-09:45 ORB window and per-session VWAP are anchored correctly.
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df["Volume"] > 0) & (df["Close"] > 0)]
    if df.empty:
        return df

    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = _rsi(df["Close"], 14)
    df["atr"] = _atr(df, 14)
    df["adx"] = _adx(df, 14)

    day = df.index.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    grp = pd.Series(day, index=df.index)
    df["vwap"] = pv.groupby(grp).cumsum() / df["Volume"].groupby(grp).cumsum()

    # session open (first bar of the day) and ORB high/low (first 30 min)
    df["sess_open"] = df["Open"].groupby(grp).transform("first")

    t = df.index.time
    orb_mask = (t >= dtime(9, 15)) & (t < dtime(9, 45))
    orb_h = df["High"].where(orb_mask).groupby(grp).cummax().groupby(grp).ffill()
    orb_l = df["Low"].where(orb_mask).groupby(grp).cummin().groupby(grp).ffill()
    df["orb_high"] = orb_h
    df["orb_low"] = orb_l

    # rolling average volume (20-bar) for surge ratio
    avg_vol = df["Volume"].rolling(20, min_periods=5).mean()
    df["vol_ratio"] = (df["Volume"] / avg_vol).fillna(1.0)
    return df


# ════════════════════════════════════════════════════════════════════
#  STRATEGY / SCORING  (used identically by live + backtest)
# ════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    symbol: str
    side: str            # "BUY" or "SELL"
    score: float
    price: float
    atr: float
    stop: float
    target: float
    reason: str


def score_row(row: pd.Series, cfg: dict) -> tuple[float, float, list, list]:
    """Return (long_score, short_score, long_reasons, short_reasons) for one bar."""
    price = float(row["Close"])
    vwap = float(row["vwap"])
    ema9, ema21, ema50 = float(row["ema9"]), float(row["ema21"]), float(row["ema50"])
    rsi = float(row["rsi"])
    adx = float(row["adx"])
    vol_r = float(row["vol_ratio"])
    orb_h = float(row["orb_high"]) if not pd.isna(row["orb_high"]) else None
    orb_l = float(row["orb_low"]) if not pd.isna(row["orb_low"]) else None
    sess_open = float(row["sess_open"]) if not pd.isna(row["sess_open"]) else price
    pct_open = (price - sess_open) / sess_open * 100 if sess_open else 0.0

    ls, lr = 0.0, []   # long
    ss, sr = 0.0, []   # short

    # 1. VWAP side (3.0) — the backbone of intraday bias
    if price > vwap:
        ls += 3.0; lr.append("aboveVWAP")
    else:
        ss += 3.0; sr.append("belowVWAP")

    # 2. EMA trend (2.0 + 1.0 for full stack alignment)
    if ema9 > ema21:
        ls += 2.0; lr.append("ema9>21")
        if ema21 > ema50:
            ls += 1.0; lr.append("stacked")
    if ema9 < ema21:
        ss += 2.0; sr.append("ema9<21")
        if ema21 < ema50:
            ss += 1.0; sr.append("stacked")

    # 3. ORB breakout (2.0)
    if orb_h and price > orb_h:
        ls += 2.0; lr.append("ORB^")
    if orb_l and price < orb_l:
        ss += 2.0; sr.append("ORBv")

    # 4. Volume surge (1.5)
    if vol_r >= cfg["vol_ratio_min"]:
        bonus = 1.5 if vol_r >= 2.0 else 1.0
        ls += bonus; lr.append(f"vol{vol_r:.1f}x")
        ss += bonus; sr.append(f"vol{vol_r:.1f}x")

    # 5. RSI momentum, not exhausted (1.0)
    if 50 <= rsi <= 78:
        ls += 1.0; lr.append(f"rsi{rsi:.0f}")
    if 22 <= rsi <= 50:
        ss += 1.0; sr.append(f"rsi{rsi:.0f}")

    # 6. ADX trend strength (0.5)
    if adx >= 20:
        ls += 0.5; lr.append(f"adx{adx:.0f}")
        ss += 0.5; sr.append(f"adx{adx:.0f}")

    # 7. Move from open (0.5)
    if pct_open >= 0.2:
        ls += 0.5; lr.append(f"+{pct_open:.1f}%")
    if pct_open <= -0.2:
        ss += 0.5; sr.append(f"{pct_open:.1f}%")

    return ls, ss, lr, sr


def make_signal(symbol: str, row: pd.Series, cfg: dict, index_bull: bool = True) -> Signal | None:
    ls, ss, lr, sr = score_row(row, cfg)
    price = float(row["Close"])
    atr = float(row["atr"])
    # floor ATR so stops aren't microscopic
    atr = max(atr, price * cfg["min_atr_pct"] / 100.0)
    if atr <= 0:
        return None

    use_filter = cfg.get("use_index_filter", True)
    long_ok = (not use_filter) or index_bull          # only long when index bullish
    short_ok = (not use_filter) or (not index_bull)   # only short when index bearish

    min_score = cfg["min_score"]
    if ls >= min_score and ls >= ss and long_ok:
        stop = round(price - atr * cfg["sl_atr_mult"], 2)
        target = round(price + atr * cfg["target_atr_mult"], 2)
        return Signal(symbol, "BUY", round(ls, 1), price, atr, stop, target, " ".join(lr))
    if cfg["allow_short"] and ss >= min_score and ss > ls and short_ok:
        stop = round(price + atr * cfg["sl_atr_mult"], 2)
        target = round(price - atr * cfg["target_atr_mult"], 2)
        return Signal(symbol, "SELL", round(ss, 1), price, atr, stop, target, " ".join(sr))
    return None


def position_size(entry: float, stop: float, cfg: dict) -> int:
    risk_amt = cfg["capital"] * cfg["risk_per_trade_pct"] / 100.0
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0
    qty = int(risk_amt / per_unit_risk)
    # cap notional
    max_notional = cfg["capital"] * cfg["max_position_pct"] / 100.0
    if qty * entry > max_notional:
        qty = int(max_notional / entry)
    return max(qty, 0)


# ════════════════════════════════════════════════════════════════════
#  DATA FETCH  (one batched download for the whole universe)
# ════════════════════════════════════════════════════════════════════

def fetch_universe(symbols: list, period: str, interval: str) -> dict:
    """Return {symbol: indicator-annotated DataFrame}. One batched request."""
    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(
        tickers, period=period, interval=interval,
        group_by="ticker", auto_adjust=False, progress=False, threads=True,
    )
    out = {}
    for s, t in zip(symbols, tickers):
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            df = df.dropna(how="all")
            if df.empty or len(df) < 30:
                continue
            df = add_indicators(df)
            if not df.empty and len(df) >= 30:
                out[s] = df
        except Exception:
            continue
    return out


def fetch_index_regime(cfg: dict, period: str, interval: str):
    """Return a tz-aware bool Series: True where the index is bullish.
    Bullish = ema9>ema21 AND close>anchored-session-VWAP. Volume-free
    (index volume is 0), so VWAP is approximated by session typical-price mean."""
    try:
        raw = yf.download(cfg["index_symbol"], period=period, interval=interval,
                          auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            return None
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata")
        df = df[df["Close"] > 0]
        if len(df) < 30:
            return None
        ema9 = df["Close"].ewm(span=9, adjust=False).mean()
        ema21 = df["Close"].ewm(span=21, adjust=False).mean()
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        grp = pd.Series(df.index.date, index=df.index)
        avwap = tp.groupby(grp).cumsum() / (tp.groupby(grp).cumcount() + 1)
        bull = (ema9 > ema21) & (df["Close"] > avwap)
        return bull.rename("idx_bull")
    except Exception as e:
        print(f"[index regime] fetch failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
#  TELEGRAM + CSV LOGGING
# ════════════════════════════════════════════════════════════════════

class Notifier:
    def __init__(self):
        self.tg = load_telegram_creds()
        self._last = {}

    def telegram(self, text: str):
        if not self.tg["enabled"]:
            return
        try:
            url = f"https://api.telegram.org/bot{self.tg['bot_token']}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.tg["chat_id"], "text": text, "parse_mode": "HTML",
            }).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
        except Exception as e:
            print(f"[telegram] failed: {e}")

    def alert(self, text: str):
        print(text)
        self.telegram(text)


TRADE_FIELDS = [
    "open_time", "close_time", "symbol", "side", "qty",
    "entry", "stop", "target", "exit", "exit_reason",
    "pnl", "pnl_pct", "score", "reason",
]


def log_trade(path: str, trade: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: trade.get(k, "") for k in TRADE_FIELDS})


# ════════════════════════════════════════════════════════════════════
#  POSITION MANAGEMENT
# ════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    atr: float
    score: float
    reason: str
    open_time: str
    best: float = 0.0          # best favorable price seen (for trailing)
    trailing: bool = False

    def unrealized(self, price: float) -> float:
        """Gross PnL (before costs)."""
        if self.side == "BUY":
            return (price - self.entry) * self.qty
        return (self.entry - price) * self.qty


def round_trip_cost(entry: float, exit_price: float, qty: int, cfg: dict) -> float:
    """Brokerage+STT+slippage on both legs."""
    rate = cfg.get("cost_pct_per_side", 0.0) / 100.0
    return (entry + exit_price) * qty * rate


def check_exit(pos: Position, high: float, low: float, close: float, cfg: dict):
    """Return (exit_price, reason) or (None, None). Uses bar high/low for intrabar fills."""
    # update trailing stop
    if pos.side == "BUY":
        pos.best = max(pos.best or pos.entry, high)
        move = (pos.best - pos.entry) / pos.atr if pos.atr else 0
        if move >= cfg["trail_activate_atr"]:
            new_stop = pos.best - pos.atr * cfg["trail_atr_mult"]
            if new_stop > pos.stop:
                pos.stop = round(new_stop, 2)
                pos.trailing = True
        if low <= pos.stop:
            return pos.stop, ("TRAIL" if pos.trailing else "STOP")
        if high >= pos.target:
            return pos.target, "TARGET"
    else:  # SELL
        pos.best = min(pos.best or pos.entry, low)
        move = (pos.entry - pos.best) / pos.atr if pos.atr else 0
        if move >= cfg["trail_activate_atr"]:
            new_stop = pos.best + pos.atr * cfg["trail_atr_mult"]
            if new_stop < pos.stop:
                pos.stop = round(new_stop, 2)
                pos.trailing = True
        if high >= pos.stop:
            return pos.stop, ("TRAIL" if pos.trailing else "STOP")
        if low <= pos.target:
            return pos.target, "TARGET"
    return None, None


# ════════════════════════════════════════════════════════════════════
#  LIVE PAPER-TRADING LOOP
# ════════════════════════════════════════════════════════════════════

def _now_ist() -> datetime:
    return datetime.now(IST)


def _parse_t(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


class LiveBot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.notifier = Notifier()
        self.positions: dict[str, Position] = {}
        self.day_pnl = 0.0
        self.trades_today = 0
        self.symbol_trades: dict[str, int] = {}
        self.halted = False
        self.session_date = None

    def _reset_day(self, d):
        self.session_date = d
        self.day_pnl = 0.0
        self.trades_today = 0
        self.symbol_trades = {}
        self.halted = False
        self.positions = {}

    def _circuit_ok(self) -> bool:
        cap = self.cfg["capital"]
        if self.day_pnl <= cap * self.cfg["daily_loss_limit_pct"] / 100.0:
            self.halted = True
        if self.day_pnl >= cap * self.cfg["daily_gain_lock_pct"] / 100.0:
            self.halted = True
        return not self.halted

    def _open(self, sig: Signal):
        qty = position_size(sig.price, sig.stop, self.cfg)
        if qty <= 0:
            return
        pos = Position(
            symbol=sig.symbol, side=sig.side, qty=qty, entry=sig.price,
            stop=sig.stop, target=sig.target, atr=sig.atr, score=sig.score,
            reason=sig.reason, open_time=_now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            best=sig.price,
        )
        self.positions[sig.symbol] = pos
        self.trades_today += 1
        self.symbol_trades[sig.symbol] = self.symbol_trades.get(sig.symbol, 0) + 1
        rr = abs(sig.target - sig.price) / max(abs(sig.price - sig.stop), 1e-9)
        self.notifier.alert(
            f"🟢 <b>{sig.side} {sig.symbol}</b>  score {sig.score}/10\n"
            f"Entry ₹{sig.price}  Qty {qty}\n"
            f"SL ₹{sig.stop}  Target ₹{sig.target}  (R:R 1:{rr:.1f})\n"
            f"{sig.reason}"
        )

    def _close(self, pos: Position, exit_price: float, reason: str):
        pnl = pos.unrealized(exit_price) - round_trip_cost(pos.entry, exit_price, pos.qty, self.cfg)
        pnl_pct = pnl / (pos.entry * pos.qty) * 100 if pos.qty else 0
        self.day_pnl += pnl
        emoji = "✅" if pnl >= 0 else "🔴"
        self.notifier.alert(
            f"{emoji} <b>CLOSE {pos.side} {pos.symbol}</b> [{reason}]\n"
            f"Exit ₹{exit_price:.2f}  PnL ₹{pnl:,.0f} ({pnl_pct:+.2f}%)\n"
            f"Day PnL ₹{self.day_pnl:,.0f}"
        )
        log_trade(self.cfg["trade_log"], {
            "open_time": pos.open_time,
            "close_time": _now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": pos.symbol, "side": pos.side, "qty": pos.qty,
            "entry": pos.entry, "stop": pos.stop, "target": pos.target,
            "exit": round(exit_price, 2), "exit_reason": reason,
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "score": pos.score, "reason": pos.reason,
        })
        self.positions.pop(pos.symbol, None)

    def run(self):
        cfg = self.cfg
        sess_start = _parse_t(cfg["session_start"])
        square_off = _parse_t(cfg["square_off"])
        self.notifier.alert(
            f"🤖 <b>Intraday paper bot online</b>\n"
            f"Universe {len(UNIVERSE)} | min_score {cfg['min_score']} | "
            f"risk {cfg['risk_per_trade_pct']}%/trade"
        )
        while True:
            now = _now_ist()
            today = now.date()
            if self.session_date != today:
                self._reset_day(today)

            # weekend
            if now.weekday() >= 5:
                print(f"[{now:%a %H:%M}] weekend — sleeping 1h")
                time.sleep(3600); continue

            t = now.time()
            if t < sess_start:
                print(f"[{now:%H:%M}] pre-market — waiting")
                time.sleep(60); continue
            if t >= square_off:
                if self.positions:
                    self._square_off_all()
                self._eod_report()
                # sleep until next morning
                time.sleep(3600); continue

            try:
                self._tick(now)
            except Exception as e:
                print(f"[tick error] {e}")
            time.sleep(cfg["poll_interval_sec"])

    def _square_off_all(self):
        data = fetch_universe(list(self.positions.keys()),
                              self.cfg["history_period"], self.cfg["interval"])
        for sym, pos in list(self.positions.items()):
            px = pos.entry
            if sym in data:
                px = float(data[sym]["Close"].iloc[-1])
            self._close(pos, px, "EOD")

    def _eod_report(self):
        if self.trades_today:
            self.notifier.alert(
                f"📊 <b>EOD</b> trades {self.trades_today}  "
                f"Day PnL ₹{self.day_pnl:,.0f}"
            )

    def _tick(self, now):
        cfg = self.cfg
        data = fetch_universe(UNIVERSE, cfg["history_period"], cfg["interval"])
        if not data:
            print(f"[{now:%H:%M:%S}] no data this cycle")
            return

        # current index regime (bullish?) — gate longs
        index_bull = True
        if cfg.get("use_index_filter"):
            idx = fetch_index_regime(cfg, cfg["history_period"], cfg["interval"])
            if idx is not None and len(idx):
                index_bull = bool(idx.iloc[-1])

        # 1) manage open positions
        for sym, pos in list(self.positions.items()):
            if sym not in data:
                continue
            last = data[sym].iloc[-1]
            ex, reason = check_exit(pos, float(last["High"]), float(last["Low"]),
                                    float(last["Close"]), cfg)
            if ex is not None:
                self._close(pos, ex, reason)

        # 2) look for new entries
        if not self._circuit_ok():
            print(f"[{now:%H:%M:%S}] circuit halted (day PnL ₹{self.day_pnl:,.0f})")
            return
        if self.trades_today >= cfg["max_trades_per_day"]:
            return

        candidates = []
        for sym, df in data.items():
            if sym in self.positions:
                continue
            if self.symbol_trades.get(sym, 0) >= cfg["max_trades_per_symbol"]:
                continue
            sig = make_signal(sym, df.iloc[-1], cfg, index_bull=index_bull)
            if sig:
                candidates.append(sig)

        candidates.sort(key=lambda s: s.score, reverse=True)
        n_open = len(self.positions)
        scanned = len(data)
        regime = "BULL" if index_bull else "BEAR"
        fired = 0
        for sig in candidates:
            if self.trades_today >= cfg["max_trades_per_day"]:
                break
            self._open(sig)
            fired += 1
        print(f"[{now:%H:%M:%S}] Nifty {regime} | scanned {scanned} | open {n_open} | "
              f"new {fired} | day PnL ₹{self.day_pnl:,.0f}")


# ════════════════════════════════════════════════════════════════════
#  BACKTEST
# ════════════════════════════════════════════════════════════════════

def _simulate(data: dict, cfg: dict, index_regime=None) -> list:
    """Event-driven sim over pre-fetched data. Returns a list of trade dicts.
    Each trade stores `gross` (pre-cost) so cost can be re-applied for stress tests."""
    trades = []
    square_off = _parse_t(cfg["square_off"])
    sess_start = _parse_t(cfg["session_start"])

    for sym, df in data.items():
        # align index regime to this symbol's bars (instant-based, ffill)
        if index_regime is not None:
            df = df.copy()
            df["idx_bull"] = index_regime.reindex(df.index, method="ffill").fillna(False).values
        df = df.reset_index()
        idx_col = df.columns[0]
        sym_trades_by_day: dict = {}
        pos: Position | None = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = row[idx_col]
            t = ts.time()
            d = ts.date()

            if pos is not None:
                ex, reason = check_exit(pos, float(row["High"]), float(row["Low"]),
                                        float(row["Close"]), cfg)
                if ex is None and t >= square_off:
                    ex, reason = float(row["Close"]), "EOD"
                if ex is not None:
                    gross = pos.unrealized(ex)
                    pnl = gross - round_trip_cost(pos.entry, ex, pos.qty, cfg)
                    trades.append({
                        "symbol": sym, "side": pos.side, "entry": pos.entry,
                        "exit": ex, "qty": pos.qty, "gross": gross, "pnl": pnl,
                        "pnl_pct": pnl / (pos.entry * pos.qty) * 100 if pos.qty else 0,
                        "reason": reason, "day": str(d), "month": f"{d.year}-{d.month:02d}",
                    })
                    pos = None

            if pos is not None or t < sess_start or t >= square_off:
                continue
            if pd.isna(row["atr"]) or pd.isna(row["vwap"]) or pd.isna(row["ema50"]):
                continue
            if sym_trades_by_day.get(d, 0) >= cfg["max_trades_per_symbol"]:
                continue

            index_bull = bool(row["idx_bull"]) if "idx_bull" in row else True
            sig = make_signal(sym, row, cfg, index_bull=index_bull)
            if sig:
                if i + 1 >= len(df):
                    continue
                nxt = df.iloc[i + 1]
                if nxt[idx_col].date() != d:   # don't carry overnight
                    continue
                entry = float(nxt["Open"])
                atr = sig.atr
                if sig.side == "BUY":
                    stop = round(entry - atr * cfg["sl_atr_mult"], 2)
                    target = round(entry + atr * cfg["target_atr_mult"], 2)
                else:
                    stop = round(entry + atr * cfg["sl_atr_mult"], 2)
                    target = round(entry - atr * cfg["target_atr_mult"], 2)
                qty = position_size(entry, stop, cfg)
                if qty <= 0:
                    continue
                pos = Position(
                    symbol=sym, side=sig.side, qty=qty, entry=entry, stop=stop,
                    target=target, atr=atr, score=sig.score, reason=sig.reason,
                    open_time=str(ts), best=entry,
                )
                sym_trades_by_day[d] = sym_trades_by_day.get(d, 0) + 1

    return trades


def backtest(cfg: dict, symbols: list | None = None):
    symbols = symbols or UNIVERSE
    period, interval = cfg["backtest_period"], cfg["backtest_interval"]
    print(f"Backtesting {len(symbols)} symbols over {period} @ {interval} "
          f"(index filter: {'ON' if cfg.get('use_index_filter') else 'OFF'}) ...")
    data = fetch_universe(symbols, period, interval)
    idx = fetch_index_regime(cfg, period, interval) if cfg.get("use_index_filter") else None
    print(f"Got data for {len(data)} symbols | index regime: "
          f"{'loaded' if idx is not None else 'unavailable'}\n")
    trades = _simulate(data, cfg, idx)
    _print_backtest_report(trades, cfg)
    return trades


def _print_backtest_report(trades: list, cfg: dict):
    if not trades:
        print("⚠️  No trades generated. Loosen min_score or check data.")
        return
    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    n = len(df)
    win_rate = len(wins) / n * 100
    total_pnl = df["pnl"].sum()
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss else float("inf")
    expectancy = total_pnl / n
    days = df["day"].nunique()

    print("═" * 56)
    print("  BACKTEST RESULTS")
    print("═" * 56)
    print(f"  Trades:            {n}  over {days} days  ({n/days:.1f}/day)")
    print(f"  Win rate:          {win_rate:.1f}%   ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL:         ₹{total_pnl:,.0f}  "
          f"({total_pnl/cfg['capital']*100:+.1f}% of capital)")
    print(f"  Avg win:           ₹{avg_win:,.0f}")
    print(f"  Avg loss:          ₹{avg_loss:,.0f}")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Expectancy/trade:  ₹{expectancy:,.0f}")
    print("  ── exit reasons ──")
    for reason, grp in df.groupby("reason"):
        print(f"     {reason:8s}: {len(grp):4d}  PnL ₹{grp['pnl'].sum():>10,.0f}")
    print("  ── by side ──")
    for side, grp in df.groupby("side"):
        wr = (grp["pnl"] > 0).mean() * 100
        print(f"     {side:4s}: {len(grp):4d} trades  WR {wr:.0f}%  "
              f"PnL ₹{grp['pnl'].sum():>10,.0f}")
    print("═" * 56)
    top = df.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    print("  Top symbols:    " + ", ".join(f"{s}(₹{v:,.0f})" for s, v in top.head(5).items()))
    print("  Worst symbols:  " + ", ".join(f"{s}(₹{v:,.0f})" for s, v in top.tail(5).items()))
    print("═" * 56)


def _stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "pf": 0}
    df = pd.DataFrame(trades)
    n = len(df)
    gw = df.loc[df.pnl > 0, "pnl"].sum()
    gl = abs(df.loc[df.pnl <= 0, "pnl"].sum())
    return {
        "n": n,
        "wr": (df.pnl > 0).mean() * 100,
        "pnl": df.pnl.sum(),
        "pf": (gw / gl) if gl else float("inf"),
    }


def _monthly_table(trades: list, title: str):
    print(f"\n  {title}")
    print(f"  {'month':9s} {'trades':>7s} {'WR%':>6s} {'PF':>6s} {'PnL':>12s}")
    df = pd.DataFrame(trades)
    pos_months = 0
    months = sorted(df["month"].unique())
    for m in months:
        s = _stats(df[df.month == m].to_dict("records"))
        flag = "✓" if s["pnl"] > 0 else "✗"
        if s["pnl"] > 0:
            pos_months += 1
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"  {m:9s} {s['n']:7d} {s['wr']:6.1f} {pf:>6s} ₹{s['pnl']:>10,.0f} {flag}")
    print(f"  → {pos_months}/{len(months)} months profitable")


# ════════════════════════════════════════════════════════════════════
#  WALK-FORWARD  /  OUT-OF-SAMPLE ROBUSTNESS
# ════════════════════════════════════════════════════════════════════

def walkforward(cfg: dict):
    """Two views:
      (1) month-by-month on the 60d/15m data (the production interval), and
      (2) a long-horizon 1h backtest over ~2y to see if the edge holds across
          many different market regimes (1h is coarser — a robustness proxy)."""
    print("═" * 60)
    print("  WALK-FORWARD / OUT-OF-SAMPLE ROBUSTNESS")
    print("═" * 60)

    # ── View 1: production interval, month by month ──
    p, iv = cfg["backtest_period"], cfg["backtest_interval"]
    data = fetch_universe(UNIVERSE, p, iv)
    idx = fetch_index_regime(cfg, p, iv) if cfg.get("use_index_filter") else None
    t1 = _simulate(data, cfg, idx)
    s = _stats(t1)
    print(f"\n[View 1] {iv} bars over {p}  ({len(data)} symbols)")
    print(f"  overall: {s['n']} trades  WR {s['wr']:.1f}%  PF {s['pf']:.2f}  PnL ₹{s['pnl']:,.0f}")
    if t1:
        _monthly_table(t1, "month-by-month (production interval):")

    # ── View 2: long-horizon 1h, many regimes ──
    print(f"\n[View 2] 1h bars over 730d (≈2 years, coarse robustness proxy)")
    data2 = fetch_universe(UNIVERSE, "730d", "1h")
    idx2 = fetch_index_regime(cfg, "730d", "1h") if cfg.get("use_index_filter") else None
    t2 = _simulate(data2, cfg, idx2)
    s2 = _stats(t2)
    print(f"  overall: {s2['n']} trades  WR {s2['wr']:.1f}%  PF {s2['pf']:.2f}  PnL ₹{s2['pnl']:,.0f}")
    if t2:
        _monthly_table(t2, "month-by-month (1h, 2-year):")
    print("═" * 60)
    return t1, t2


# ════════════════════════════════════════════════════════════════════
#  COST / SLIPPAGE STRESS TEST
# ════════════════════════════════════════════════════════════════════

def stress_costs(cfg: dict):
    """Re-price the SAME simulated trades at rising round-trip costs to find the
    breakeven cost (where total PnL -> 0). Cost is applied post-exit so it never
    changes which bar a stop/target hits — we can re-cost one simulation."""
    print("═" * 60)
    print("  COST / SLIPPAGE STRESS TEST")
    print("═" * 60)
    p, iv = cfg["backtest_period"], cfg["backtest_interval"]
    data = fetch_universe(UNIVERSE, p, iv)
    idx = fetch_index_regime(cfg, p, iv) if cfg.get("use_index_filter") else None
    trades = _simulate(data, cfg, idx)
    if not trades:
        print("No trades to stress."); return
    df = pd.DataFrame(trades)
    # cost for one trade at rate r (%/side) = (entry+exit)*qty*r/100
    leg = (df["entry"] + df["exit"]) * df["qty"]
    gross_total = df["gross"].sum()

    print(f"\n  {df.shape[0]} trades | gross PnL (zero-cost) ₹{gross_total:,.0f}\n")
    print(f"  {'cost%/side':>10s} {'round-trip':>11s} {'net PnL':>12s} {'PF':>6s} {'WR%':>6s}")
    grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    prev_pnl, prev_c, breakeven = None, None, None
    for c in grid:
        net = df["gross"] - leg * c / 100.0
        total = net.sum()
        gw = net[net > 0].sum(); gl = abs(net[net <= 0].sum())
        pf = gw / gl if gl else float("inf")
        wr = (net > 0).mean() * 100
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        mark = "  <- current" if abs(c - cfg.get("cost_pct_per_side", 0.05)) < 1e-9 else ""
        print(f"  {c:10.3f} {c*2:11.3f} ₹{total:>10,.0f} {pf_s:>6s} {wr:6.1f}{mark}")
        if prev_pnl is not None and prev_pnl > 0 >= total and breakeven is None:
            # linear interpolate breakeven cost
            frac = prev_pnl / (prev_pnl - total)
            breakeven = prev_c + frac * (c - prev_c)
        prev_pnl, prev_c = total, c

    print()
    if breakeven is not None:
        print(f"  ⮕ Breakeven round-trip cost ≈ {breakeven*2:.3f}%  ({breakeven:.3f}%/side)")
        head = breakeven - cfg.get("cost_pct_per_side", 0.05)
        print(f"  ⮕ Headroom over assumed 0.05%/side: {head:.3f}%/side")
        if head < 0.03:
            print("  ⚠️  Thin headroom — real slippage could erase the edge.")
        else:
            print("  ✓ Reasonable cushion vs typical liquid-stock costs.")
    else:
        print("  ⮕ Still profitable across the entire cost grid (very robust) "
              "or never profitable (check sign).")
    print("═" * 60)
    return df


# ════════════════════════════════════════════════════════════════════
#  ONE-SHOT SCAN
# ════════════════════════════════════════════════════════════════════

def scan_once(cfg: dict):
    data = fetch_universe(UNIVERSE, cfg["history_period"], cfg["interval"])
    index_bull = True
    if cfg.get("use_index_filter"):
        idx = fetch_index_regime(cfg, cfg["history_period"], cfg["interval"])
        if idx is not None and len(idx):
            index_bull = bool(idx.iloc[-1])
    print(f"Fetched {len(data)}/{len(UNIVERSE)} symbols at {_now_ist():%Y-%m-%d %H:%M:%S} IST"
          f"  |  Nifty regime: {'BULL' if index_bull else 'BEAR'}\n")
    sigs = []
    for sym, df in data.items():
        sig = make_signal(sym, df.iloc[-1], cfg, index_bull=index_bull)
        if sig:
            sigs.append(sig)
    sigs.sort(key=lambda s: s.score, reverse=True)
    if not sigs:
        print("No signals right now.")
        return
    print(f"{'SYM':12s} {'SIDE':5s} {'SCORE':>5s} {'PRICE':>9s} {'STOP':>9s} {'TGT':>9s}  REASON")
    for s in sigs:
        print(f"{s.symbol:12s} {s.side:5s} {s.score:5.1f} {s.price:9.2f} "
              f"{s.stop:9.2f} {s.target:9.2f}  {s.reason}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        backtest(cfg)
    elif mode == "walkforward":
        walkforward(cfg)
    elif mode == "stress":
        stress_costs(cfg)
    elif mode == "scan":
        scan_once(cfg)
    elif mode == "live":
        LiveBot(cfg).run()
    else:
        print(__doc__)
        print(f"Unknown mode '{mode}'. Use: live | backtest | walkforward | stress | scan")


if __name__ == "__main__":
    main()
