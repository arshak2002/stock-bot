"""
swing_bot.py — Daily-bar Swing Strategy + Honest Walk-Forward
=============================================================
The intraday bot was proven to have NO edge (0/36 months over 2 years) — the
friction of intraday costs + ATR stops whipsawed by noise beats every direction.

This tests the next hypothesis: a DAILY-bar swing strategy (multi-day holds),
where transaction cost is a tiny fraction of the move and trend/mean-reversion
have documented persistence. Held to the same honest bar: walk-forward across
~10 years, one fold per calendar year. No single-period cherry-picking.

Two documented edges are tested:
  1. TREND  — Donchian breakout above a long uptrend (SMA200), ride with a
              chandelier (ATR) trailing stop. Classic trend-following.
  2. MEANREV — Connors RSI(2): in an uptrend (>SMA200), buy a deep oversold
              dip (RSI2<10), exit when it snaps back (close>SMA5) or stalls.

Usage:
    python swing_bot.py            # compare both strategies + walk-forward
    python swing_bot.py trend      # detailed report for trend strategy
    python swing_bot.py meanrev    # detailed report for mean-reversion
"""
from __future__ import annotations
import sys, warnings
from dataclasses import dataclass
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

CONFIG = {
    "capital": 100_000.0,
    "risk_per_trade_pct": 1.0,
    "max_position_pct": 25.0,
    "cost_pct_per_side": 0.05,     # ~0.1% round trip (daily swing: negligible vs move)
    "history_period": "10y",
    # trend params
    "donchian": 20,                # breakout lookback (prior N-day high)
    "trend_sma": 200,
    "chandelier_atr": 3.0,         # trailing stop distance
    "init_atr_stop": 2.5,
    # meanrev params
    "rsi_buy": 10.0,
    "rsi_exit": 70.0,
    "mr_max_hold": 10,
    "mr_atr_stop": 3.0,
}

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


# ── indicators ──
def _rsi(close, n):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_indicators(df, cfg):
    df = df.copy()
    df = df[df["Close"] > 0]
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(cfg["trend_sma"]).mean()
    df["sma5"] = df["Close"].rolling(5).mean()
    df["rsi2"] = _rsi(df["Close"], 2)
    df["atr"] = _atr(df, 14)
    df["don_high"] = df["High"].rolling(cfg["donchian"]).max().shift(1)  # prior N-day high
    return df


def fetch_daily(symbols, period):
    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, period=period, interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    for s, t in zip(symbols, tickers):
        try:
            df = raw[t].dropna(how="all") if len(tickers) > 1 else raw.copy()
            if len(df) < 250:
                continue
            out[s] = df
        except Exception:
            continue
    return out


def round_trip_cost(entry, exit_, qty, cfg):
    return (entry + exit_) * qty * cfg["cost_pct_per_side"] / 100.0


def position_size(entry, stop, cfg):
    risk = cfg["capital"] * cfg["risk_per_trade_pct"] / 100.0
    per = abs(entry - stop)
    if per <= 0:
        return 0
    qty = int(risk / per)
    cap = cfg["capital"] * cfg["max_position_pct"] / 100.0
    if qty * entry > cap:
        qty = int(cap / entry)
    return max(qty, 0)


@dataclass
class Pos:
    side: str; qty: int; entry: float; stop: float; atr: float
    bars: int = 0; best: float = 0.0


# ── strategy sims (long-only; both are long strategies) ──
def sim_trend(data, cfg):
    trades = []
    for sym, df in data.items():
        df = add_indicators(df, cfg).reset_index()
        tcol = df.columns[0]
        pos = None
        for i in range(1, len(df) - 1):
            row, nxt = df.iloc[i], df.iloc[i + 1]
            if pos is not None:
                pos.bars += 1
                pos.best = max(pos.best, float(row["High"]))
                chand = pos.best - pos.atr * cfg["chandelier_atr"]
                pos.stop = max(pos.stop, chand)
                ex = None; reason = ""
                if float(row["Low"]) <= pos.stop:
                    ex, reason = pos.stop, "STOP/TRAIL"
                elif float(row["Close"]) < float(row["sma50"]):
                    ex, reason = float(nxt["Open"]), "TRENDBREAK"
                if ex is not None:
                    g = (ex - pos.entry) * pos.qty
                    pnl = g - round_trip_cost(pos.entry, ex, pos.qty, cfg)
                    trades.append({"symbol": sym, "pnl": pnl, "gross": g, "bars": pos.bars,
                                   "reason": reason, "year": str(df.iloc[i][tcol].year)})
                    pos = None
            if pos is not None:
                continue
            if pd.isna(row["sma200"]) or pd.isna(row["atr"]) or pd.isna(row["don_high"]):
                continue
            # entry: uptrend + breakout of prior 20d high
            if float(row["Close"]) > float(row["sma200"]) and float(row["Close"]) > float(row["don_high"]):
                entry = float(nxt["Open"])
                stop = entry - float(row["atr"]) * cfg["init_atr_stop"]
                qty = position_size(entry, stop, cfg)
                if qty > 0:
                    pos = Pos("BUY", qty, entry, stop, float(row["atr"]), best=entry)
    return trades


def sim_meanrev(data, cfg):
    trades = []
    for sym, df in data.items():
        df = add_indicators(df, cfg).reset_index()
        tcol = df.columns[0]
        pos = None
        for i in range(1, len(df) - 1):
            row, nxt = df.iloc[i], df.iloc[i + 1]
            if pos is not None:
                pos.bars += 1
                ex = None; reason = ""
                if float(row["Low"]) <= pos.stop:
                    ex, reason = pos.stop, "STOP"
                elif float(row["Close"]) > float(row["sma5"]) or float(row["rsi2"]) > cfg["rsi_exit"]:
                    ex, reason = float(nxt["Open"]), "SNAPBACK"
                elif pos.bars >= cfg["mr_max_hold"]:
                    ex, reason = float(nxt["Open"]), "TIME"
                if ex is not None:
                    g = (ex - pos.entry) * pos.qty
                    pnl = g - round_trip_cost(pos.entry, ex, pos.qty, cfg)
                    trades.append({"symbol": sym, "pnl": pnl, "gross": g, "bars": pos.bars,
                                   "reason": reason, "year": str(df.iloc[i][tcol].year)})
                    pos = None
            if pos is not None:
                continue
            if pd.isna(row["sma200"]) or pd.isna(row["atr"]):
                continue
            # entry: uptrend + deep oversold
            if float(row["Close"]) > float(row["sma200"]) and float(row["rsi2"]) < cfg["rsi_buy"]:
                entry = float(nxt["Open"])
                stop = entry - float(row["atr"]) * cfg["mr_atr_stop"]
                qty = position_size(entry, stop, cfg)
                if qty > 0:
                    pos = Pos("BUY", qty, entry, stop, float(row["atr"]), best=entry)
    return trades


# ── reporting ──
def _stats(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "pf": 0, "avg_bars": 0}
    d = pd.DataFrame(trades)
    gw = d.loc[d.pnl > 0, "pnl"].sum(); gl = abs(d.loc[d.pnl <= 0, "pnl"].sum())
    return {"n": len(d), "wr": (d.pnl > 0).mean() * 100, "pnl": d.pnl.sum(),
            "pf": (gw / gl) if gl else float("inf"), "avg_bars": d["bars"].mean()}


def report(trades, label):
    s = _stats(trades)
    if not trades:
        print(f"\n{label}: NO TRADES"); return
    d = pd.DataFrame(trades)
    print("═" * 60)
    print(f"  {label}")
    print("═" * 60)
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    print(f"  Trades {s['n']}  | WR {s['wr']:.1f}%  | PF {pf}  | "
          f"PnL ₹{s['pnl']:,.0f} ({s['pnl']/CONFIG['capital']*100:+.0f}% of capital)")
    print(f"  Avg hold {s['avg_bars']:.1f} days  | "
          f"avg win ₹{d.loc[d.pnl>0,'pnl'].mean():,.0f}  avg loss ₹{d.loc[d.pnl<=0,'pnl'].mean():,.0f}")
    print("  ── walk-forward by year ──")
    print(f"  {'year':6s} {'trades':>7s} {'WR%':>6s} {'PF':>6s} {'PnL':>12s}")
    pos_years = 0; years = sorted(d["year"].unique())
    for y in years:
        sy = _stats(d[d.year == y].to_dict("records"))
        flag = "✓" if sy["pnl"] > 0 else "✗"
        if sy["pnl"] > 0: pos_years += 1
        pfy = "inf" if sy["pf"] == float("inf") else f"{sy['pf']:.2f}"
        print(f"  {y:6s} {sy['n']:7d} {sy['wr']:6.1f} {pfy:>6s} ₹{sy['pnl']:>10,.0f} {flag}")
    print(f"  → {pos_years}/{len(years)} years profitable")
    print("  ── exit reasons ──")
    for r, g in d.groupby("reason"):
        print(f"     {r:12s}: {len(g):4d}  PnL ₹{g['pnl'].sum():>10,.0f}")
    print("═" * 60)


def portfolio_meanrev(data, cfg, max_concurrent=8):
    """Realistic portfolio sim: shared capital, capped concurrent positions,
    next-open fills (no lookahead), daily equity curve -> CAGR + max drawdown.
    Tests the correlated-entry / capital-constraint reality RSI2 faces."""
    # pre-index each symbol by date
    prepped = {}
    all_dates = set()
    for sym, df in data.items():
        d = add_indicators(df, cfg)
        d.index = pd.to_datetime(d.index).normalize()
        prepped[sym] = d
        all_dates.update(d.index)
    dates = sorted(all_dates)

    equity = cfg["capital"]
    cash = equity
    positions = {}           # sym -> Pos
    pending_entries = []     # list of sym to enter at next open
    pending_exits = []       # list of (sym, reason)
    trades = []
    eq_curve = []

    def row_at(sym, day):
        d = prepped[sym]
        if day in d.index:
            return d.loc[day]
        return None

    for day in dates:
        # 1) execute pending exits at today's open
        for sym, reason in pending_exits:
            if sym not in positions:
                continue
            r = row_at(sym, day)
            if r is None:
                continue
            px = float(r["Open"])
            p = positions.pop(sym)
            g = (px - p.entry) * p.qty
            pnl = g - round_trip_cost(p.entry, px, p.qty, cfg)
            cash += p.qty * px - round_trip_cost(p.entry, px, p.qty, cfg)
            trades.append({"symbol": sym, "pnl": pnl, "gross": g, "bars": p.bars,
                           "reason": reason, "year": str(day.year)})
        pending_exits = []

        # 2) execute pending entries at today's open (cap concurrent + cash)
        for sym in pending_entries:
            if len(positions) >= max_concurrent or sym in positions:
                continue
            r = row_at(sym, day)
            if r is None or pd.isna(r["atr"]):
                continue
            entry = float(r["Open"])
            stop = entry - float(r["atr"]) * cfg["mr_atr_stop"]
            qty = position_size(entry, stop, cfg)
            qty = min(qty, int(cash / entry)) if entry > 0 else 0
            if qty > 0:
                cash -= qty * entry
                positions[sym] = Pos("BUY", qty, entry, stop, float(r["atr"]), best=entry)
        pending_entries = []

        # 3) intraday stop check on today's bar
        for sym in list(positions.keys()):
            r = row_at(sym, day)
            if r is None:
                continue
            p = positions[sym]
            p.bars += 1
            if float(r["Low"]) <= p.stop:
                px = p.stop
                positions.pop(sym)
                g = (px - p.entry) * p.qty
                pnl = g - round_trip_cost(p.entry, px, p.qty, cfg)
                cash += p.qty * px - round_trip_cost(p.entry, px, p.qty, cfg)
                trades.append({"symbol": sym, "pnl": pnl, "gross": g, "bars": p.bars,
                               "reason": "STOP", "year": str(day.year)})

        # 4) evaluate signals on today's close -> pending for next open
        held_val = 0.0
        signals = []
        for sym, d in prepped.items():
            if day not in d.index:
                continue
            r = d.loc[day]
            if sym in positions:
                held_val += positions[sym].qty * float(r["Close"])
                # exit conditions
                if float(r["Close"]) > float(r["sma5"]) or float(r["rsi2"]) > cfg["rsi_exit"] \
                        or positions[sym].bars >= cfg["mr_max_hold"]:
                    pending_exits.append((sym, "SNAPBACK"))
            else:
                if not pd.isna(r["sma200"]) and not pd.isna(r["atr"]) \
                        and float(r["Close"]) > float(r["sma200"]) and float(r["rsi2"]) < cfg["rsi_buy"]:
                    signals.append((sym, float(r["rsi2"])))
        # rank deepest-oversold first, queue up to free slots
        free = max_concurrent - len(positions)
        for sym, _ in sorted(signals, key=lambda x: x[1])[:max(free, 0)]:
            pending_entries.append(sym)

        equity = cash + held_val
        eq_curve.append((day, equity))

    # metrics
    ec = pd.Series({d: v for d, v in eq_curve})
    ret = ec.iloc[-1] / cfg["capital"]
    years = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = (ret ** (1 / years) - 1) * 100 if years > 0 and ret > 0 else -100
    peak = ec.cummax()
    dd = ((ec - peak) / peak).min() * 100
    daily_ret = ec.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() else 0
    return trades, {"final": ec.iloc[-1], "cagr": cagr, "maxdd": dd,
                    "sharpe": sharpe, "years": years, "ec": ec}


def report_portfolio(cfg, data, max_concurrent=8):
    trades, m = portfolio_meanrev(data, cfg, max_concurrent)
    s = _stats(trades)
    print("═" * 60)
    print(f"  PORTFOLIO MEANREV — realistic (max {max_concurrent} concurrent, shared capital)")
    print("═" * 60)
    print(f"  Final equity ₹{m['final']:,.0f} from ₹{cfg['capital']:,.0f}  "
          f"over {m['years']:.1f}y")
    print(f"  CAGR {m['cagr']:.1f}%  |  Max drawdown {m['maxdd']:.1f}%  |  "
          f"Sharpe {m['sharpe']:.2f}")
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    print(f"  Trades {s['n']}  WR {s['wr']:.1f}%  PF {pf}  avg hold {s['avg_bars']:.1f}d")
    d = pd.DataFrame(trades)
    print("  ── by year ──")
    pos = 0; yrs = sorted(d["year"].unique())
    for y in yrs:
        sy = _stats(d[d.year == y].to_dict("records"))
        f = "✓" if sy["pnl"] > 0 else "✗"
        if sy["pnl"] > 0: pos += 1
        print(f"     {y}: {sy['n']:4d} tr  WR {sy['wr']:4.0f}%  PnL ₹{sy['pnl']:>9,.0f} {f}")
    print(f"  → {pos}/{len(yrs)} years profitable")
    print("═" * 60)


def stress_meanrev(cfg, data):
    print("\n  COST STRESS (per-symbol meanrev, gross re-priced):")
    trades = sim_meanrev(data, cfg)
    d = pd.DataFrame(trades)
    leg = None
    # recompute net from gross needs entry/exit; approximate via cost already in pnl.
    # Re-run sim at each cost level (cheap enough).
    print(f"  {'cost%/side':>10s} {'PnL':>12s} {'PF':>6s}")
    import copy
    for c in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        cc = copy.deepcopy(cfg); cc["cost_pct_per_side"] = c
        t = sim_meanrev(data, cc)
        s = _stats(t)
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"  {c:10.3f} ₹{s['pnl']:>10,.0f} {pf:>6s}")


def main():
    cfg = CONFIG
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"Fetching {len(UNIVERSE)} symbols, {cfg['history_period']} daily ...")
    data = fetch_daily(UNIVERSE, cfg["history_period"])
    print(f"Got {len(data)} symbols\n")
    if mode in ("trend", "both"):
        report(sim_trend(data, cfg), "TREND  (Donchian breakout > SMA200, chandelier trail)")
    if mode in ("meanrev", "both"):
        report(sim_meanrev(data, cfg), "MEANREV (Connors RSI2 dip in uptrend)")
    if mode in ("portfolio", "both"):
        report_portfolio(cfg, data, max_concurrent=8)
        stress_meanrev(cfg, data)


if __name__ == "__main__":
    main()
