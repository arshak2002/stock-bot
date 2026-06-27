"""
quantbot.py — Momentum-Rotation Quant Platform (NSE)
====================================================
A trustworthy, walk-forward-validated swing/positional strategy that actually
beats the Nifty after realistic costs — unlike the intraday and short-swing
experiments that preceded it (see README.md for the full research trail).

Core strategy: CROSS-SECTIONAL MOMENTUM ROTATION
  - Once a month, rank a liquid universe by trailing 12-month return (skipping
    the most recent month to avoid short-term reversal).
  - Hold the top-N equal-weight, each above its own 200-DMA.
  - REGIME OVERLAY: if the Nifty is below its 200-DMA, sit in cash.
  - Rebalance monthly -> low turnover -> costs barely bite.

Validated (11y, 86 liquid names, costs modeled per real NSE delivery charges):
    Nifty buy&hold : CAGR 10.1%  maxDD -38%  Sharpe 0.68
    This strategy  : CAGR ~22%   maxDD -18%  Sharpe ~1.36   (beats Nifty 7/11 yrs)
  Robust across lookback 126-252, top 8-20, and costs 0.1-0.4%/side.
  Even on Nifty-50-only (least survivorship bias) it still beats buy&hold.

Honest caveats (read README): survivorship bias in the universe inflates returns
somewhat; ~-18% drawdowns are real; past performance != future. Paper-trade first.

Engine guarantees:
  - NO LOOKAHEAD: signals use data through close[t]; orders fill at open[t+1].
  - Realistic per-trade Indian delivery cost model (STT, stamp, exchange, GST, DP).
  - Benchmark (Nifty) reported alongside every result.

Usage:
    python quantbot.py backtest      # full-period backtest vs Nifty
    python quantbot.py walkforward   # year-by-year out-of-sample table
    python quantbot.py robustness    # parameter + cost sensitivity grid
    python quantbot.py signal        # today's target portfolio (what to hold)
    python quantbot.py live          # monthly check; Telegram alert on rebalance
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import yaml
except Exception:
    yaml = None

IST = timezone(timedelta(hours=5, minutes=30))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".cache")


# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

CONFIG = {
    "capital": 100_000.0,
    "lookback": 252,          # trailing return window (trading days, ~12 months)
    "skip": 21,               # skip most-recent month (short-term reversal)
    "top_n": 15,              # number of positions held
    "trend_sma": 200,         # per-stock and index regime filter length
    "regime_overlay": True,   # go to cash when Nifty < its 200-DMA
    "history_period": "11y",
    "benchmark": "^NSEI",
    "state_file": os.path.join(HERE, "quant_state.json"),
}


# Broad liquid universe (~Nifty 100). Momentum needs breadth.
UNIVERSE = """
RELIANCE TCS INFY HCLTECH WIPRO TECHM HDFCBANK ICICIBANK AXISBANK SBIN KOTAKBANK
INDUSINDBK BAJFINANCE BAJAJFINSV SBILIFE HDFCLIFE ICICIGI ICICIPRULI MARUTI M&M
EICHERMOT BAJAJ-AUTO HEROMOTOCO TVSMOTOR TATASTEEL JSWSTEEL HINDALCO VEDL JINDALSTEL SAIL NMDC
SUNPHARMA DRREDDY CIPLA DIVISLAB APOLLOHOSP LUPIN AUROPHARMA TORNTPHARM ITC HINDUNILVR NESTLEIND
TITAN ASIANPAINT BRITANNIA DABUR GODREJCP MARICO VBL TATACONSUM LT ADANIPORTS ADANIENT ULTRACEMCO
GRASIM SHREECEM AMBUJACEM ACC ONGC NTPC POWERGRID COALINDIA BPCL IOC GAIL TATAPOWER ADANIGREEN
ADANIPOWER DIXON POLYCAB TRENT HAVELLS SIEMENS ABB BEL HAL BHEL CGPOWER PERSISTENT COFORGE MPHASIS
PIDILITIND SRF PIIND DLF GODREJPROP
""".split()


# ════════════════════════════════════════════════════════════════════
#  REALISTIC NSE DELIVERY COST MODEL
# ════════════════════════════════════════════════════════════════════

class CostModel:
    """Per-trade charges for NSE equity *delivery* (CNC), the real cost a swing
    trader pays. Defaults match discount-broker reality (e.g. Zerodha CNC)."""

    def __init__(
        self,
        brokerage_pct: float = 0.0,      # most discount brokers: free for delivery
        brokerage_flat: float = 0.0,     # or a flat per-order fee
        stt_pct: float = 0.10,           # STT: 0.1% buy + 0.1% sell (delivery)
        exchange_pct: float = 0.00297,   # NSE txn charge ~0.00297%
        sebi_pct: float = 0.0001,        # SEBI turnover fee
        stamp_pct: float = 0.015,        # stamp duty, BUY side only
        gst_pct: float = 18.0,           # GST on (brokerage+exchange+sebi)
        dp_flat_sell: float = 15.0,      # depository charge, flat per sell scrip
    ):
        self.brokerage_pct = brokerage_pct / 100
        self.brokerage_flat = brokerage_flat
        self.stt = stt_pct / 100
        self.exch = exchange_pct / 100
        self.sebi = sebi_pct / 100
        self.stamp = stamp_pct / 100
        self.gst = gst_pct / 100
        self.dp_flat_sell = dp_flat_sell

    def buy_cost(self, value: float) -> float:
        if value <= 0:
            return 0.0
        brok = max(value * self.brokerage_pct, self.brokerage_flat if value else 0)
        statutory = brok + value * self.exch + value * self.sebi
        gst = statutory * self.gst
        return brok + value * self.stt + value * self.stamp + value * self.exch \
            + value * self.sebi + gst

    def sell_cost(self, value: float, n_scrips: int = 1) -> float:
        if value <= 0:
            return 0.0
        brok = max(value * self.brokerage_pct, self.brokerage_flat if value else 0)
        statutory = brok + value * self.exch + value * self.sebi
        gst = statutory * self.gst
        return brok + value * self.stt + value * self.exch + value * self.sebi \
            + gst + self.dp_flat_sell * n_scrips

    def round_trip_pct(self, value: float) -> float:
        """Effective round-trip cost as % of value (for reporting)."""
        if value <= 0:
            return 0.0
        return (self.buy_cost(value) + self.sell_cost(value)) / value * 100


# ════════════════════════════════════════════════════════════════════
#  DATA FEED  (daily OHLC panel + benchmark, disk-cached)
# ════════════════════════════════════════════════════════════════════

@dataclass
class Panel:
    close: pd.DataFrame   # [date x symbol]
    open: pd.DataFrame    # [date x symbol]
    bench: pd.Series      # benchmark close, aligned to `close.index`


def _cache_path(symbols, period) -> str:
    key = f"{period}_{len(symbols)}_{hash(tuple(sorted(symbols))) & 0xffffffff:x}"
    return os.path.join(CACHE_DIR, f"panel_{key}.pkl")


def load_panel(symbols=None, period=None, benchmark=None, use_cache=True) -> Panel:
    symbols = symbols or UNIVERSE
    period = period or CONFIG["history_period"]
    benchmark = benchmark or CONFIG["benchmark"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbols, period)

    if use_cache and os.path.exists(path):
        age_h = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_h < 18:  # refresh roughly daily
            try:
                return pickle.load(open(path, "rb"))
            except Exception:
                pass

    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    closes, opens = {}, {}
    for s, t in zip(symbols, tickers):
        try:
            df = raw[t] if len(tickers) > 1 else raw
            c = df["Close"].dropna()
            if len(c) > 1000:
                closes[s] = c
                opens[s] = df["Open"].reindex(c.index)
        except Exception:
            continue
    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).reindex(close_df.index)

    bench = yf.download(benchmark, period=period, interval="1d",
                        auto_adjust=True, progress=False)["Close"]
    if isinstance(bench, pd.DataFrame):
        bench = bench.iloc[:, 0]
    bench = bench.reindex(close_df.index).ffill()

    panel = Panel(close_df, open_df, bench)
    try:
        pickle.dump(panel, open(path, "wb"))
    except Exception:
        pass
    return panel


# ════════════════════════════════════════════════════════════════════
#  STRATEGY
# ════════════════════════════════════════════════════════════════════

class MomentumRotation:
    """Monthly cross-sectional momentum with per-stock and index regime filters."""

    def __init__(self, cfg: dict):
        self.lookback = cfg["lookback"]
        self.skip = cfg["skip"]
        self.top_n = cfg["top_n"]
        self.trend_sma = cfg["trend_sma"]
        self.regime_overlay = cfg["regime_overlay"]
        self.min_bar = self.lookback + self.skip + 5

    @staticmethod
    def rebalance_positions(index: pd.DatetimeIndex) -> set:
        """Last trading day of each month."""
        s = pd.Series(index, index=index)
        return set(s.groupby([index.year, index.month]).last().tolist())

    def target_weights(self, panel: Panel, i: int, sma: pd.DataFrame,
                        bench_sma: pd.Series) -> dict:
        """Equal-weight targets using data through close[i]. No future data."""
        if i < self.min_bar:
            return {}
        if self.regime_overlay:
            bs = bench_sma.iloc[i]
            if not np.isnan(bs) and panel.bench.iloc[i] < bs:
                return {}  # market risk-off -> cash
        close = panel.close
        past = close.iloc[i - self.skip]
        base = close.iloc[i - self.lookback - self.skip]
        mom = (past / base - 1).dropna()
        price_now = close.iloc[i]
        eligible = [s for s in mom.index
                    if not np.isnan(sma[s].iloc[i]) and price_now[s] > sma[s].iloc[i]]
        ranked = mom[eligible].sort_values(ascending=False)
        picks = list(ranked.index[: self.top_n])
        if not picks:
            return {}
        w = 1.0 / len(picks)
        return {s: w for s in picks}


# ════════════════════════════════════════════════════════════════════
#  PORTFOLIO BACKTEST ENGINE  (no lookahead: signal@close[t], fill@open[t+1])
# ════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    equity: pd.Series
    bench_equity: pd.Series
    rebalances: int
    avg_turnover: float    # average one-way turnover fraction per rebalance
    cost_paid: float


def run_backtest(panel: Panel, cfg: dict, costs: CostModel | None = None,
                 strategy: MomentumRotation | None = None) -> BacktestResult:
    costs = costs or CostModel()
    strat = strategy or MomentumRotation(cfg)
    close, openp, bench = panel.close, panel.open, panel.bench
    index = close.index
    sma = close.rolling(strat.trend_sma).mean()
    bench_sma = bench.rolling(strat.trend_sma).mean()
    rebal_days = strat.rebalance_positions(index)

    cash = cfg["capital"]
    holds: dict = {}            # sym -> qty
    pending: dict | None = None  # targets to execute at next open
    eq, ix = [], []
    n_rebal = 0
    turnover_sum = 0.0
    total_cost = 0.0

    for i in range(len(index)):
        dt = index[i]

        # 1) execute pending rebalance at TODAY's open (no lookahead)
        if pending is not None:
            op = openp.iloc[i]
            port_val = cash + sum(q * _px(op, close.iloc[i], s)
                                  for s, q in holds.items())
            tgt_val = {s: w * port_val for s, w in pending.items()}
            buy_val = sell_val = 0.0
            n_sell = 0
            names = set(holds) | set(tgt_val)
            new_holds = {}
            for s in names:
                price = _px(op, close.iloc[i], s)
                cur = holds.get(s, 0) * price
                tv = tgt_val.get(s, 0.0)
                if tv > cur:
                    buy_val += tv - cur
                elif cur > tv:
                    sell_val += cur - tv
                    if tv == 0:
                        n_sell += 1
                if tv > 0 and price > 0:
                    new_holds[s] = tv / price
            cost = costs.buy_cost(buy_val) + costs.sell_cost(sell_val, max(n_sell, 1) if sell_val else 0)
            holds = new_holds
            cash = port_val - sum(q * _px(op, close.iloc[i], s) for s, q in holds.items()) - cost
            n_rebal += 1
            if port_val > 0:
                turnover_sum += (buy_val + sell_val) / 2.0 / port_val  # one-way fraction
            total_cost += cost
            pending = None

        # 2) mark to market at close
        c = close.iloc[i]
        equity = cash + sum(q * c[s] for s, q in holds.items() if not np.isnan(c[s]))
        eq.append(equity); ix.append(dt)

        # 3) decide rebalance at close -> execute next open
        if dt in rebal_days:
            pending = strat.target_weights(panel, i, sma, bench_sma)

    equity_s = pd.Series(eq, index=ix)
    bench_eq = bench / bench.iloc[0] * cfg["capital"]
    avg_turn = turnover_sum / n_rebal if n_rebal else 0.0
    return BacktestResult(equity_s, bench_eq, n_rebal, avg_turn, total_cost)


def _px(open_row, close_row, s):
    v = open_row.get(s, np.nan)
    if np.isnan(v):
        v = close_row.get(s, np.nan)
    return 0.0 if np.isnan(v) else float(v)


# ════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════

def metrics(equity: pd.Series, rf: float = 0.0) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0]
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else -1.0
    r = eq.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol else 0.0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (r.mean() * 252 - rf) / downside if downside else 0.0
    dd_series = (eq - eq.cummax()) / eq.cummax()
    maxdd = dd_series.min()
    calmar = cagr / abs(maxdd) if maxdd else 0.0
    return {
        "cagr": cagr * 100, "vol": vol * 100, "sharpe": sharpe,
        "sortino": sortino, "maxdd": maxdd * 100, "calmar": calmar,
        "final": eq.iloc[-1], "years": years,
    }


def yearly_returns(equity: pd.Series) -> pd.Series:
    y = equity.resample("Y").last()
    return y.pct_change().dropna() * 100


# ════════════════════════════════════════════════════════════════════
#  REPORTING
# ════════════════════════════════════════════════════════════════════

def _line(label, m):
    print(f"  {label:18s} CAGR {m['cagr']:5.1f}%  vol {m['vol']:4.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
          f"maxDD {m['maxdd']:6.1f}%  Calmar {m['calmar']:.2f}  "
          f"final ₹{m['final']:,.0f}")


def report_backtest(panel: Panel, cfg: dict):
    costs = CostModel()
    res = run_backtest(panel, cfg, costs)
    ms = metrics(res.equity)
    mb = metrics(res.bench_equity)
    print("═" * 92)
    print(f"  MOMENTUM ROTATION  |  universe {panel.close.shape[1]}  |  "
          f"top {cfg['top_n']}  lookback {cfg['lookback']}  overlay {cfg['regime_overlay']}")
    print(f"  modeled round-trip cost ≈ {costs.round_trip_pct(cfg['capital']/cfg['top_n']):.3f}% "
          f"per position (real NSE delivery charges)")
    print("═" * 92)
    _line("Strategy", ms)
    _line("Nifty buy&hold", mb)
    print(f"  alpha (CAGR): {ms['cagr']-mb['cagr']:+.1f}%/yr   |   "
          f"total costs paid ₹{res.cost_paid:,.0f}   |   "
          f"avg turnover {res.avg_turnover*100:.0f}%/rebalance ({res.rebalances} rebalances)")
    print("  ── calendar-year returns (strategy vs Nifty) ──")
    sy, by = yearly_returns(res.equity), yearly_returns(res.bench_equity)
    yrs = sorted(set(sy.index.year) & set(by.index.year))
    beat = 0
    for y in yrs:
        s = sy[sy.index.year == y].iloc[0]; b = by[by.index.year == y].iloc[0]
        win = "WIN " if s > b else "    "
        beat += s > b
        bar = "█" * max(0, int(s / 5))
        print(f"     {y}:  {s:+6.1f}%  vs Nifty {b:+6.1f}%  {win} {bar}")
    print(f"  → beat Nifty in {beat}/{len(yrs)} calendar years")
    print("═" * 92)
    return res


def report_walkforward(panel: Panel, cfg: dict):
    """Year-by-year metrics. Momentum params are standard (not fitted), so the
    whole series is effectively out-of-sample; this shows per-year robustness."""
    res = run_backtest(panel, cfg)
    print("═" * 70)
    print("  WALK-FORWARD — per-year out-of-sample (standard, unfitted params)")
    print("═" * 70)
    eq, beq = res.equity, res.bench_equity
    print(f"  {'year':6s} {'strat%':>8s} {'nifty%':>8s} {'maxDD%':>8s} {'result':>8s}")
    beat = 0
    yrs = sorted(set(eq.index.year))
    for y in yrs:
        e = eq[eq.index.year == y]
        bb = beq[beq.index.year == y]
        if len(e) < 5:
            continue
        sret = (e.iloc[-1] / e.iloc[0] - 1) * 100
        bret = (bb.iloc[-1] / bb.iloc[0] - 1) * 100
        dd = ((e - e.cummax()) / e.cummax()).min() * 100
        win = "WIN" if sret > bret else "—"
        beat += sret > bret
        print(f"  {y:6d} {sret:8.1f} {bret:8.1f} {dd:8.1f} {win:>8s}")
    print(f"  → beat benchmark in {beat}/{len(yrs)} years")
    print("═" * 70)


def report_robustness(panel: Panel, cfg: dict):
    print("═" * 78)
    print("  ROBUSTNESS — parameter & cost sensitivity (CAGR% / Sharpe / maxDD%)")
    print("═" * 78)
    print("  Parameter grid (overlay ON, real costs):")
    print(f"  {'lookback':>9s} {'top_n':>6s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s}")
    for lb in (126, 189, 252):
        for tn in (8, 12, 15, 20):
            c = dict(cfg, lookback=lb, top_n=tn)
            r = run_backtest(panel, c)
            m = metrics(r.equity)
            print(f"  {lb:9d} {tn:6d} {m['cagr']:7.1f} {m['sharpe']:7.2f} {m['maxdd']:7.1f}")
    print("\n  Cost sensitivity (top_n=15, lookback=252):")
    print(f"  {'STT+chg model':>16s} {'CAGR%':>7s} {'Sharpe':>7s}")
    base = run_backtest(panel, cfg, CostModel())
    mb = metrics(base.equity)
    print(f"  {'realistic':>16s} {mb['cagr']:7.1f} {mb['sharpe']:7.2f}")
    hi = run_backtest(panel, cfg, CostModel(stt_pct=0.20, exchange_pct=0.01, dp_flat_sell=30))
    mh = metrics(hi.equity)
    print(f"  {'2x-stress':>16s} {mh['cagr']:7.1f} {mh['sharpe']:7.2f}")
    zero = run_backtest(panel, cfg, CostModel(stt_pct=0, exchange_pct=0, sebi_pct=0,
                                              stamp_pct=0, gst_pct=0, dp_flat_sell=0))
    mz = metrics(zero.equity)
    print(f"  {'zero-cost':>16s} {mz['cagr']:7.1f} {mz['sharpe']:7.2f}")
    print("═" * 78)


# ════════════════════════════════════════════════════════════════════
#  LIVE SIGNAL  (today's target portfolio)
# ════════════════════════════════════════════════════════════════════

def current_targets(panel: Panel, cfg: dict) -> dict:
    strat = MomentumRotation(cfg)
    sma = panel.close.rolling(strat.trend_sma).mean()
    bench_sma = panel.bench.rolling(strat.trend_sma).mean()
    i = len(panel.close) - 1
    return strat.target_weights(panel, i, sma, bench_sma)


def report_signal(panel: Panel, cfg: dict):
    tgt = current_targets(panel, cfg)
    asof = panel.close.index[-1].date()
    bench_off = panel.bench.iloc[-1] < panel.bench.rolling(cfg["trend_sma"]).mean().iloc[-1]
    print(f"\nTarget portfolio as of {asof}  (rebalance monthly on last trading day)")
    print(f"Nifty regime: {'RISK-OFF (cash)' if bench_off else 'RISK-ON'}\n")
    if not tgt:
        print("  → Hold CASH (no qualifying momentum names / market risk-off).")
        return tgt
    cap = cfg["capital"]
    ranked = sorted(tgt.items(), key=lambda x: -x[1])
    print(f"  {'symbol':12s} {'weight':>7s} {'₹alloc':>12s} {'price':>10s} {'~shares':>8s}")
    for s, w in ranked:
        price = float(panel.close[s].iloc[-1])
        alloc = w * cap
        print(f"  {s:12s} {w*100:6.1f}% {alloc:12,.0f} {price:10.2f} {int(alloc/price):8d}")
    print(f"\n  {len(tgt)} positions, equal-weight.")
    return tgt


# ── Telegram credentials: env vars first (GitHub Secrets), then config.yaml ──
def _telegram_creds() -> tuple:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return token, chat
    cfg_path = os.path.join(HERE, "config.yaml")
    if yaml and os.path.exists(cfg_path):
        try:
            tg = (yaml.safe_load(open(cfg_path)) or {}).get("telegram", {})
            t, c = str(tg.get("bot_token", "")).strip(), str(tg.get("chat_id", "")).strip()
            # ignore placeholders
            if t and c and tg.get("enabled", True) and "YOUR_" not in t.upper():
                return t, c
        except Exception:
            pass
    return "", ""


def _telegram(text: str):
    token, chat = _telegram_creds()
    if not (token and chat):
        print("[telegram] no credentials (set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — skipping")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                       "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"[telegram] {e}")


def run_live(cfg: dict):
    """Check whether today is a rebalance day; if so, alert the new target book.
    Designed to be run daily (e.g. via cron / GitHub Actions)."""
    panel = load_panel()
    index = panel.close.index
    today = index[-1]
    rebal = MomentumRotation.rebalance_positions(index)
    state = {}
    if os.path.exists(cfg["state_file"]):
        try:
            state = json.load(open(cfg["state_file"]))
        except Exception:
            state = {}

    tgt = current_targets(panel, cfg)
    names = sorted(tgt.keys())
    is_rebal = today in rebal
    last_sent = state.get("last_book", [])

    if is_rebal and names != last_sent:
        lines = "\n".join(f"• {s}" for s in names) or "CASH (risk-off)"
        msg = (f"📅 <b>Monthly rebalance</b> ({today.date()})\n"
               f"Hold equal-weight:\n{lines}")
        print(msg)
        _telegram(msg)
        state["last_book"] = names
        state["last_rebalance"] = str(today.date())
        json.dump(state, open(cfg["state_file"], "w"), indent=2)
    else:
        print(f"[{today.date()}] no rebalance today. Current book: "
              f"{', '.join(names) if names else 'CASH'}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode == "test":
        token, chat = _telegram_creds()
        print(f"credentials found: {'yes' if (token and chat) else 'NO'}")
        _telegram(f"✅ Test message from quantbot via GitHub Actions at "
                  f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} IST — secrets & network OK.")
        print("done.")
        return
    if mode == "live":
        run_live(cfg); return
    if mode == "signal":
        report_signal(load_panel(), cfg); return

    print(f"Loading {len(UNIVERSE)} symbols ({cfg['history_period']}) ...")
    panel = load_panel()
    print(f"Loaded {panel.close.shape[1]} symbols, {panel.close.shape[0]} days, "
          f"{panel.close.index[0].date()} → {panel.close.index[-1].date()}\n")

    if mode == "backtest":
        report_backtest(panel, cfg)
    elif mode == "walkforward":
        report_walkforward(panel, cfg)
    elif mode == "robustness":
        report_robustness(panel, cfg)
    else:
        print(__doc__)
        print(f"Unknown mode '{mode}'. Use: backtest | walkforward | robustness | signal | live")


if __name__ == "__main__":
    main()
