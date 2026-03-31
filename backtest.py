"""
backtest.py — V5.0 Backtesting Engine
========================================
Replays historical 1-min candles through strategy.py and risk.py.
Generates an HTML report with equity curve, metrics, and trade table.

Usage:
  python3 backtest.py --symbol RELIANCE --days 90
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import (
    rsi, ema, vwap, volume_ratio, atr,
    supertrend, macd, bollinger_bands, adx, pivot_points,
)
from execution import load_config, SlippageModel


def run_backtest(symbol: str, days: int = 90, cfg: Dict = None):
    """Run full backtest and return results dict."""
    if cfg is None:
        cfg = load_config()

    if "." not in symbol:
        symbol = f"{symbol}.NS"
    symbol = symbol.upper()

    print(f"\n[BACKTEST] {symbol} — {days} days")
    print(f"[BACKTEST] Loading data...")

    ticker = yf.Ticker(symbol)

    # Load daily data for previous day pivots
    df_daily = ticker.history(period=f"{days + 10}d", interval="1d")
    if df_daily.empty:
        print("[ERROR] No daily data available")
        return None

    # Load 1-min data in chunks (yfinance only allows 7 days of 1m data)
    # For longer backtests, use 5-min candles instead
    interval = "5m" if days > 7 else "1m"
    period = f"{days}d"

    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print("[ERROR] No intraday data available")
        return None

    df = df[df["Volume"] > 0].copy()  # Clean
    print(f"[BACKTEST] Loaded {len(df)} candles ({interval})")

    # ── Simulation state ──
    sc = cfg["scoring"]
    rm = cfg["risk_management"]
    slippage_pct = SlippageModel.get_slippage_pct(symbol, cfg)

    trades: List[Dict] = []
    equity = [cfg["account"]["capital"]]
    capital = cfg["account"]["capital"]
    position = None  # {"direction", "entry", "sl", "tgt", "time"}

    min_lookback = 30  # Need enough candles for indicators
    closes = df["Close"]

    print(f"[BACKTEST] Running simulation...")

    for i in range(min_lookback, len(df)):
        window = df.iloc[:i + 1]
        current = float(window["Close"].iloc[-1])
        candle_time = window.index[-1]

        # Skip non-market hours (rough filter)
        try:
            t = candle_time.time() if hasattr(candle_time, 'time') else None
            if t and (t < dtime(9, 30) or t > dtime(14, 30)):
                continue
        except Exception:
            pass

        # ── Check if position hit SL or TGT ──
        if position is not None:
            high = float(window["High"].iloc[-1])
            low = float(window["Low"].iloc[-1])

            hit_sl = False
            hit_tgt = False

            if position["direction"] == "BUY":
                if low <= position["sl"]:
                    hit_sl = True
                    exit_price = SlippageModel.apply_slippage(position["sl"], "SELL", slippage_pct)
                elif high >= position["tgt"]:
                    hit_tgt = True
                    exit_price = SlippageModel.apply_slippage(position["tgt"], "SELL", slippage_pct)
            else:  # SELL
                if high >= position["sl"]:
                    hit_sl = True
                    exit_price = SlippageModel.apply_slippage(position["sl"], "BUY", slippage_pct)
                elif low <= position["tgt"]:
                    hit_tgt = True
                    exit_price = SlippageModel.apply_slippage(position["tgt"], "BUY", slippage_pct)

            if hit_sl or hit_tgt:
                if position["direction"] == "BUY":
                    pnl = exit_price - position["entry"]
                else:
                    pnl = position["entry"] - exit_price

                pnl_pct = (pnl / position["entry"]) * 100
                capital += pnl
                equity.append(capital)

                trades.append({
                    "entry_time": str(position["time"]),
                    "exit_time": str(candle_time),
                    "direction": position["direction"],
                    "entry": position["entry"],
                    "exit": exit_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": "TARGET" if hit_tgt else "STOPLOSS",
                })
                position = None
                continue

        # ── Skip if already in position ──
        if position is not None:
            continue

        # ── Compute indicators ──
        w_closes = window["Close"]
        w_volumes = window["Volume"]
        atr_val = atr(window, cfg["indicators"]["atr_period"])
        adx_val = adx(window, cfg["indicators"]["adx_period"])
        rsi_val = rsi(w_closes, cfg["indicators"]["rsi_period"])
        st_res = supertrend(window, cfg["indicators"]["supertrend_period"],
                            cfg["indicators"]["supertrend_multiplier"])
        macd_res = macd(w_closes, cfg["indicators"]["macd_fast"],
                        cfg["indicators"]["macd_slow"], cfg["indicators"]["macd_signal"])
        boll_res = bollinger_bands(w_closes, cfg["indicators"]["bollinger_period"],
                                    cfg["indicators"]["bollinger_std"])
        vol_r = volume_ratio(w_volumes, 20)
        vwap_val = vwap(window)
        ema9 = ema(w_closes, cfg["indicators"]["ema_fast"])

        if vwap_val is None:
            vwap_val = current

        # Get previous day data for pivots
        try:
            daily_before = df_daily[df_daily.index < candle_time]
            if len(daily_before) >= 2:
                prev_d = daily_before.iloc[-1]
                pivots_data = pivot_points(float(prev_d["High"]), float(prev_d["Low"]), float(prev_d["Close"]))
            else:
                pivots_data = None
        except Exception:
            pivots_data = None

        open_price = float(window["Open"].iloc[-1])
        prev_candle = float(w_closes.iloc[-2]) if len(w_closes) >= 2 else current
        pct_from_open = round(((current - open_price) / open_price) * 100, 2)

        # ── Simple scoring (uses same weights from config) ──
        data = {
            "current_price": current, "open_price": open_price,
            "high": float(window["High"].max()), "low": float(window["Low"].min()),
            "vwap": vwap_val, "prev_candle_close": prev_candle,
            "rsi": rsi_val, "volume_ratio": vol_r, "adx": adx_val,
            "supertrend": st_res, "macd": macd_res, "bollinger": boll_res,
            "pivots": pivots_data, "atr": atr_val,
        }

        buy_score = _quick_score(data, pct_from_open, "BUY", cfg)
        sell_score = _quick_score(data, pct_from_open, "SELL", cfg)

        min_score = sc["min_score"]

        # ── Entry logic ──
        if buy_score >= min_score and pct_from_open <= -sc["move_threshold_pct"]:
            if st_res is None or st_res[1] != "DOWN":
                entry = SlippageModel.apply_slippage(current, "BUY", slippage_pct)
                sl_off = (atr_val * rm["sl_atr_multiplier"]) if atr_val else entry * 0.005
                tgt_off = (atr_val * rm["target_atr_multiplier"]) if atr_val else entry * 0.015
                position = {
                    "direction": "BUY", "entry": entry,
                    "sl": round(entry - sl_off, 2), "tgt": round(entry + tgt_off, 2),
                    "time": candle_time,
                }

        elif sell_score >= min_score and pct_from_open >= sc["move_threshold_pct"]:
            if st_res is None or st_res[1] != "UP":
                entry = SlippageModel.apply_slippage(current, "SELL", slippage_pct)
                sl_off = (atr_val * rm["sl_atr_multiplier"]) if atr_val else entry * 0.005
                tgt_off = (atr_val * rm["target_atr_multiplier"]) if atr_val else entry * 0.015
                position = {
                    "direction": "SELL", "entry": entry,
                    "sl": round(entry + sl_off, 2), "tgt": round(entry - tgt_off, 2),
                    "time": candle_time,
                }

    # ── Close open position at last price ──
    if position:
        last_price = float(df["Close"].iloc[-1])
        if position["direction"] == "BUY":
            pnl = last_price - position["entry"]
        else:
            pnl = position["entry"] - last_price
        capital += pnl
        equity.append(capital)
        trades.append({
            "entry_time": str(position["time"]),
            "exit_time": "EOD",
            "direction": position["direction"],
            "entry": position["entry"],
            "exit": last_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round((pnl / position["entry"]) * 100, 2),
            "result": "WIN" if pnl > 0 else "LOSS",
            "reason": "FORCED_CLOSE",
        })

    # ── Compute metrics ──
    metrics = _compute_metrics(trades, equity, cfg["account"]["capital"], days)
    metrics["symbol"] = symbol
    metrics["period"] = f"{days} days"
    metrics["interval"] = interval
    metrics["slippage_pct"] = slippage_pct

    # ── Generate HTML report ──
    report_path = _generate_report(symbol, trades, equity, metrics)
    metrics["report_path"] = report_path

    # ── Print summary ──
    _print_summary(metrics)

    return metrics


def _quick_score(data, pct, direction, cfg):
    """Simplified scoring for backtest speed."""
    sc = cfg["scoring"]
    score = 0.0
    current = data["current_price"]
    vwap_val = data["vwap"]
    vwap_dist = abs((current - vwap_val) / vwap_val) * 100 if vwap_val else 999

    if direction == "BUY":
        if pct <= -sc["move_threshold_pct"]: score += sc["weight_move"]
        if vwap_dist <= sc["vwap_proximity_pct"]: score += sc["weight_vwap"]
        if current > data["prev_candle_close"]: score += sc["weight_candle"]
        if data["rsi"] and data["rsi"] < sc["rsi_oversold"]: score += sc["weight_rsi"]
        if data["volume_ratio"] and data["volume_ratio"] >= sc["volume_spike_ratio"]: score += sc["weight_volume"]
        if data["supertrend"] and data["supertrend"][1] == "UP": score += sc["weight_supertrend"]
        if data["macd"] and data["macd"]["bullish"]: score += sc["weight_macd"]
        if data["bollinger"] and current <= data["bollinger"]["lower"] * 1.002: score += sc["weight_bollinger"]
        if data["adx"] and data["adx"] >= sc["adx_trending"]: score += sc["weight_adx"]
    else:
        if pct >= sc["move_threshold_pct"]: score += sc["weight_move"]
        if vwap_dist <= sc["vwap_proximity_pct"]: score += sc["weight_vwap"]
        if current < data["prev_candle_close"]: score += sc["weight_candle"]
        if data["rsi"] and data["rsi"] > sc["rsi_overbought"]: score += sc["weight_rsi"]
        if data["volume_ratio"] and data["volume_ratio"] >= sc["volume_spike_ratio"]: score += sc["weight_volume"]
        if data["supertrend"] and data["supertrend"][1] == "DOWN": score += sc["weight_supertrend"]
        if data["macd"] and not data["macd"]["bullish"]: score += sc["weight_macd"]
        if data["bollinger"] and current >= data["bollinger"]["upper"] * 0.998: score += sc["weight_bollinger"]
        if data["adx"] and data["adx"] >= sc["adx_trending"]: score += sc["weight_adx"]

    return score


def _compute_metrics(trades, equity, initial_capital, days):
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "gross_pnl": 0, "net_pnl": 0, "max_drawdown_pct": 0,
            "sharpe": 0, "avg_hold_time": "N/A", "best_trade": 0, "worst_trade": 0,
        }

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]

    # Max drawdown
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = abs(float(dd.min())) * 100

    # Sharpe ratio (annualized, assuming ~250 trading days)
    if len(pnl_pcts) > 1:
        returns = np.array(pnl_pcts)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(250 / max(days, 1)) if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "gross_pnl": round(sum(pnls), 2),
        "net_pnl": round(equity[-1] - initial_capital, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "best_trade": round(max(pnl_pcts), 2) if pnl_pcts else 0,
        "worst_trade": round(min(pnl_pcts), 2) if pnl_pcts else 0,
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 2) if losses else 0,
    }


def _generate_report(symbol, trades, equity, metrics):
    """Generate HTML report with Plotly equity curve."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_name = f"backtest_{symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    report_path = os.path.join(base_dir, report_name)

    equity_chart_data = ",".join([str(round(e, 2)) for e in equity])

    trades_html = ""
    for t in trades:
        color = "#22c55e" if t["result"] == "WIN" else "#ef4444"
        trades_html += f"""<tr>
            <td>{t['entry_time']}</td><td>{t['exit_time']}</td>
            <td>{t['direction']}</td><td>₹{t['entry']}</td><td>₹{t['exit']}</td>
            <td style="color:{color};font-weight:bold">₹{t['pnl']} ({t['pnl_pct']}%)</td>
            <td>{t['reason']}</td></tr>"""

    # Warnings
    warnings = ""
    if metrics["sharpe"] < 1.0:
        warnings += '<div style="color:#f59e0b;padding:8px;background:#451a03;border-radius:8px;margin:8px 0">⚠️ Sharpe ratio below 1.0 — strategy may not be profitable enough</div>'
    if metrics["max_drawdown_pct"] > 8.0:
        warnings += '<div style="color:#ef4444;padding:8px;background:#450a0a;border-radius:8px;margin:8px 0">🚨 Max drawdown exceeds 8% — high risk</div>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest: {symbol}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e5e5e5; padding: 20px; }}
  h1 {{ color: #818cf8; }} h2 {{ color: #a78bfa; }}
  .card {{ background: #171717; border-radius: 12px; padding: 20px; margin: 16px 0; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
  .metric {{ background: #262626; padding: 16px; border-radius: 8px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: bold; color: #818cf8; }}
  .metric .label {{ font-size: 12px; color: #a3a3a3; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #262626; padding: 10px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #262626; }}
</style></head><body>
<h1>📊 Backtest Report — {symbol}</h1>
<p>{metrics['period']} | Interval: {metrics['interval']} | Slippage: {metrics['slippage_pct']}%</p>
{warnings}
<div class="card"><div class="metrics">
  <div class="metric"><div class="value">{metrics['total_trades']}</div><div class="label">Total Trades</div></div>
  <div class="metric"><div class="value">{metrics['win_rate']}%</div><div class="label">Win Rate</div></div>
  <div class="metric"><div class="value">₹{metrics['net_pnl']}</div><div class="label">Net PnL</div></div>
  <div class="metric"><div class="value">{metrics['sharpe']}</div><div class="label">Sharpe Ratio</div></div>
  <div class="metric"><div class="value">{metrics['max_drawdown_pct']}%</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{metrics['best_trade']}%</div><div class="label">Best Trade</div></div>
  <div class="metric"><div class="value">{metrics['worst_trade']}%</div><div class="label">Worst Trade</div></div>
</div></div>
<div class="card"><h2>Equity Curve</h2><div id="chart"></div></div>
<div class="card"><h2>Trade Log ({len(trades)} trades)</h2>
<table><tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry₹</th><th>Exit₹</th><th>PnL</th><th>Reason</th></tr>
{trades_html}</table></div>
<script>
var data = [{{{equity_chart_data}}}];
Plotly.newPlot('chart', [{{y: [{equity_chart_data}], type: 'scatter', mode: 'lines',
  line: {{color: '#818cf8', width: 2}}, fill: 'tozeroy', fillcolor: 'rgba(129,140,248,0.1)'}}],
  {{paper_bgcolor: '#171717', plot_bgcolor: '#171717', font: {{color: '#e5e5e5'}},
  xaxis: {{title: 'Trade #', gridcolor: '#262626'}},
  yaxis: {{title: 'Equity (₹)', gridcolor: '#262626'}},
  margin: {{t: 20, b: 40}}}});
</script></body></html>"""

    with open(report_path, "w") as f:
        f.write(html)

    print(f"[BACKTEST] Report saved: {report_name}")
    return report_path


def _print_summary(m):
    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS — {m['symbol']}")
    print(f"{'='*50}")
    print(f"  Trades:       {m['total_trades']} ({m['wins']}W / {m['losses']}L)")
    print(f"  Win Rate:     {m['win_rate']}%")
    print(f"  Net PnL:      ₹{m['net_pnl']}")
    print(f"  Sharpe:       {m['sharpe']}")
    print(f"  Max Drawdown: {m['max_drawdown_pct']}%")
    print(f"  Best Trade:   {m['best_trade']}%")
    print(f"  Worst Trade:  {m['worst_trade']}%")
    if m['sharpe'] < 1.0:
        print(f"  ⚠️  WARNING: Sharpe < 1.0")
    if m['max_drawdown_pct'] > 8.0:
        print(f"  🚨 WARNING: Max drawdown > 8%")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest intraday strategy")
    parser.add_argument("--symbol", default="RELIANCE", help="NSE symbol")
    parser.add_argument("--days", type=int, default=90, help="Number of days")
    args = parser.parse_args()
    run_backtest(args.symbol, args.days)
