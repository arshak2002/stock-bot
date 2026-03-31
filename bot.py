"""
bot.py — V5.0 Main Runner (Institutional Pro Edition)
=======================================================
Pipeline: Config → Data → ORB → Regime → Liquidity → News →
          Circuit Breaker → Loss Guard → Slippage → Score → Signal

Run: python3 bot.py
"""

import os
import time
import json
import schedule

# Force structural execution environment identically to IST for cloud containers
if hasattr(time, 'tzset'):
    os.environ['TZ'] = 'Asia/Kolkata'
    time.tzset()

from datetime import datetime
import subprocess
import yfinance as yf
import pandas as pd

from execution import (
    load_config, YFinanceProvider, SessionFilter, NewsFilter,
    SlippageModel, TradeLogger, TelegramAlerter, play_alert,
)
from strategy import (
    MarketRegime, OpeningRangeBreakout, LiquiditySweepFilter,
    CorrelationGuard, MasterStrategy,
)
from risk import (
    LossStreakProtection, CircuitBreaker, PositionSizer, ExpectancyTracker,
)
from indicators import adx as compute_adx, ema as compute_ema


class IntradayBot:
    """V5.0 institutional-grade intraday trading system."""

    def __init__(self, symbols: list):
        self.cfg = load_config()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.symbols = symbols

        self.providers   = {s: YFinanceProvider(s) for s in symbols}
        self.orbs        = {s: OpeningRangeBreakout() for s in symbols}
        
        self.logger      = TradeLogger(os.path.join(base_dir, "trade_log.csv"))
        self.loss_guard  = LossStreakProtection(self.cfg)
        self.circuit     = CircuitBreaker(self.cfg["account"]["capital"], self.cfg)
        self.expectancy  = ExpectancyTracker(os.path.join(base_dir, "expectancy.json"))
        self.pos_sizer   = PositionSizer()

        self.trade_count = {"BUY": 0, "SELL": 0}
        
        # Schedule morning brief
        schedule.every().day.at("09:14").do(self.send_morning_brief)

    def reload_symbols(self, new_symbols: list):
        """Dynamically hot-swaps active universe without breaking PnL or circuit state."""
        if not new_symbols or set(self.symbols) == set(new_symbols):
            return
            
        print(f"\n[*] 🔄 Hot-loading new optimal watchlist: {new_symbols}")
        # Only instantiate components for newly injected symbols to save memory
        for sym in new_symbols:
            if sym not in self.providers:
                self.providers[sym] = YFinanceProvider(sym)
                self.orbs[sym] = OpeningRangeBreakout()
                
        self.symbols = new_symbols

    def _fmt(self, val, prefix="", suffix=""):
        return f"{prefix}{val}{suffix}" if val is not None else "N/A"

    def _display(self, data, result, regime_info, orb_res, liquidity, extras):
        now = datetime.now().strftime("%H:%M:%S")
        cfg = self.cfg
        st = data.get("supertrend", {})
        st_str = f"₹{st.get('st_value', 0)} (1m:{st.get('dir_1min')}|5m:{st.get('dir_5min')})" if st else "N/A"
        
        macd_div = data.get('macd_div', 'NONE')
        macd_str = f"Div: {macd_div}"
        b = data["bollinger"]
        boll_str = f"₹{b['lower']}–₹{b['upper']} (BW:{b['bandwidth']}%)" if b else "N/A"
        piv = data.get("pivots")

        regime = regime_info["regime"]
        confidence = regime_info["confidence"]
        news_st = extras.get("news_status", "N/A")

        print(f"\n{'═' * 70}")
        print(f"  [{now}]  {data['symbol']}    Session: {SessionFilter.status(cfg)}")
        print(f"  Regime: {regime} ({confidence}% conf)  |  News: {news_st}")
        print(f"{'═' * 70}")

        print(f"  Price:      ₹{data['current_price']}  (Open: ₹{data['open_price']})")
        print(f"  H / L:      ₹{data['high']} / ₹{data['low']}     Range: ₹{result['intraday_range']}")
        print(f"  Volume:     {data['volume']:,}    Ratio: {self._fmt(data['volume_ratio'], suffix='×')}")
        print(f"  VWAP:       ₹{data['vwap']}   Gap: {data['gap_pct']:+.2f}%   ATR: {self._fmt(data['atr'], '₹')}")
        print(f"  % Open:     {result['pct_from_open']}%")

        print(f"  ── Indicators ──")
        rsi = data.get("rsi_dual", {})
        print(f"  RSI:        7: {self._fmt(rsi.get('rsi7'))}  14: {self._fmt(rsi.get('rsi14'))}")
        print(f"  EMA 9/21:   {self._fmt(data['ema9'], '₹')} / {self._fmt(data['ema21'], '₹')}")
        print(f"  Supertrend: {st_str}")
        print(f"  MACD:       {macd_str}")
        print(f"  Bollinger:  {boll_str}")
        print(f"  ADX:        {self._fmt(data['adx'])}")
        if piv:
            print(f"  Pivots:     S3=₹{piv['s3']}  S2=₹{piv['s2']}  S1=₹{piv['s1']}  "
                  f"P=₹{piv['pivot']}  R1=₹{piv['r1']}  R2=₹{piv['r2']}  R3=₹{piv['r3']}")

        orb_h = orb_res.get("orb_high", "—")
        orb_l = orb_res.get("orb_low", "—")
        print(f"  ORB:        ₹{orb_l} – ₹{orb_h}   Signal: {orb_res['orb_signal']}")

        if liquidity["is_liquidity_trap"]:
            print(f"  ⚠️  Trap:    {liquidity['trap_type']}")

        sig = result["signal"]
        print(f"  ── Signal ──")
        print(f"  Signal:     {sig}   "
              f"(B={self.trade_count['BUY']}  S={self.trade_count['SELL']})")
        print(f"  Score:      {result['score_display']}")
        print(f"  Grade:      {result['grade']}")

        if sig != "NO TRADE":
            slip = extras.get("slippage_pct", 0)
            edge = extras.get("true_edge", 0)
            pos = extras.get("position_size", 0)
            risk_amt = extras.get("risk_amount", 0)
            print(f"  Entry:      ₹{result['entry']}  (slip: {slip}%)")
            print(f"  SL:         ₹{result['stop_loss']}  (ATR-based)")
            print(f"  Target:     ₹{result['target']}  (ATR-based)")
            print(f"  True Edge:  {edge}%")
            print(f"  Size:       {pos} units  |  Risk: ₹{risk_amt}")
            if result.get("breakdown"):
                print(f"  Breakdown:  {result['breakdown']}")
        else:
            print(f"  Entry/SL/T: —")

        print(f"  Reason:     {result['reason']}")

        print(f"  ── Risk ──")
        print(f"  Loss Guard: {self.loss_guard.status()}")
        print(f"  Circuit:    {self.circuit.status()}")
        print(f"  Positions:  {CorrelationGuard.status()}")
        exp = self.expectancy.compute()
        if exp["total_trades"] > 0:
            print(f"  Expectancy: {exp['expectancy']}  ({exp['status']})")
            print(f"  Win Rate:   {exp['win_rate']}%  W={exp['avg_win']}%  L={exp['avg_loss']}%")
        else:
            print(f"  Expectancy: No data")

        print(f"{'═' * 70}")

    def run(self):
        cfg = self.cfg
        cap = cfg["account"]["capital"]
        poll = cfg["session"]["poll_interval_sec"]

        print("╔══════════════════════════════════════════════════════════╗")
        print("║  INTRADAY BOT V5.0 — INSTITUTIONAL PRODUCTION EDITION   ║")
        print("║  Modules: indicators · strategy · risk · execution       ║")
        print("║  5-State Regime | 11-pt Scoring | ATR SL | Kelly Sizing  ║")
        print("║  ORB | Liquidity Filter | Circuit Breaker | News RSS     ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"\n[*] Symbols: {', '.join(self.symbols)}")
        print(f"[*] Capital: ₹{cap:,.0f}")
        print(f"[*] Min score: {cfg['scoring']['min_score']}/11  |  Max trades: {cfg['session']['max_trades_per_day']}/day")
        print(f"[*] Polling every {poll}s — Ctrl+C to stop.\n")

        # Startup Diagnostic Alert
        try:
            if SessionFilter.is_holiday():
                msg = "🛑 *Market Closed*\nToday is a weekend or public holiday. Bot standing by..."
                print(f"[!] {msg}")
                TelegramAlerter.send(msg, "STARTUP", cfg)
            else:
                TelegramAlerter.send("✅ *Bot V5.0 Online*\nPipeline initialized. Awaiting market open...", "STARTUP", cfg)
        except Exception:
            pass

        try:
            while True:
                schedule.run_pending()
                
                # If it is currently a holiday, sleep for 1 hour then re-check (allows seamless next-day crossover)
                if SessionFilter.is_holiday():
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛑 Holiday resting mode. Interrogating again in 1 hour...")
                    time.sleep(3600)
                    continue
                
                # Hot-reload config
                self.cfg = load_config()
                cfg = self.cfg

                # Hot-reload optimal watchlist generated by screener.py (if freshly updated)
                try:
                    fresh_wl = get_watchlist()
                    self.reload_symbols(fresh_wl)
                except Exception as e:
                    print(f"[*] Error reloading watchlist: {e}")

                for sym in self.symbols:
                    # ── 1. Fetch ──
                    data = self.providers[sym].fetch(cfg)
                    if data is None:
                        # try next symbol rather than sleeping
                        continue

                    df_1m = data.pop("df_1m", None)

                    # ── 2. ORB ──
                    self.orbs[sym].capture_range(df_1m)
                    orb_result = self.orbs[sym].evaluate(data["current_price"], data["volume_ratio"])

                    # ── 3. Regime ──
                    regime_info = MarketRegime.detect(data, cfg)

                    # ── 4. Liquidity Sweep ──
                    atr_val = data.get("atr") or 0
                    liquidity = LiquiditySweepFilter.detect(df_1m, data["current_price"], atr_val, cfg)

                    # ── 5. Circuit Breaker ──
                    if not self.circuit.is_allowed():
                        result = _no_trade(data, self.circuit.status())
                        self._display(data, result, regime_info, orb_result, liquidity, {})
                        continue

                    # ── 6. Loss Guard ──
                    if not self.loss_guard.is_allowed():
                        result = _no_trade(data, self.loss_guard.status())
                        self._display(data, result, regime_info, orb_result, liquidity, {})
                        continue

                    # ── 7. News ──
                    news_safe = NewsFilter.is_safe_to_trade(data["symbol"], cfg)
                    news_status = NewsFilter.status(data["symbol"], cfg)
                    if not news_safe:
                        result = _no_trade(data, f"🗞️ News block — {news_status}")
                        self._display(data, result, regime_info, orb_result, liquidity,
                                      {"news_status": news_status})
                        continue

                    # ── 8. Strategy ──
                    result = MasterStrategy.evaluate(data, regime_info, orb_result, liquidity, cfg)

                    extras = {
                        "regime": regime_info["regime"],
                        "regime_confidence": regime_info["confidence"],
                        "orb_signal": orb_result["orb_signal"],
                        "news_status": news_status,
                        "circuit_active": self.circuit.halted,
                    }

                    # ── 9. Post-signal processing ──
                    if result["signal"] != "NO TRADE":
                        # Slippage check
                        slippage_pct = SlippageModel.get_slippage_pct(data["symbol"], cfg)
                        true_edge = SlippageModel.compute_true_edge(
                            result["entry"], result["target"], result["stop_loss"],
                            slippage_pct, result["signal"],
                        )
                        extras["slippage_pct"] = slippage_pct
                        extras["true_edge"] = true_edge

                        if not SlippageModel.passes_edge_filter(true_edge, data["symbol"], cfg):
                            min_te = SlippageModel.get_true_edge_threshold(data["symbol"], cfg)
                            result = _no_trade(data, f"Edge too low ({true_edge}% < {min_te}%)")
                            self._display(data, result, regime_info, orb_result, liquidity, extras)
                            continue

                        # Correlation guard
                        if not CorrelationGuard.is_allowed(data["symbol"], cfg):
                            result = _no_trade(data, f"Correlated with existing position")
                            self._display(data, result, regime_info, orb_result, liquidity, extras)
                            continue

                        # Position sizing
                        exp = self.expectancy.compute()
                        pos = PositionSizer.calculate(
                            cap, atr_val,
                            exp["win_rate"], exp["avg_win"], exp["avg_loss"],
                            regime_info["size_mult"], cfg,
                        )
                        extras["position_size"] = pos["quantity"]
                        extras["risk_amount"] = pos["risk_amount"]

                        # Record trade
                        self.trade_count[result["signal"]] += 1
                        CorrelationGuard.add_position(data["symbol"])

                    # ── 10. Display + Log + Alert ──
                    self._display(data, result, regime_info, orb_result, liquidity, extras)

                    if result["signal"] != "NO TRADE":
                        self.logger.log(data, result, extras)
                        play_alert()
                        TelegramAlerter.send(
                            TelegramAlerter.format_signal(data, result, regime_info["regime"]),
                            data["symbol"], cfg,
                        )

                # Wait poll interval after checking all symbols
                time.sleep(poll)

        except KeyboardInterrupt:
            print(f"\n[!] Bot stopped.")
            print(f"[*] Trades: B={self.trade_count['BUY']}, S={self.trade_count['SELL']}")
            print(f"[*] {self.circuit.status()}")
            self.logger.export_eod_json()
            exp = self.expectancy.compute()
            if exp["total_trades"] > 0:
                print(f"[*] Expectancy: {exp['expectancy']} ({exp['status']})")

    def send_morning_brief(self):
        """Fetches Nifty trend and sends a Telegram brief at 9:14 AM."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 Generating Morning Brief...")
        cfg = self.cfg
        
        try:
            # 1. Fetch Nifty Trend
            nifty = yf.download("^NSEI", period="60d", interval="1d", progress=False)
            if nifty.empty:
                raise ValueError("Nifty data fetch returned empty")
            
            # Flatten MultiIndex columns if present
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
                
            nifty_adx = compute_adx(nifty, 14) or "N/A"
            ema9_val = compute_ema(nifty["Close"], 9)
            ema21_val = compute_ema(nifty["Close"], 21)
            
            if ema9_val is None or ema21_val is None:
                trend = "Unknown"
            else:
                trend = "Bullish" if ema9_val > ema21_val else "Bearish"
                if abs(ema9_val - ema21_val) / ema21_val < 0.002:
                    trend = "Neutral/Range"
                
            # 2. Get Watchlist
            watchlist_symbols = []
            if os.path.exists("watchlist.json"):
                with open("watchlist.json", "r") as f:
                    wdata = json.load(f)
                    watchlist_symbols = [f"{c['symbol']}({c['score']})" for c in wdata.get("candidates", [])]
            
            watchlist_str = ", ".join(watchlist_symbols) if watchlist_symbols else "Manual Input Required"
            
            # 3. Risk Stats
            cap = cfg["account"]["capital"]
            risk_pct = cfg["account"]["risk_per_trade_pct"]
            risk_amt = cap * (risk_pct / 100)
            loss_limit_pct = cfg["circuit_breaker"]["daily_loss_limit_pct"]
            loss_limit_amt = abs(cap * (loss_limit_pct / 100))
            
            # 4. Format Message
            brief = (
                f"─────────────────────────────────\n"
                f"📊 *TRADING BRIEF — {datetime.now().strftime('%d %b %Y')}*\n"
                f"Nifty Trend: {trend} (ADX: {nifty_adx})\n"
                f"Watchlist: {watchlist_str}\n"
                f"Capital: ₹{cap:,.0f} | Risk/Trade: ₹{risk_amt:,.0f}\n"
                f"Circuit Breaker: ARMED (Daily Loss Limit: ₹{loss_limit_amt:,.0f})\n"
                f"─────────────────────────────────"
            )
            
            print(brief)
            TelegramAlerter.send(brief, "BRIEF", cfg)
            
        except Exception as e:
            print(f"[ERROR] Morning Brief failed: {e}")


def _no_trade(data, reason):
    pct = round(((data["current_price"] - data["open_price"]) / data["open_price"]) * 100, 2)
    rng = round(data["high"] - data["low"], 2)
    return {
        "signal": "NO TRADE", "score": 0, "grade": "-",
        "score_display": "—", "entry": 0, "stop_loss": 0, "target": 0,
        "reason": reason, "breakdown": "",
        "pct_from_open": pct, "intraday_range": rng,
    }


def run_screener_job():
    """Runs screener.py script via subprocess to update watchlist.json"""
    from execution import SessionFilter
    if SessionFilter.is_holiday():
        print("[!] 🛑 Market Closed. Skipping screener run.")
        return
        
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    screener_path = os.path.join(base_dir, "screener.py")
    try:
        subprocess.run([sys.executable, screener_path], check=True)
        print("[*] Screener completed. Watchlist updated.")
    except Exception as e:
        print(f"[ERROR] Screener execution failed: {e}")

def get_watchlist():
    if os.path.exists("watchlist.json"):
        try:
            with open("watchlist.json", "r") as f:
                data = json.load(f)
            age_minutes = (datetime.now() - datetime.fromisoformat(data["generated_at"])).seconds / 60
            if age_minutes < 30:  # use if generated within last 30 min
                return [c["symbol"] for c in data["candidates"]]
        except Exception:
            pass
    return None

if __name__ == "__main__":
    from keep_alive import keep_alive
    try:
        keep_alive()
    except Exception as e:
        print(f"[*] Could not start keep_alive webserver: {e}")

    # Schedule screener
    schedule.every().day.at("09:00").do(run_screener_job)
    
    watchlist = get_watchlist()
    
    from execution import SessionFilter
    is_holiday = SessionFilter.is_holiday()

    # If no fresh watchlist exists, automatically run screener now
    if not watchlist and not is_holiday:
        print("[*] No valid watchlist found. Running screener now...")
        run_screener_job()
        watchlist = get_watchlist()

    if watchlist:
        print(f"Auto-loaded watchlist: {watchlist}")
        symbols = watchlist
    else:
        if is_holiday:
            symbols = ["RELIANCE.NS"]
            print("[*] Market Closed. Standing by with default symbol.")
        else:
            print("[!] Screener returned no stocks. Defaulting to NIFTY50 leaders.")
            symbols = ["RELIANCE.NS", "HDFCBANK.NS"]

    bot = IntradayBot(symbols)
    bot.run()
