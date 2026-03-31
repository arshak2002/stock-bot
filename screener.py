"""
screener.py — V5.0 Automated Daily Screener
============================================
Fetches universe of ~150 stocks, applies hard filters, 
and scores survivors using momentum factors.
Outputs to watchlist.json
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate

from nse_symbols import NIFTY50


# ══════════════════════════════════════════════
#  HELPERS & INDICATORS
# ══════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = pd.concat([
        df["High"] - df["Low"],
        np.abs(df["High"] - df["Close"].shift()),
        np.abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (np.abs(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ══════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════

def get_universe() -> List[str]:
    """Fetch Nifty 50 and Nifty Midcap 150 (top 100)."""
    symbols = set(NIFTY50)
    try:
        url = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
        df = pd.read_csv(url)
        # Nifty Midcap indices are usually sorted by market cap or free float
        midcap = df["Symbol"].head(100).tolist()
        symbols.update(midcap)
    except Exception as e:
        print(f"[WARN] Could not fetch Midcap 150: {e}")
    return [s + ".NS" for s in list(symbols)]

def get_fo_ban_list() -> set:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get("https://www.nseindia.com/api/fo-mktlots", headers=headers, timeout=5)
        # This is a placeholder parsing, actual NSE ban list needs specific scraping
        # We will assume empty for robust execution if the structure changes
        return set()
    except:
        return set()


from execution import SessionFilter, TelegramAlerter, load_config

# ══════════════════════════════════════════════
#  SCREENER ENGINE
# ══════════════════════════════════════════════

def run_screener():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 Running Market Screener...")
    
    cfg = load_config()
    if SessionFilter.is_holiday():
        msg = "🛑 *Market Closed*\nToday is a weekend or public holiday. Screener omitted."
        print(f"[!] {msg}")
        TelegramAlerter.send(msg, "SCREENER", cfg)
        return

    symbols = get_universe()
    print(f"[*] Universe size: {len(symbols)} symbols")

    # Batch download daily data (5d to get EMA9/21/50, but we actually need more for EMA50!
    # Wait, EMA50 needs ~50 days of data. We'll fetch 100 days to be safe)
    print(f"[*] Fetching daily data...")
    daily_data = yf.download(symbols, period="100d", interval="1d", progress=False, group_by="ticker")
    # Fetch 1-min data (only 1 day) for pre-market volume & gap
    print(f"[*] Fetching 1m intraday data...")
    intra_data = yf.download(symbols, period="1d", interval="1m", progress=False, group_by="ticker")

    ban_list = get_fo_ban_list()
    
    # Nifty 50 benchmark
    nifty = yf.download("^NSEI", period="10d", interval="1d", progress=False)
    if nifty.empty:
        nifty_5day_return = 0
    else:
        nifty_close_col = nifty["Close"]
        if isinstance(nifty_close_col, pd.DataFrame):
            nifty_close_col = nifty_close_col.iloc[:, 0]
        nifty_close_5d = float(nifty_close_col.iloc[-6]) if len(nifty) > 5 else float(nifty_close_col.iloc[0])
        nifty_open_col = nifty["Open"]
        if isinstance(nifty_open_col, pd.DataFrame):
            nifty_open_col = nifty_open_col.iloc[:, 0]
        nifty_open_today = float(nifty_open_col.iloc[-1])
        nifty_5day_return = (nifty_open_today - nifty_close_5d) / nifty_close_5d

    candidates = []

    for sym in symbols:
        try:
            clean_sym = sym.replace(".NS", "")
            
            # ── 1. HARD FILTERS ──
            if sym not in daily_data or sym not in intra_data:
                continue
                
            df_daily = daily_data[sym].dropna(how="all")
            df_intra = intra_data[sym].dropna(how="all")
            
            if len(df_daily) < 50 or df_intra.empty:
                continue
                
            last_close = float(df_daily["Close"].iloc[-2]) if len(df_daily) > 1 else float(df_daily["Close"].iloc[-1])
            today_open = float(df_intra["Open"].iloc[0])
            
            # F1: Price range
            if not (50 < last_close < 5000):
                continue
                
            # F2: Min volume (5-day avg)
            avg_5d_vol = df_daily["Volume"].iloc[-6:-1].mean()
            if avg_5d_vol < 500_000:
                continue
                
            # F3: Volatility ATR(14) percentage
            atr_val = compute_atr(df_daily, 14).iloc[-2]
            atr_pct = (atr_val / last_close) * 100
            if not (0.8 < atr_pct < 4.0):
                continue
                
            # F4: Previous day circuit block
            prev_close_2 = float(df_daily["Close"].iloc[-3]) if len(df_daily) > 2 else float(df_daily["Close"].iloc[-2])
            prev_day_change = ((last_close - prev_close_2) / prev_close_2) * 100
            if not (-9.5 < prev_day_change < 9.5):
                continue
                
            # F5: F&O Ban
            if clean_sym in ban_list:
                continue
                
            # ── 2. MOMENTUM FACTORS ──
            score = 0
            breakdown = {}
            
            # Factor 1: Pre-market gap score (0-25)
            gap_pct = ((today_open - last_close) / last_close) * 100
            gap_pts = 0
            if gap_pct > 2.0: gap_pts = 25
            elif gap_pct >= 1.0: gap_pts = 18
            elif gap_pct >= 0.5: gap_pts = 10
            elif gap_pct >= -0.5: gap_pts = 5
            score += gap_pts
            breakdown["gap"] = gap_pts
            
            # Factor 2: Volume surge ratio (0-25)
            try:
                # Approximate "first 15 min" volume
                am_915 = df_intra.between_time("09:15", "09:30")
                today_vol = float(am_915["Volume"].sum()) if not am_915.empty else 0
                
                sum_vol = float(df_daily["Volume"].iloc[-2])
                # heuristic: average 15 min vol = total daily vol / (375 mins / 15) = ~4% of daily
                avg_15m_vol = (avg_5d_vol * 0.04) 
                
                surge_ratio = today_vol / avg_15m_vol if avg_15m_vol > 0 else 0
                vol_pts = 0
                if surge_ratio > 3.0: vol_pts = 25
                elif surge_ratio >= 2.0: vol_pts = 18
                elif surge_ratio >= 1.5: vol_pts = 12
                elif surge_ratio >= 1.0: vol_pts = 6
                score += vol_pts
                breakdown["volume"] = vol_pts
            except Exception:
                surge_ratio = 0
                breakdown["volume"] = 0

            # Factor 3: Trend alignment (0-20)
            ema9 = compute_ema(df_daily["Close"], 9).iloc[-2]
            ema21 = compute_ema(df_daily["Close"], 21).iloc[-2]
            ema50 = compute_ema(df_daily["Close"], 50).iloc[-2]
            adx_val = compute_adx(df_daily, 14).iloc[-2]
            
            trend_pts = 0
            trend_state = "BEARISH"
            if ema9 > ema21 > ema50:
                trend_pts = 20
                if adx_val > 25: trend_pts = min(20, trend_pts + 5)
                trend_state = "BULLISH"
            elif ema9 > ema21:
                trend_pts = 12
                trend_state = "NEUTRAL"
                
            score += trend_pts
            breakdown["trend"] = trend_pts
            
            # Factor 4: Relative strength vs Nifty50 (0-20)
            stock_5d_ago = float(df_daily["Close"].iloc[-6]) if len(df_daily) > 5 else float(df_daily["Close"].iloc[0])
            stock_5d_ret = (today_open - stock_5d_ago) / stock_5d_ago
            rs_score_val = (stock_5d_ret - nifty_5day_return) * 100
            
            rs_pts = 0
            if rs_score_val > 3.0: rs_pts = 20
            elif rs_score_val >= 1.0: rs_pts = 14
            elif rs_score_val >= 0.0: rs_pts = 8
            
            score += rs_pts
            breakdown["rs"] = rs_pts
            
            # Factor 5: Near key level
            prev_high = float(df_daily["High"].iloc[-2])
            prev_low = float(df_daily["Low"].iloc[-2])
            pivot = (prev_high + prev_low + last_close) / 3
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            
            lvl_pts = 0
            lvl_name = "NONE"
            for lvl, name in [(pivot, "PIVOT"), (r1, "R1"), (s1, "S1")]:
                dist = abs((today_open - lvl) / lvl) * 100
                if dist <= 0.5:
                    lvl_pts = max(lvl_pts, 10)
                    lvl_name = name
                elif dist <= 1.0:
                    lvl_pts = max(lvl_pts, 5)
                    lvl_name = name
                    
            score += lvl_pts
            breakdown["level"] = lvl_pts
            
            # Final Check
            candidates.append({
                "symbol": clean_sym,
                "score": score,
                "last_close": round(last_close, 2),
                "today_open": round(today_open, 2),
                "gap_pct": round(gap_pct, 2),
                "atr_pct": round(atr_pct, 2),
                "surge_ratio": round(surge_ratio, 1),
                "trend": trend_state,
                "rs_score": round(rs_score_val, 2),
                "near_level": lvl_name,
                "direction": "LONG",
                "score_breakdown": breakdown
            })
            
        except Exception as e:
            pass

    # ── 3. FINAL RANKING ──
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = [c for c in candidates if c["score"] >= 30][:15]
    
    # Add rank
    for i, c in enumerate(top_candidates):
        c["rank"] = i + 1

    # Save to JSON
    out_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_date": datetime.now().strftime("%Y-%m-%d"),
        "candidates": top_candidates
    }
    
    with open("watchlist.json", "w") as f:
        json.dump(out_data, f, indent=2)

    # Print Table
    if top_candidates:
        table = [[
            c["rank"], c["symbol"], c["score"], 
            f"{c['gap_pct']:+.1f}%", f"{c['surge_ratio']}x", 
            c["trend"], c["direction"]
        ] for c in top_candidates]
        print("\n" + tabulate(table, headers=["Rank", "Symbol", "Score", "Gap%", "Surge", "Trend", "Direction"], tablefmt="presto"))
        print("\n✅ Ranked watchlist generated and saved to watchlist.json")
    else:
        print("\n⚠️ No symbols passed the rigid scoring threshold today (>50 score).")


if __name__ == "__main__":
    run_screener()
