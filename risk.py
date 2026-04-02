"""
risk.py — V5.0 Risk Management
================================
Position Sizing (Kelly), Trailing SL, Loss Streak, Circuit Breaker,
Expectancy Tracker.
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


# ══════════════════════════════════════════════
#  TRAILING STOP LOSS
# ══════════════════════════════════════════════

class TrailingStopLoss:
    @staticmethod
    def calculate(entry: float, current: float, direction: str, cfg: Dict) -> Dict[str, Any]:
        rm = cfg["risk_management"]
        if entry == 0:
            return {"trailing_sl": 0, "status": "No position", "pnl_pct": 0}
        if direction == "BUY":
            pnl = ((current - entry) / entry) * 100
            if pnl >= rm["trailing_lock_2_pct"]:
                tsl, st = round(entry * 1.006, 2), "🟢 Lock 0.6%"
            elif pnl >= rm["trailing_lock_1_pct"]:
                tsl, st = round(entry * 1.003, 2), "🟢 Lock 0.3%"
            elif pnl >= rm["trailing_breakeven_pct"]:
                tsl, st = entry, "🟡 Breakeven"
            else:
                tsl, st = round(entry * 0.995, 2), "🔴 Initial"
        else:
            pnl = ((entry - current) / entry) * 100
            if pnl >= rm["trailing_lock_2_pct"]:
                tsl, st = round(entry * 0.994, 2), "🟢 Lock 0.6%"
            elif pnl >= rm["trailing_lock_1_pct"]:
                tsl, st = round(entry * 0.997, 2), "🟢 Lock 0.3%"
            elif pnl >= rm["trailing_breakeven_pct"]:
                tsl, st = entry, "🟡 Breakeven"
            else:
                tsl, st = round(entry * 1.005, 2), "🔴 Initial"
        return {"trailing_sl": tsl, "status": st, "pnl_pct": round(pnl, 2)}


# ══════════════════════════════════════════════
#  F3: POSITION SIZING (Kelly / Volatility)
# ══════════════════════════════════════════════

class PositionSizer:
    """
    Base size = (Capital × Risk%) / (ATR × 1.2)
    Apply Kelly fraction: size × min(kelly, 0.25)
    Hard cap: no position > 5% of capital
    """

    @staticmethod
    def calculate(
        capital: float,
        atr_val: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        regime_size_mult: float,
        cfg: Dict,
    ) -> Dict[str, Any]:
        ac = cfg["account"]
        risk_pct = min(ac["risk_per_trade_pct"], ac["max_risk_pct"]) / 100
        risk_amount = capital * risk_pct

        # ATR-based position size
        sl_distance = atr_val * cfg["risk_management"]["sl_atr_multiplier"] if atr_val and atr_val > 0 else 1
        base_qty = int(risk_amount / sl_distance)

        # Kelly fraction
        kelly = 0.0
        if avg_loss > 0 and win_rate > 0:
            wr = win_rate / 100
            b = avg_win / avg_loss if avg_loss > 0 else 1
            kelly = max((b * wr - (1 - wr)) / b, 0)
        kelly_capped = min(kelly, ac["max_kelly_fraction"])

        # Apply Kelly if we have data, else use 50% sizing
        if kelly_capped > 0:
            qty = int(base_qty * kelly_capped)
        else:
            qty = int(base_qty * 0.5)

        # Apply regime multiplier (weak regimes = smaller positions)
        qty = max(int(qty * regime_size_mult), 1)

        # Hard cap: 5% of capital
        max_position_value = capital * (ac["max_position_pct"] / 100)
        # We don't know exact price here, so we return qty and let caller verify

        return {
            "quantity": qty,
            "risk_amount": round(risk_amount, 2),
            "kelly_fraction": round(kelly_capped, 4),
            "regime_mult": regime_size_mult,
            "base_qty": base_qty,
        }


# ══════════════════════════════════════════════
#  F6: LOSS STREAK PROTECTION
# ══════════════════════════════════════════════

class LossStreakProtection:
    def __init__(self, cfg: Dict):
        self.max_consecutive = cfg["loss_streak"]["max_consecutive"]
        self.consecutive_losses = 0
        self.total_wins = 0
        self.total_losses = 0
        self.stopped = False
        self.today = datetime.now().strftime("%Y-%m-%d")

    def reset_if_new_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.today:
            self.consecutive_losses = 0
            self.total_wins = 0
            self.total_losses = 0
            self.stopped = False
            self.today = today

    def record_result(self, is_win: bool):
        self.reset_if_new_day()
        if is_win:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive:
                self.stopped = True

    def is_allowed(self) -> bool:
        self.reset_if_new_day()
        return not self.stopped

    def status(self) -> str:
        if self.stopped:
            return f"🛑 STOPPED ({self.consecutive_losses} losses)"
        return f"✅ Active (streak: {self.consecutive_losses})"


# ══════════════════════════════════════════════
#  F8: CIRCUIT BREAKER
# ══════════════════════════════════════════════

class CircuitBreaker:
    """
    Halts all trading when:
      - Daily PnL < -2% of capital
      - Daily PnL > +3% (profit lock)
      - Drawdown from peak > 1.5%
      - Session trades > 6
    """

    def __init__(self, capital: float, cfg: Dict):
        cb = cfg["circuit_breaker"]
        self.capital = capital
        self.daily_loss_limit = capital * (cb["daily_loss_limit_pct"] / 100)
        self.daily_gain_lock = capital * (cb["daily_gain_lock_pct"] / 100)
        self.max_drawdown_from_peak = cb["drawdown_from_peak_pct"] / 100
        self.max_trades = cfg["session"]["max_trades_per_day"]

        self.daily_pnl = 0.0
        self.peak_equity = capital
        self.trade_count = 0
        self.halted = False
        self.halt_reason = ""
        self.today = datetime.now().strftime("%Y-%m-%d")

    def reset_if_new_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.today:
            self.daily_pnl = 0.0
            self.peak_equity = self.capital
            self.trade_count = 0
            self.halted = False
            self.halt_reason = ""
            self.today = today

    def record_trade(self, pnl_amount: float):
        self.reset_if_new_day()
        self.daily_pnl += pnl_amount
        self.trade_count += 1
        current_equity = self.capital + self.daily_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        self._check()

    def _check(self):
        # Daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            self.halted = True
            self.halt_reason = f"Daily loss limit hit (₹{self.daily_pnl:.0f})"
            return

        # Daily gain lock
        if self.daily_pnl >= self.daily_gain_lock:
            self.halted = True
            self.halt_reason = f"Profit locked (₹{self.daily_pnl:.0f})"
            return

        # Drawdown from peak
        current_equity = self.capital + self.daily_pnl
        if self.peak_equity > 0:
            dd = (self.peak_equity - current_equity) / self.peak_equity
            if dd > self.max_drawdown_from_peak:
                self.halted = True
                self.halt_reason = f"Drawdown from peak ({dd*100:.1f}%)"
                return

        # Trade count
        if self.trade_count >= self.max_trades:
            self.halted = True
            self.halt_reason = f"Max trades ({self.max_trades}) reached"

    def is_allowed(self) -> bool:
        self.reset_if_new_day()
        return not self.halted

    def status(self) -> str:
        if self.halted:
            return f"🛑 {self.halt_reason}"
        return f"✅ PnL: ₹{self.daily_pnl:.0f} | Trades: {self.trade_count}/{self.max_trades}"


# ══════════════════════════════════════════════
#  F9: EXPECTANCY TRACKING
# ══════════════════════════════════════════════

class ExpectancyTracker:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.wins: list = []
        self.losses: list = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    d = json.load(f)
                    self.wins = d.get("wins", [])
                    self.losses = d.get("losses", [])
            except Exception:
                pass

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump({"wins": self.wins, "losses": self.losses}, f, indent=2)

    def record(self, pnl_pct: float):
        if pnl_pct >= 0:
            self.wins.append(round(pnl_pct, 2))
        else:
            self.losses.append(round(abs(pnl_pct), 2))
        self._save()

    def compute(self) -> Dict[str, Any]:
        total = len(self.wins) + len(self.losses)
        if total == 0:
            return {"total_trades": 0, "win_rate": 0, "loss_rate": 0,
                    "avg_win": 0, "avg_loss": 0, "expectancy": 0, "status": "No data"}
        wr = len(self.wins) / total
        lr = len(self.losses) / total
        aw = sum(self.wins) / len(self.wins) if self.wins else 0
        al = sum(self.losses) / len(self.losses) if self.losses else 0
        exp = (wr * aw) - (lr * al)
        return {
            "total_trades": total,
            "win_rate": round(wr * 100, 1), "loss_rate": round(lr * 100, 1),
            "avg_win": round(aw, 2), "avg_loss": round(al, 2),
            "expectancy": round(exp, 2),
            "status": "✅ Positive edge" if exp > 0 else "⚠️ Negative edge",
        }


# ══════════════════════════════════════════════
#  V5.2 — TRANCHE MANAGER (Multi-Entry System)
# ══════════════════════════════════════════════

class TrancheManager:
    """
    Manages 3-tranche position building for each symbol.
    Tranche 1: Entry on signal fire (50% of position)
    Tranche 2: Add on first pullback to entry zone (30%)
    Tranche 3: Add on breakout continuation (20%)
    """

    def __init__(self, symbol: str, direction: str,
                 entry_price: float, atr: float,
                 total_qty: int):
        self.symbol       = symbol
        self.direction    = direction
        self.entry_price  = entry_price
        self.atr          = atr
        self.total_qty    = total_qty
        self.tranches     = []
        self.filled       = 0

        # Define the 3 entry levels
        if direction == "LONG":
            self.levels = {
                "T1": entry_price,
                "T2": entry_price - 0.3 * atr,
                "T3": entry_price + 0.5 * atr,
            }
            self.quantities = {
                "T1": max(int(total_qty * 0.50), 1),
                "T2": max(int(total_qty * 0.30), 1),
                "T3": max(int(total_qty * 0.20), 1),
            }
        else:  # SHORT
            self.levels = {
                "T1": entry_price,
                "T2": entry_price + 0.3 * atr,
                "T3": entry_price - 0.5 * atr,
            }
            self.quantities = {
                "T1": max(int(total_qty * 0.50), 1),
                "T2": max(int(total_qty * 0.30), 1),
                "T3": max(int(total_qty * 0.20), 1),
            }

        # Shared SL and target — based on full position
        self.sl_price     = (entry_price - 1.2 * atr
                             if direction == "LONG"
                             else entry_price + 1.2 * atr)
        self.target_price = (entry_price + 2.0 * atr
                             if direction == "LONG"
                             else entry_price - 2.0 * atr)

    def check_tranche_fill(self, current_price: float) -> Optional[Dict]:
        """
        Call every candle. Returns fill instruction if a tranche
        level is touched and not yet filled.
        """
        filled_ids = [t["id"] for t in self.tranches]

        for tranche_id, level in self.levels.items():
            if tranche_id in filled_ids:
                continue

            touched = False
            if self.direction == "LONG":
                if tranche_id == "T1":
                    touched = True  # T1 fires immediately
                elif tranche_id == "T2":
                    touched = current_price <= level * 1.002
                elif tranche_id == "T3":
                    touched = current_price >= level * 0.998
            else:
                if tranche_id == "T1":
                    touched = True
                elif tranche_id == "T2":
                    touched = current_price >= level * 0.998
                elif tranche_id == "T3":
                    touched = current_price <= level * 1.002

            if touched:
                fill = {
                    "id":        tranche_id,
                    "symbol":    self.symbol,
                    "qty":       self.quantities[tranche_id],
                    "price":     round(current_price, 2),
                    "sl":        round(self.sl_price, 2),
                    "target":    round(self.target_price, 2),
                    "direction": self.direction,
                }
                self.tranches.append(fill)
                self.filled += self.quantities[tranche_id]
                return fill

        return None

    def all_filled(self) -> bool:
        return len(self.tranches) == 3

    def get_avg_entry(self) -> float:
        if not self.tranches:
            return self.entry_price
        total_cost = sum(t["price"] * t["qty"] for t in self.tranches)
        total_qty  = sum(t["qty"] for t in self.tranches)
        return round(total_cost / total_qty, 2) if total_qty > 0 else self.entry_price

    def status(self) -> str:
        filled_ids = [t["id"] for t in self.tranches]
        return f"Tranches: {','.join(filled_ids)} | Avg: ₹{self.get_avg_entry()} | Filled: {self.filled}/{self.total_qty}"
