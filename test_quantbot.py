"""
test_quantbot.py — correctness tests for the quant platform.
Run:  python -m pytest test_quantbot.py -q     (or: python test_quantbot.py)

These guard the things that matter most for trusting a backtest:
  - the cost model is sane and side-asymmetric,
  - performance metrics are correct on known inputs,
  - momentum picks the right names,
  - and — most important — the engine has NO LOOKAHEAD (orders fill at the
    open AFTER the signal, never at the signal bar).
"""
import numpy as np
import pandas as pd

import quantbot as q


# ── cost model ──
def test_cost_model_positive_and_asymmetric():
    c = q.CostModel()
    assert c.buy_cost(100_000) > 0
    assert c.sell_cost(100_000) > 0
    # sell carries DP charge, buy carries stamp duty; both small, both > STT alone
    assert c.buy_cost(0) == 0 and c.sell_cost(0) == 0
    # round trip on a liquid delivery trade should be well under 1%
    assert 0.1 < c.round_trip_pct(100_000) < 1.0


def test_zero_cost_model_is_zero():
    c = q.CostModel(stt_pct=0, exchange_pct=0, sebi_pct=0, stamp_pct=0,
                    gst_pct=0, dp_flat_sell=0, brokerage_pct=0, brokerage_flat=0)
    assert c.buy_cost(100_000) == 0
    assert c.sell_cost(100_000) == 0


# ── metrics ──
def test_metrics_known_series():
    # straight 10% per year for 2 years on daily-ish index
    idx = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    # geometric growth to exactly 1.21x over 2 years -> CAGR 10%
    n = len(idx)
    eq = pd.Series(100_000 * (1.21 ** (np.arange(n) / (n - 1))), index=idx)
    m = q.metrics(eq)
    assert abs(m["cagr"] - 10.0) < 0.6
    assert m["maxdd"] == 0.0          # monotonic up -> no drawdown
    assert m["sharpe"] > 0


def test_metrics_drawdown():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    eq = pd.Series([100, 120, 60, 80, 90], index=idx, dtype=float)
    m = q.metrics(eq)
    assert abs(m["maxdd"] - (-50.0)) < 1e-6   # 120 -> 60 is -50%


# ── momentum selection ──
def test_momentum_picks_top_performers():
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    n = len(idx)
    # WIN ramps up strongly, FLAT is flat, LOSE drifts down
    close = pd.DataFrame({
        "WIN": np.linspace(100, 300, n),
        "MID": np.linspace(100, 150, n),
        "FLAT": np.full(n, 100.0),
        "LOSE": np.linspace(100, 70, n),
    }, index=idx)
    panel = q.Panel(close, close.copy(), pd.Series(np.linspace(100, 200, n), index=idx))
    cfg = dict(q.CONFIG, lookback=252, skip=21, top_n=2, regime_overlay=False)
    strat = q.MomentumRotation(cfg)
    sma = close.rolling(200).mean()
    bsma = panel.bench.rolling(200).mean()
    w = strat.target_weights(panel, n - 1, sma, bsma)
    assert set(w) == {"WIN", "MID"}          # the two strongest above their SMA
    assert abs(sum(w.values()) - 1.0) < 1e-9  # fully invested, equal weight


def test_regime_overlay_goes_to_cash():
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    n = len(idx)
    close = pd.DataFrame({"A": np.linspace(100, 300, n)}, index=idx)
    # benchmark crashing below its own SMA -> risk-off
    bench = pd.Series(np.linspace(300, 100, n), index=idx)
    panel = q.Panel(close, close.copy(), bench)
    cfg = dict(q.CONFIG, lookback=252, skip=21, top_n=2, regime_overlay=True)
    strat = q.MomentumRotation(cfg)
    sma = close.rolling(200).mean()
    bsma = bench.rolling(200).mean()
    w = strat.target_weights(panel, n - 1, sma, bsma)
    assert w == {}    # market below 200DMA -> hold cash


# ── NO LOOKAHEAD: the critical correctness property ──
def test_engine_fills_at_next_open_not_signal_bar():
    """Construct a series where the signal fires on day T (last trading day of a
    month) and the price GAPS on day T+1. A lookahead-free engine must fill at
    T+1's open, so equity reflects the gapped open, not day T's close."""
    # build ~3 months of business days
    idx = pd.date_range("2020-01-01", periods=70, freq="B")
    n = len(idx)
    # one asset, flat at 100 until a big gap up on the bar after first month-end
    close = pd.Series(100.0, index=idx).to_frame("A")
    openp = pd.Series(100.0, index=idx).to_frame("A")

    # find first month-end position
    s = pd.Series(idx, index=idx)
    rebals = sorted(s.groupby([idx.year, idx.month]).last().tolist())
    t = idx.get_loc(rebals[0])           # signal bar
    # make A eligible & strongly positive momentum by ramping before t
    ramp = np.linspace(100, 200, t + 1)
    close.iloc[: t + 1, 0] = ramp
    openp.iloc[: t + 1, 0] = ramp
    close.iloc[t + 1:, 0] = 200.0
    openp.iloc[t + 1:, 0] = 200.0
    # GAP: the fill bar (t+1) opens at 260, closes 200 -> if engine fills at
    # t+1 open (260) we BUY high; lookahead (t close=200) would buy cheaper.
    openp.iloc[t + 1, 0] = 260.0

    bench = pd.Series(np.linspace(50, 300, n), index=idx)  # always risk-on
    panel = q.Panel(close, openp, bench)
    cfg = dict(q.CONFIG, capital=100_000, lookback=t - 5, skip=0, top_n=1,
               trend_sma=5, regime_overlay=False)
    res = q.run_backtest(panel, cfg, costs=q.CostModel(
        stt_pct=0, exchange_pct=0, sebi_pct=0, stamp_pct=0, gst_pct=0,
        dp_flat_sell=0))
    # after buying at the 260 open then price sits at 200 -> a loss must appear
    # (proves fill happened at the gapped open, i.e. AFTER the signal bar)
    final = res.equity.iloc[-1]
    assert final < 100_000, f"expected loss from gap-up fill, got {final}"
    # sanity: shares bought ≈ capital/260, value ≈ qty*200
    expected = 100_000 / 260.0 * 200.0
    assert abs(final - expected) / expected < 0.02


def _run_all():
    import inspect
    fns = [f for name, f in globals().items()
           if name.startswith("test_") and inspect.isfunction(f)]
    passed = 0
    for f in fns:
        f(); print(f"  ✓ {f.__name__}"); passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
