# NSE Quant Bot — Momentum Rotation

A walk-forward-validated, cost-realistic **momentum rotation** strategy for NSE
equities that beats the Nifty on both absolute and risk-adjusted terms — and an
honest research trail showing the strategies that *didn't* work.

> **Paper / signal tool only.** No broker is connected. It tells you what to
> hold; you decide what to do. Past performance is not future performance.

---

## The headline result

11-year backtest (2015–2026, 86 liquid names, **real NSE delivery costs modeled**:
STT, stamp duty, exchange, SEBI, GST, DP charges):

| | CAGR | Max drawdown | Sharpe | Sortino | Beats Nifty |
|---|---|---|---|---|---|
| **Momentum rotation** | **~21.5%** | **−18%** | **1.32** | 1.43 | **8 / 12 years** |
| Nifty buy & hold | 10.1% | −38% | 0.68 | 0.84 | — |

It is **robust**, not cherry-picked:
- Every parameter combo (lookback 126–252, top 8–20) lands at CAGR 19–27% /
  Sharpe 1.2–1.4 — a broad plateau, not a single lucky point.
- Cost-insensitive: 21.5% at realistic cost, 20.3% at 2× cost, 22.6% at zero cost
  (monthly rebalancing keeps turnover and cost drag low).
- Holds up even on a **Nifty-50-only** universe (least survivorship bias):
  ~14% CAGR / −18% DD, still well ahead of buy & hold.

## The strategy (plain English)

1. **Once a month** (last trading day), rank the universe by trailing 12-month
   return, *skipping the most recent month* (avoids short-term reversal).
2. Buy the **top 15, equal-weight** — but only names trading **above their own
   200-day moving average**.
3. **Regime overlay:** if the **Nifty is below its 200-DMA**, hold **cash**
   instead. This is what cuts the drawdown from −38% to −18%.
4. Hold until next month. Repeat.

Why it works where day-trading didn't: it harvests the well-documented
*cross-sectional momentum premium*, trades rarely (so costs barely bite), and
the regime filter sidesteps the worst of bear markets.

## Usage

```bash
python quantbot.py backtest      # full 11-yr backtest vs Nifty
python quantbot.py walkforward   # per-year out-of-sample table
python quantbot.py robustness    # parameter + cost sensitivity grid
python quantbot.py signal        # TODAY's target portfolio (what to hold)
python quantbot.py live          # monthly check; Telegram alert on rebalance day
python test_quantbot.py          # correctness tests (incl. no-lookahead proof)
```

Telegram credentials are read from `config.yaml` (`telegram:` section) if present.
Run `live` daily (e.g. via the included GitHub Action); it only alerts when the
monthly book actually changes.

## Engine guarantees

- **No lookahead.** Signals use data through `close[t]`; orders fill at
  `open[t+1]`. This is enforced and unit-tested
  (`test_engine_fills_at_next_open_not_signal_bar`).
- **Real cost model.** `CostModel` computes per-trade NSE delivery charges, not a
  hand-waved flat %. Tune it in `quantbot.py` for your broker.
- **Benchmark always shown.** Every report compares against Nifty buy & hold —
  the only number that matters is whether you beat the index after costs.

## Honest caveats (read these)

- **Survivorship bias.** The universe is stocks liquid *today*; some of the
  outperformance comes from that. The Nifty-50-only test mitigates but does not
  eliminate this. A truly clean test needs point-in-time index constituents.
- **Drawdowns are real.** −18% peaks happen. You must be able to sit through them.
- **~10–15 trades/month.** Real fills, taxes (STT/LTCG/STCG), and slippage on
  less-liquid names will shave returns below the backtest.
- **Not advice.** Validate on paper for several months before risking capital.
  For many people a plain Nifty index fund is the rational, zero-effort choice.

## Repo layout

```
quantbot.py        # the platform (strategy, cost model, engine, metrics, live)
test_quantbot.py   # correctness tests
config.yaml        # Telegram creds (+ legacy params)
research/          # the honest trail of what DIDN'T work (see below)
.cache/            # cached price panels (gitignored)
```

## Research trail — what didn't work (and why it matters)

The strategies in `research/` are kept deliberately, because knowing what fails
is half of quant work:

- **`intraday_bot.py`** — 5-min intraday momentum (VWAP/EMA/ORB). Verdict:
  **no edge, 0/36 months profitable over 2 years.** A timezone bug initially
  *faked* a +38% result; once fixed, both the strategy and its exact inverse lost
  ~equally — proof the losses were pure friction (costs + noise-driven stops),
  not direction. Lesson: intraday mechanical trading is a coin flip you pay to play.
- **`swing_bot.py`** — daily RSI-2 mean reversion + Donchian trend. Verdict:
  **a real statistical edge, but too thin** — realistic portfolio CAGR ~3.8%,
  and it breaks even right around true delivery costs. Underperforms the index.

Momentum rotation is the one that survived honest testing. That's the whole point.
