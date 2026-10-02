# rdt-relative-strength

Deterministic relative-strength math for stock analysis: a Python CLI and a
matching TradingView indicator. Both implement Real Relative Strength (RRS),
which measures how far a stock moved beyond what the market's move implies,
normalized by volatility.

Built as the calculation layer for an AI-assisted analysis workflow. The model
fetches bars through a brokerage data connector and reasons about the setup;
every number it reasons from is computed here, so the math is repeatable and
not estimated by the model.

## What's in here

| File | What it does |
| --- | --- |
| `rdt_metrics.py` | CLI for RRS, time-of-day relative volume, pivot levels, context summary and position sizing. Standard library only. |
| `indicators/rrs_vs_spy.pine` | TradingView Pine Script v6 indicator: RRS vs a benchmark with fast/slow lines, confirmed cross signals and alerts. |
| `examples/sample_daily_bars.txt` | Synthetic bars (not market data) so the commands below run as-is. |

## Quick start

Requires Python 3. No dependencies.

```bash
# Relative strength of DEMO vs SPY over a 5-bar window
python rdt_metrics.py rrs --bars examples/sample_daily_bars.txt --symbol DEMO --vs SPY --length 5

# Support/resistance pivots and room to the range edges
python rdt_metrics.py levels --bars examples/sample_daily_bars.txt --symbol DEMO

# One-screen summary: RRS, relative volume, 8 EMA extension, room up/down
python rdt_metrics.py context --bars examples/sample_daily_bars.txt --symbol DEMO --vs SPY

# Position size from entry, stop, target and account risk
python rdt_metrics.py size --entry 100.20 --stop 99.40 --target 102.00 --account 50000
```

`rvol` compares today's cumulative volume with the same point in prior
sessions, so it needs at least two prior sessions of intraday bars.

## Input formats

**Compact text**, one `#SYMBOL` header followed by `timestamp,open,high,low,close,volume` rows:

```
#SPY
2026-01-05T14:30,500.00,501.20,499.10,500.85,1200000
#DEMO
2026-01-05T14:30,100.00,100.90,99.60,100.40,950000
```

**JSON**, a list of `{symbol, bars[]}` results (optionally wrapped in
`{"data": {"results": [...]}}`), where each bar carries `begins_at`,
`open_price`, `high_price`, `low_price`, `close_price` and `volume`. Bars
flagged `interpolated` are dropped.

Timestamps are UTC and must match across symbols. RRS compares each stock bar
with the benchmark bar at the same instant, so misaligned series give wrong
readings without raising an error.

## How RRS is calculated

```
power_index = benchmark_move / benchmark_ATR
RRS         = (stock_move - power_index * stock_ATR) / stock_ATR
```

ATR uses Wilder smoothing. A positive reading that persists across bars is the
signal; a single large reading is not.

## TradingView indicator

Paste `indicators/rrs_vs_spy.pine` into the Pine Editor and add it to a chart.
The fast line above the slow line means relative strength is building. Cross
signals print only on closed bars; set alerts to "Once Per Bar Close" to match.

The indicator smooths with EMA(8, 21) for earlier turns, while the CLI reports
the raw Wilder-ATR reading, so the two are complementary and will not match
bar for bar.

## Notes

This is an independent implementation of a publicly described method, not
vendor source code. It is an analysis tool and not financial advice.
