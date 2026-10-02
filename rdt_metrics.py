#!/usr/bin/env python3
"""
rdt_metrics.py -- deterministic math for the RealDayTrading method.

Why this exists: Real Relative Strength needs Wilder-smoothed ATR over a rolling
window, and time-of-day relative volume needs cumulative-volume comparison across
multiple sessions. Both are easy to get subtly wrong by hand and the errors are
invisible -- a mis-seeded ATR shifts RRS by enough to flip a trade decision.
Compute them here instead.

Input is the raw JSON returned by the Robinhood get_equity_historicals tool.
Save that response to a file (the whole thing, including the {"data": ...} wrapper)
and point this script at it.

Usage:
  python rdt_metrics.py rrs      --bars FILE --symbol NVDA [--vs SPY] [--length 12]
  python rdt_metrics.py rvol     --bars FILE --symbol NVDA
  python rdt_metrics.py levels   --bars FILE --symbol NVDA [--lookback 2]
  python rdt_metrics.py context  --bars FILE --symbol NVDA [--vs SPY]
  python rdt_metrics.py size     --entry 100.20 --stop 99.40 --target 102.00
                                 --account 50000 [--risk-pct 1.0] [--min-rr 2.0]

All price outputs are rounded for display only; math runs at full precision.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------- data loading

def load_bars(path):
    """Return {symbol: [bar, ...]} from either input format.

    Two formats are accepted because they trade off differently. JSON is the
    raw get_equity_historicals response -- paste it and go. The compact format
    costs far fewer tokens to write out, which matters when a single analysis
    needs several hundred bars across two symbols:

        #SPY
        2026-08-07T13:30,770.97,771.62,770.63,771.48,605466
        ...
        #NVDA
        2026-08-07T13:30,181.20,181.90,180.95,181.55,2201110

    Columns are timestamp,open,high,low,close,volume. Timestamps are UTC and
    must match across symbols -- RRS compares each stock bar to the market bar
    at the same instant, so misaligned series silently produce garbage.
    """
    with open(path) as f:
        text = f.read().strip()

    if not text.startswith("{") and not text.startswith("["):
        return _load_compact(text)

    raw = json.loads(text)
    if isinstance(raw, dict):
        results = raw.get("data", raw).get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        raise SystemExit("Unrecognised bars file structure.")

    out = {}
    for res in results:
        sym = res["symbol"].upper()
        bars = []
        for b in res.get("bars", []):
            if b.get("interpolated"):
                # Gap-fill bars carry no new information and would corrupt both
                # the true-range series and the volume averages.
                continue
            bars.append({
                "t": datetime.fromisoformat(b["begins_at"].replace("Z", "+00:00")),
                "o": float(b["open_price"]),
                "h": float(b["high_price"]),
                "l": float(b["low_price"]),
                "c": float(b["close_price"]),
                "v": float(b.get("volume") or 0),
            })
        bars.sort(key=lambda x: x["t"])
        out[sym] = bars
    return out


def _load_compact(text):
    out, cur = {}, None
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("#"):
            cur = line[1:].strip().upper()
            out.setdefault(cur, [])
            continue
        if cur is None:
            raise SystemExit(
                f"Line {lineno}: data before any '#SYMBOL' header."
            )
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise SystemExit(
                f"Line {lineno}: expected timestamp,o,h,l,c,v -- got {len(parts)} fields."
            )
        ts = parts[0].replace("Z", "")
        if "+" not in ts:
            ts += "+00:00"
        try:
            out[cur].append({
                "t": datetime.fromisoformat(ts),
                "o": float(parts[1]), "h": float(parts[2]),
                "l": float(parts[3]), "c": float(parts[4]),
                "v": float(parts[5]),
            })
        except ValueError as e:
            raise SystemExit(f"Line {lineno}: {e}")
    for sym in out:
        out[sym].sort(key=lambda x: x["t"])
    return out


def require(bars_by_sym, sym):
    sym = sym.upper()
    if sym not in bars_by_sym or not bars_by_sym[sym]:
        raise SystemExit(
            f"No bars for {sym} in the file. Fetch it in the same "
            f"get_equity_historicals call as the comparison symbol so both "
            f"series share identical bar timestamps."
        )
    return bars_by_sym[sym]


# ------------------------------------------------------------------- indicators

def true_ranges(bars):
    """TR series, index-aligned to bars. TR[0] is None (no prior close)."""
    tr = [None]
    for i in range(1, len(bars)):
        pc = bars[i - 1]["c"]
        tr.append(max(bars[i]["h"], pc) - min(bars[i]["l"], pc))
    return tr


def wilder_atr(bars, length, lag_one_bar=True):
    """Wilder-smoothed ATR, index-aligned to bars, None until seeded.

    lag_one_bar mirrors the reference ThinkScript, which feeds the *previous*
    bar's high/low/close into TrueRange. That keeps the current bar's own range
    out of the denominator -- otherwise a stock's big move inflates the very
    ATR used to judge whether the move was big, muting the signal exactly when
    it matters.
    """
    tr = true_ranges(bars)
    series = tr[:-1] if lag_one_bar else tr
    series = ([None] + series) if lag_one_bar else series

    atr = [None] * len(bars)
    vals, start = [], None
    for i, x in enumerate(series):
        if i >= len(bars):
            break
        if x is None:
            continue
        vals.append(x)
        if len(vals) == length:
            start = i
            break
    if start is None:
        return atr

    atr[start] = sum(vals) / length
    for i in range(start + 1, len(bars)):
        x = series[i] if i < len(series) else None
        if x is None:
            atr[i] = atr[i - 1]
        else:
            atr[i] = (atr[i - 1] * (length - 1) + x) / length
    return atr


def ema(values, period):
    out, k = [None] * len(values), 2.0 / (period + 1)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def rrs_series(sym_bars, cmp_bars, length):
    """Real Relative Strength, per bar, on the intersection of timestamps.

    RRS answers: given how far the market moved relative to its own normal
    range, how far *should* this stock have moved -- and how far did it
    actually move instead? Positive means it outran the expectation.
    """
    cmp_by_t = {b["t"]: b for b in cmp_bars}
    pairs = [(s, cmp_by_t[s["t"]]) for s in sym_bars if s["t"] in cmp_by_t]
    if len(pairs) <= length:
        raise SystemExit(
            f"Need more than {length} overlapping bars; got {len(pairs)}. "
            f"Widen the time range."
        )

    s_bars = [p[0] for p in pairs]
    c_bars = [p[1] for p in pairs]
    s_atr = wilder_atr(s_bars, length)
    c_atr = wilder_atr(c_bars, length)

    out = []
    for i in range(length, len(pairs)):
        sa, ca = s_atr[i], c_atr[i]
        if not sa or not ca:
            out.append({"t": s_bars[i]["t"], "rrs": None})
            continue
        cmp_move = c_bars[i]["c"] - c_bars[i - length]["c"]
        sym_move = s_bars[i]["c"] - s_bars[i - length]["c"]
        power = cmp_move / ca
        expected = power * sa
        out.append({
            "t": s_bars[i]["t"],
            "rrs": (sym_move - expected) / sa,
            "power_index": power,
            "sym_move": sym_move,
            "expected_move": expected,
            "sym_atr": sa,
        })
    return out


def session_key(t):
    """Group bars into US trading sessions (UTC timestamps -> ET calendar day)."""
    return (t - timedelta(hours=5)).date()


def time_of_day_rvol(bars):
    """Cumulative volume so far today vs the same point in prior sessions.

    Comparing today's total volume to a full-day average understates the
    morning and overstates the afternoon. The RDT criterion is volume relative
    to what this stock normally has done *by this hour*, so that's what this
    measures.
    """
    by_session = defaultdict(list)
    for b in bars:
        by_session[session_key(b["t"])].append(b)
    sessions = sorted(by_session)
    if len(sessions) < 2:
        raise SystemExit(
            "Need at least two sessions of intraday bars. Request a start_time "
            "covering ~10 trading days at 5minute interval."
        )

    today = sessions[-1]
    prior = sessions[:-1]

    def cumulative(day_bars):
        cum, total = {}, 0.0
        for b in day_bars:
            total += b["v"]
            cum[b["t"].strftime("%H:%M")] = total
        return cum

    today_cum = cumulative(by_session[today])
    if not today_cum:
        raise SystemExit("No bars in the most recent session.")
    last_clock = sorted(today_cum)[-1]
    today_vol = today_cum[last_clock]

    baselines = []
    for d in prior:
        c = cumulative(by_session[d])
        if last_clock in c:
            baselines.append(c[last_clock])
    if not baselines:
        raise SystemExit(
            "No prior session reached this time of day -- can't build a "
            "baseline. Check that the range includes full prior sessions."
        )

    avg = sum(baselines) / len(baselines)
    return {
        "session": str(today),
        "as_of_utc": last_clock,
        "cumulative_volume": today_vol,
        "baseline_avg": avg,
        "baseline_sessions": len(baselines),
        "rvol": (today_vol / avg) if avg else None,
    }


def swing_levels(bars, lookback=2):
    """Pivot highs/lows: a bar whose high (low) exceeds `lookback` neighbours
    on each side. These are the horizontal levels price actually reacted to,
    which is what 'void' and 'stop below structure' mean in practice."""
    highs, lows = [], []
    for i in range(lookback, len(bars) - lookback):
        win = bars[i - lookback:i + lookback + 1]
        if bars[i]["h"] == max(b["h"] for b in win):
            highs.append({"t": bars[i]["t"], "price": bars[i]["h"]})
        if bars[i]["l"] == min(b["l"] for b in win):
            lows.append({"t": bars[i]["t"], "price": bars[i]["l"]})
    return highs, lows


def void_analysis(bars, lookback=2):
    """Distance from the last close to the nearest overhead and underfoot
    pivot. A long needs room above; a short needs room below."""
    last = bars[-1]["c"]
    highs, lows = swing_levels(bars, lookback)
    above = sorted([h["price"] for h in highs if h["price"] > last])
    below = sorted([l["price"] for l in lows if l["price"] < last], reverse=True)

    res = above[0] if above else None
    sup = below[0] if below else None
    hi = max(b["h"] for b in bars)
    lo = min(b["l"] for b in bars)

    return {
        "last": last,
        "nearest_resistance": res,
        "void_up_pct": ((res - last) / last * 100) if res else None,
        "void_up_to_range_high_pct": ((hi - last) / last * 100),
        "nearest_support": sup,
        "void_down_pct": ((last - sup) / last * 100) if sup else None,
        "void_down_to_range_low_pct": ((last - lo) / last * 100),
        "range_high": hi,
        "range_low": lo,
    }


# ------------------------------------------------------------------- reporting

def fmt(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def cmd_rrs(a):
    bars = load_bars(a.bars)
    s = require(bars, a.symbol)
    c = require(bars, a.vs)
    series = [r for r in rrs_series(s, c, a.length) if r["rrs"] is not None]
    if not series:
        raise SystemExit("ATR never seeded -- widen the time range.")

    tail = series[-a.show:]
    latest = series[-1]
    recent = [r["rrs"] for r in series[-a.length:]]
    avg = sum(recent) / len(recent)

    print(f"Real Relative Strength: {a.symbol.upper()} vs {a.vs.upper()} "
          f"(rolling {a.length} bars)")
    print(f"  latest RRS          : {fmt(latest['rrs'])}")
    print(f"  avg of last {len(recent):>2} bars : {fmt(avg)}")
    print(f"  SPY power index     : {fmt(latest['power_index'])}")
    print(f"  moved / expected    : {fmt(latest['sym_move'])} vs "
          f"{fmt(latest['expected_move'])}")
    print()
    print("  Recent readings (UTC):")
    for r in tail:
        n = int(min(abs(r["rrs"]) * 2, 16))
        bar = ("+" if r["rrs"] >= 0 else "-") * n
        print(f"    {r['t'].strftime('%m-%d %H:%M')}  {r['rrs']:+7.2f}  {bar}")
    print()
    print("  Persistence is the signal, not the peak. A single strong bar")
    print("  decays out of the rolling window; institutional accumulation")
    print("  holds the reading positive bar after bar.")


def cmd_rvol(a):
    bars = load_bars(a.bars)
    r = time_of_day_rvol(require(bars, a.symbol))
    print(f"Time-of-day relative volume: {a.symbol.upper()}")
    print(f"  session          : {r['session']} (through {r['as_of_utc']} UTC)")
    print(f"  volume so far    : {r['cumulative_volume']:,.0f}")
    print(f"  normal by now    : {r['baseline_avg']:,.0f} "
          f"(avg of {r['baseline_sessions']} prior sessions)")
    print(f"  RVOL             : {fmt(r['rvol'])}x")
    verdict = ("clears the 1.2x floor" if (r["rvol"] or 0) >= 1.2
               else "BELOW the 1.2x floor -- the move is coasting, not pushed")
    print(f"  -> {verdict}")


def cmd_levels(a):
    bars = load_bars(a.bars)
    v = void_analysis(require(bars, a.symbol), a.lookback)
    print(f"Structure: {a.symbol.upper()}  (last {v['last']:.2f})")
    print(f"  nearest resistance : {fmt(v['nearest_resistance'])}  "
          f"-> void up   {fmt(v['void_up_pct'])}%")
    print(f"  nearest support    : {fmt(v['nearest_support'])}  "
          f"-> void down {fmt(v['void_down_pct'])}%")
    print(f"  range high / low   : {fmt(v['range_high'])} / {fmt(v['range_low'])}")
    print(f"  clear to range high: {fmt(v['void_up_to_range_high_pct'])}%")
    print(f"  clear to range low : {fmt(v['void_down_to_range_low_pct'])}%")
    print()
    print("  A long wants >=1% of clear air overhead, a short >=1% below.")
    print("  Without it the target is capped before the trade starts.")


def cmd_context(a):
    """Everything the checklist needs about one name, in one pass."""
    bars = load_bars(a.bars)
    s = require(bars, a.symbol)
    c = require(bars, a.vs)

    print(f"=== {a.symbol.upper()} ===")
    print()
    for length, label in ((12, "5-min chart, 1hr window"), (5, "5-bar window")):
        try:
            series = [r for r in rrs_series(s, c, length) if r["rrs"] is not None]
            if series:
                recent = [r["rrs"] for r in series[-length:]]
                print(f"RRS vs {a.vs.upper()} ({label}): latest "
                      f"{fmt(series[-1]['rrs'])}, avg {fmt(sum(recent)/len(recent))}")
        except SystemExit:
            pass

    try:
        r = time_of_day_rvol(s)
        print(f"RVOL (time-of-day)   : {fmt(r['rvol'])}x")
    except SystemExit as e:
        print(f"RVOL                 : unavailable ({e})")

    closes = [b["c"] for b in s]
    e8 = ema(closes, 8)
    if e8[-1]:
        ext = (closes[-1] - e8[-1]) / e8[-1] * 100
        print(f"8 EMA                : {e8[-1]:.2f}  (price {ext:+.2f}% away)")

    v = void_analysis(s)
    print(f"Void up / down       : {fmt(v['void_up_pct'])}% / "
          f"{fmt(v['void_down_pct'])}%")
    print(f"Last                 : {v['last']:.2f}")


def cmd_size(a):
    """Position size from the technical stop, plus the R:R gate.

    The order matters: the stop comes from where the thesis breaks and the
    target from where price runs out of room. Size is what falls out of those
    two. Choosing a size first and then hunting for a stop that justifies it
    is the mistake this ordering is designed to prevent.
    """
    risk_per_share = abs(a.entry - a.stop)
    reward_per_share = abs(a.target - a.entry)
    if risk_per_share == 0:
        raise SystemExit("Entry and stop are identical.")

    direction = "LONG" if a.target > a.entry else "SHORT"
    if direction == "LONG" and a.stop >= a.entry:
        print("!! Stop is at or above entry on a long -- check the inputs.")
    if direction == "SHORT" and a.stop <= a.entry:
        print("!! Stop is at or below entry on a short -- check the inputs.")

    rr = reward_per_share / risk_per_share
    budget = a.account * (a.risk_pct / 100.0)
    shares = int(budget // risk_per_share)

    print(f"{direction}  entry {a.entry:.2f}  stop {a.stop:.2f}  "
          f"target {a.target:.2f}")
    print(f"  risk / share     : {risk_per_share:.2f}")
    print(f"  reward / share   : {reward_per_share:.2f}")
    print(f"  reward:risk      : {rr:.2f} : 1")
    print(f"  risk budget      : {budget:,.2f}  "
          f"({a.risk_pct:.2f}% of {a.account:,.0f})")
    print(f"  shares           : {shares:,}")
    deployed = shares * a.entry
    print(f"  capital deployed : {deployed:,.2f}")
    if a.buying_power and deployed > a.buying_power:
        affordable = int(a.buying_power // a.entry)
        print(f"  !! exceeds buying power of {a.buying_power:,.0f} -- "
              f"caps at {affordable:,} shares")
        print(f"     At that size the stop only risks "
              f"{affordable * risk_per_share:,.2f}, not the full budget.")
        print("     A tight stop on a high-priced stock runs out of buying")
        print("     power before it runs out of risk budget. That is a real")
        print("     constraint, not a rounding note -- size to the cap.")
    print(f"  max loss at stop : {shares * risk_per_share:,.2f}")
    print(f"  gain at target   : {shares * reward_per_share:,.2f}")
    print()
    if rr < a.min_rr:
        print(f"  REJECT: {rr:.2f}:1 is below the {a.min_rr:.1f}:1 floor.")
        print("  Do not widen the target to fix this -- the target is set by")
        print("  structure, not by arithmetic. Either the entry is late or the")
        print("  setup does not have room. Wait for a pullback or pass.")
    else:
        print(f"  Clears the {a.min_rr:.1f}:1 floor.")
        breakeven = 1.0 / (1.0 + rr) * 100
        print(f"  Break-even win rate at this ratio: {breakeven:.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_bars(sp):
        sp.add_argument("--bars", required=True, help="saved historicals JSON")
        sp.add_argument("--symbol", required=True)

    sp = sub.add_parser("rrs", help="Real Relative Strength vs a benchmark")
    add_bars(sp)
    sp.add_argument("--vs", default="SPY")
    sp.add_argument("--length", type=int, default=12,
                    help="rolling window in bars (12 on 5min = 1hr; 5 on daily)")
    sp.add_argument("--show", type=int, default=12)
    sp.set_defaults(func=cmd_rrs)

    sp = sub.add_parser("rvol", help="time-of-day relative volume")
    add_bars(sp)
    sp.set_defaults(func=cmd_rvol)

    sp = sub.add_parser("levels", help="pivots, void, range")
    add_bars(sp)
    sp.add_argument("--lookback", type=int, default=2)
    sp.set_defaults(func=cmd_levels)

    sp = sub.add_parser("context", help="RRS + RVOL + 8EMA + void in one pass")
    add_bars(sp)
    sp.add_argument("--vs", default="SPY")
    sp.set_defaults(func=cmd_context)

    sp = sub.add_parser("size", help="position size and R:R gate")
    sp.add_argument("--entry", type=float, required=True)
    sp.add_argument("--stop", type=float, required=True)
    sp.add_argument("--target", type=float, required=True)
    sp.add_argument("--account", type=float, required=True)
    sp.add_argument("--risk-pct", type=float, default=1.0)
    sp.add_argument("--min-rr", type=float, default=2.0)
    sp.add_argument("--buying-power", type=float, default=None,
                    help="cap on deployable capital; from get_portfolio")
    sp.set_defaults(func=cmd_size)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
