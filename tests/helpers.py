"""Synthetic fixtures for the tests.

Prices here are fabricated, flat-with-overrides series so that every expected value can be
checked by hand. They are NOT market data and are never used for findings.

These builders make test inputs obey the same bar-start/bar-end and UTC conventions as production
inputs while allowing individual prices, missing bars, early closes, and extra-hours rows to be
controlled explicitly.
"""
import pandas as pd

NY = "America/New_York"


def ny(s: str) -> pd.Timestamp:
    """New York wall-clock string -> UTC Timestamp (independent of the module under test)."""
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


def make_bars(dates, fill=None, overrides=None, drop=(), bar_minutes=1, early_close=(),
              extra_hours=False, tickers=("NVDA", "SPY")) -> pd.DataFrame:
    """Full regular-session bars for the given NY dates ('YYYY-MM-DD').

    ``fill``: {ticker: price or callable(date_str)}; ``overrides``/``drop`` are keyed by
    (ticker, 'YYYY-MM-DD HH:MM'), where the time is the bar END in New York time (the time the
    close is observed). Extra-hours rows have deliberately invalid-looking prices so tests can
    prove the production calendar does not select them.
    """
    fill = {"NVDA": 100.0, "SPY": 500.0, **(fill or {})}
    frames = []
    # Build full regular sessions first; overrides and drops target a unique local bar-end key.
    for t in tickers:
        for d in dates:
            last = "13:00" if d in early_close else "16:00"
            ends = pd.date_range(f"{d} 09:{30 + bar_minutes:02d}", f"{d} {last}",
                                 freq=f"{bar_minutes}min", tz=NY)
            base = fill[t](d) if callable(fill[t]) else fill[t]
            frames.append(pd.DataFrame({"ticker": t, "end_local": ends, "close": float(base)}))
            if extra_hours:  # extended-hours junk that must never be used
                pre = pd.date_range(f"{d} 08:00", f"{d} 09:30", freq=f"{bar_minutes}min", tz=NY)[1:]
                post = pd.date_range(f"{d} 16:00", f"{d} 17:00", freq=f"{bar_minutes}min", tz=NY)[1:]
                frames.append(pd.DataFrame({"ticker": t, "end_local": pre.append(post), "close": 1.0}))
    df = pd.concat(frames, ignore_index=True)
    key = df["ticker"] + "|" + df["end_local"].dt.strftime("%Y-%m-%d %H:%M")
    for (t, ts), px in (overrides or {}).items():
        m = key == f"{t}|{ts}"
        assert m.sum() == 1, f"override {(t, ts)} matched {m.sum()} bars"
        df.loc[m, "close"] = float(px)
    drop_keys = {f"{t}|{ts}" for t, ts in drop}
    assert drop_keys <= set(key), "drop key not found in generated bars"
    df = df[~key.isin(drop_keys)]
    end = df["end_local"].dt.tz_convert("UTC")
    out = pd.DataFrame({"ticker": df["ticker"], "bar_start_utc": end - pd.Timedelta(minutes=bar_minutes),
                        "bar_end_utc": end, "close": df["close"], "volume": 100})
    return out.reset_index(drop=True)


def make_news(*events, first_seen=True) -> pd.DataFrame:
    """Build articles from ``id`` plus NY ``at`` or UTC ``utc`` timestamps and optional fields.

    ``first_seen=True`` defaults availability to publication time; pass an explicit
    ``first_seen_at_utc`` to test delayed, missing, or anomalous vendor-observation times.
    """
    rows = []
    for e in events:
        e = dict(e)
        t = ny(e.pop("at")) if "at" in e else pd.Timestamp(e.pop("utc"), tz="UTC")
        row = {"article_id": e.pop("id"), "ticker": e.pop("ticker", "NVDA"), "published_at_utc": t, **e}
        if first_seen and "first_seen_at_utc" not in row:
            row["first_seen_at_utc"] = t
        rows.append(row)
    return pd.DataFrame(rows)
