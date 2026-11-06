"""File boundaries for article metadata, market bars, and event-study outputs.

News labels are ordinary optional input columns: loading does not call a classifier, and the
event builder preserves absent labels as missing. Price loading can project/filter large files
before the core matching step.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path, **kw) -> pd.DataFrame:
    """Read a CSV or Parquet file based on its extension."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, **kw)


def load_news(path) -> pd.DataFrame:
    """Read article metadata (columns per README), including any supplied downstream labels."""
    return _read(path)


def load_prices(path, tickers=None, start=None, end=None) -> pd.DataFrame:
    """Read only the needed tickers/dates/columns from a CSV/Parquet price file.

    ``start``/``end`` filter on ``bar_end_utc`` (UTC). Only timestamp/close columns are retained;
    CSVs are read in chunks so a large file is never fully in memory. This limits input to the
    tickers and dates needed for the event study, not the whole raw market-data universe.
    """
    p = Path(path)
    keep = ["ticker", "bar_start_utc", "bar_end_utc", "close"]
    tick = set(tickers) if tickers is not None else None
    t0 = pd.Timestamp(start, tz="UTC") if start is not None else None
    t1 = pd.Timestamp(end, tz="UTC") if end is not None else None

    def filt(df: pd.DataFrame) -> pd.DataFrame:
        # Drop unrelated market-data columns before further filtering to keep memory bounded.
        df = df[[c for c in keep if c in df.columns]]
        if tick is not None:
            df = df[df["ticker"].isin(tick)]
        if t0 is not None or t1 is not None:
            col = "bar_end_utc" if "bar_end_utc" in df.columns else "bar_start_utc"
            ts = pd.to_datetime(df[col], utc=True)
            df = df[((ts >= t0) if t0 is not None else True) & ((ts <= t1) if t1 is not None else True)]
        return df

    if p.suffix.lower() in (".parquet", ".pq"):
        filters = [("ticker", "in", sorted(tick))] if tick is not None else None
        return filt(pd.read_parquet(p, filters=filters))
    return pd.concat([filt(c) for c in pd.read_csv(p, chunksize=1_000_000)], ignore_index=True)


def save_event_dataset(events: pd.DataFrame, path) -> Path:
    """Write the shared event-level handoff as Parquet or CSV, preserving UTC timestamps."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        events.to_parquet(p, index=False)
    else:
        events.to_csv(p, index=False)
    return p
