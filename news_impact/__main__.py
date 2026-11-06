"""CLI for the event-study handoff: read article/bar inputs, build events, write outputs.

Optional news-classification columns are consumed only as metadata; the command does not invoke
the RAG pipeline and remains usable before those labels are available.
"""
from __future__ import annotations

import argparse

from . import build_event_dataset, load_news, load_prices, save_event_dataset, summarize_event_impacts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="news_impact", description=__doc__)
    ap.add_argument("--news", required=True, help="news metadata CSV/Parquet")
    ap.add_argument("--prices", required=True, help="minute-bar CSV/Parquet (ticker, bar_*_utc, close)")
    ap.add_argument("--out", required=True, help="output event dataset (.parquet or .csv)")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--horizons", type=int, nargs="+", default=[60, 390])
    ap.add_argument("--bar-minutes", type=int, default=None)
    ap.add_argument("--summary-out", default=None, help="optional CSV for overall summary")
    a = ap.parse_args(argv)

    news = load_news(a.news)
    # Include every news ticker plus the benchmark; the loader filters the market-data input
    # before the more expensive event-time matching work begins.
    tickers = set(news["ticker"].astype(str)) | {a.benchmark}
    prices = load_prices(a.prices, tickers=tickers)
    events = build_event_dataset(news, prices, horizons=tuple(a.horizons), benchmark=a.benchmark,
                                 bar_minutes=a.bar_minutes)
    save_event_dataset(events, a.out)
    # Usability is horizon-specific because a missing endpoint may invalidate only that window.
    n_ok = {h: int(events[f"usable_{h}m"].sum()) for h in a.horizons}
    print(f"wrote {len(events)} events to {a.out}; usable abnormal returns per horizon: {n_ok}")
    if a.summary_out:
        summarize_event_impacts(events).to_csv(a.summary_out, index=False)
        print(f"wrote summary to {a.summary_out}")


if __name__ == "__main__":
    main()
