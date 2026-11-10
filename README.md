# news_impact: historical news-to-price event study (NVDA vs SPY)

Builds an event-level dataset of forward returns after historical financial news, and summarizes
their distribution. Python module + tests; no notebook required. Method details: `METHODOLOGY.md`.

## Install and test

```bash
pip install -r requirements.txt        # pandas, numpy, scipy, pyarrow, matplotlib, pytest
python -m pytest -q                    # 75 tests, ~3 s
```

Python 3.10+; developed on 3.13 with pandas 3.

## Inputs

**News** (CSV/Parquet), one row per article:

| Column | Required | Notes |
|---|---|---|
| `article_id` | yes | unique id (duplicates are flagged, not dropped) |
| `ticker` | yes | e.g. `NVDA` |
| `published_at_utc` | yes | publisher time; naive = UTC, tz-aware converted |
| `first_seen_at_utc` | no | preferred over publish time when present |
| `headline`, `source`, `novelty_label`, `direction_label`, `event_type` | no | passed through; absent labels stay NA, never invented |

**Prices** (CSV/Parquet): `ticker`, `close`, and `bar_start_utc` and/or `bar_end_utc`. Needs every news
ticker plus the benchmark (default `SPY`). Closes must be split-consistent. Extended-hours bars are
ignored. Bar length is inferred when both time columns exist (1-minute and 5-minute both work).

## Run

```bash
python -m news_impact --news news.csv --prices bars.parquet --out events.parquet --summary-out summary.csv
```

```python
from news_impact import (load_news, load_prices, build_event_dataset, summarize_event_impacts, save_event_dataset)
news = load_news("news.csv")
prices = load_prices("bars.parquet", tickers={"NVDA", "SPY"})   # reads only what is needed
events = build_event_dataset(news, prices, horizons=(60, 390))
save_event_dataset(events, "events.parquet")
summarize_event_impacts(events, group_columns="novelty_label")   # once labels exist
```

Functions: `calculate_return`, `calculate_abnormal_return`, `match_news_to_prices`,
`build_event_dataset`, `summarize_event_impacts` (interface from the brief), plus `compare_groups`,
`NYSECalendar`, `check_calendar_against_prices`.

## Pipeline map and RAG handoff

- `news_impact/io.py` reads article metadata and filters market bars to the needed tickers/dates.
- `news_impact/core.py` aligns each article to post-availability bars and builds stock, benchmark,
  and abnormal returns; `article_id` is preserved as the join key.
- `news_impact/summary.py` describes outcomes overall or by supplied labels such as
  `novelty_label`.
- `news_impact/trading_calendar.py` defines regular-session slots used by event alignment.
- `news_impact/__main__.py` wires these steps into the command-line run.

This package is the historical-outcome layer for the forecasting/RAG project, not the RAG
implementation. It does not retrieve articles, generate novelty/direction/event labels, or predict
returns. A downstream classifier can join labels by `article_id` before `build_event_dataset`
(which passes supported label columns through), or join them to the resulting events afterward;
event construction and overall summaries work while labels are absent. Missing labels remain NA,
never guessed.

## Output (one row per input article, input order)

Units: returns are **fractions** (0.01 = 1%); timestamps are **UTC**; horizons are **regular-session
trading minutes**. Required columns: `article_id, ticker, event_timestamp_utc, entry_timestamp_utc,
novelty_label, direction_label, event_type, return_{60,390}m, market_return_{60,390}m,
abnormal_return_{60,390}m, overlapping_event_flag, data_quality_flag`.

Extras for validation and grouping: `headline, source, published_at_utc, first_seen_at_utc,
availability_source, release_session_state, entry_session_date, entry_bar_start_utc, entry_price,
market_entry_price, price_{h}m, market_price_{h}m, exit_timestamp_{h}m_utc, usable_{h}m,
n_overlapping_events, overlap_cluster_id`.

- Missing returns are NaN, never 0. Filter with `usable_{h}m`.
- `data_quality_flag` is `ok` or `; `-joined tokens (table in `METHODOLOGY.md`).
- For inference, group by `overlap_cluster_id` (or `entry_session_date`); do not treat overlapping events as independent.

## Known data-quality caveats

- Entry is the close of the first full bar after availability, so the first 0-2 minutes of reaction are excluded.
- All off-hours news for a session enters at the same next-open bar, so `shared_entry_bar` is common.
- The calendar is rule-based; special closures not in its list need `extra_holidays`.
- Split detection is a heuristic safeguard, not a substitute for adjusted prices.
- Tests use fabricated flat/override prices; no real market data or findings are in this repo.
