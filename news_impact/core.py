"""Event-study core: return arithmetic, news-to-price matching, event dataset construction.

Conventions (see METHODOLOGY.md for the full write-up)
------------------------------------------------------
* Inputs are UTC. Naive timestamps are *assumed* UTC; tz-aware ones are converted.
  New York time is used only to decide which regular session a timestamp falls in.
* A bar covers ``[bar_start_utc, bar_end_utc)`` and its ``close`` is the price observed at
  ``bar_end_utc``.
* Entry bar = first complete regular-session bar whose ``bar_start`` is >= the event
  availability time. ``P_t0`` is that bar's close, stamped ``entry_timestamp_utc`` = bar end.
* The forward horizon advances exactly ``h`` regular-session minutes along the calendar's
  trading-minute grid (overnight, weekends, holidays and after-hours excluded), so the
  end price is the close of the bar ending ``h`` trading minutes after the entry price.
* Horizons are exact for 1-minute bars. For coarser bars they must be multiples of the bar
  length (60 and 390 both work for 5-minute bars) so they stay exact, not approximated.
* Missing data are never imputed; they produce NaN plus an explicit ``data_quality_flag``.

This is the historical-outcome layer, not the RAG or prediction layer. Optional article
classifications (for example ``novelty_label``) are carried into the output when supplied;
timestamp matching and returns work without them, so downstream label generation can be joined
later by ``article_id``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trading_calendar import NY_TZ, NYSECalendar

NAT_NS = int(np.iinfo(np.int64).min)  # numpy's NaT as int64
NS_PER_MIN = 60_000_000_000
DEFAULT_HORIZONS = (60, 390)
SPLIT_FACTORS = (2, 3, 4, 5, 10, 20, 25, 50) # consider splits - for instance a 2-for-1 split instead of a 50% price collapse. 

# --------------------------------------------------------------------------------------
# Task 1: return arithmetic
# --------------------------------------------------------------------------------------


def _wrap(out: np.ndarray, like):
    if isinstance(like, pd.Series):
        return pd.Series(out, index=like.index)
    if out.ndim == 0:
        return float(out)
    return out


def calculate_return(start_price, end_price):
    """Return simple fractional price change ``(end - start) / start``.

    Works on scalars, arrays and Series. Result is a fraction: 0.01 means +1.0%.
    """
    s = np.asarray(start_price, dtype=float)
    e = np.asarray(end_price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(s) & np.isfinite(e) & (s > 0) & (e > 0)
        out = np.where(ok, (e - s) / np.where(s > 0, s, 1.0), np.nan)
    return _wrap(out, start_price if isinstance(start_price, pd.Series) else end_price)


def calculate_abnormal_return(stock_return, market_return):
    """Return stock return minus benchmark return (market-adjusted, fractions).

    NaN in either input gives NaN. This is a benchmark-adjusted association, not a causal
    estimate of the news effect.
    """
    s = np.asarray(stock_return, dtype=float)
    m = np.asarray(market_return, dtype=float)
    out = s - m
    return _wrap(out, stock_return if isinstance(stock_return, pd.Series) else market_return)


# --------------------------------------------------------------------------------------
# Timestamp helpers
# --------------------------------------------------------------------------------------


def _to_utc_ns(values) -> np.ndarray:
    """Parse to int64 UTC epoch ns. Naive -> assumed UTC. Unparseable/missing -> NAT_NS."""
    s = pd.Series(values).reset_index(drop=True)
    parsed = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return parsed.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").view("int64")


def _ns_to_utc(arr: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.asarray(arr, dtype="int64").view("datetime64[ns]")).tz_localize("UTC")


def _require_columns(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def _validate_horizons(horizons) -> tuple:
    hs = tuple(int(h) for h in horizons)
    if not hs or any(h <= 0 for h in hs) or len(set(hs)) != len(hs):
        raise ValueError(f"horizons must be distinct positive integers (minutes); got {horizons}")
    return hs


# --------------------------------------------------------------------------------------
# Market data preparation
# --------------------------------------------------------------------------------------


@dataclass
class _PriceBook:
    series: dict          # ticker -> pd.Series(close, index=bar_end_ns int64, sorted, unique)
    bar_minutes: int


def _prepare_prices(prices_df: pd.DataFrame, bar_minutes: int | None) -> _PriceBook:
    _require_columns(prices_df, ["ticker", "close"], "prices_df")
    n = len(prices_df)
    has_start = "bar_start_utc" in prices_df.columns
    has_end = "bar_end_utc" in prices_df.columns
    if not (has_start or has_end):
        raise ValueError("prices_df needs bar_start_utc and/or bar_end_utc")
    start = _to_utc_ns(prices_df["bar_start_utc"]) if has_start else None
    end = _to_utc_ns(prices_df["bar_end_utc"]) if has_end else None

    if has_start and has_end:
        # Keep only complete bars of one inferred frequency; an incomplete bar is not a valid
        # observation for either endpoint or for the trading-minute grid.
        both = (start != NAT_NS) & (end != NAT_NS)
        dur = (end - start)[both & (end > start)]
        inferred = int(round(np.median(dur) / NS_PER_MIN)) if len(dur) else 1
        if bar_minutes is None:
            bar_minutes = max(inferred, 1)
        elif bar_minutes != inferred and len(dur):
            raise ValueError(f"bar_minutes={bar_minutes} but bars in prices_df last ~{inferred} min")
        wrong = both & ((end - start) != bar_minutes * NS_PER_MIN)
        if wrong.any():
            warnings.warn(f"{int(wrong.sum())} price bar(s) are not exactly {bar_minutes} min long "
                          "and were dropped (only complete bars are used).", stacklevel=3)
        end = np.where(both & ~wrong, end, NAT_NS)
    else:
        bar_minutes = bar_minutes or 1
        end = end if has_end else np.where(start != NAT_NS, start + bar_minutes * NS_PER_MIN, NAT_NS)

    close = pd.to_numeric(prices_df["close"], errors="coerce").to_numpy(dtype=float)
    tick = prices_df["ticker"].astype(str).to_numpy()
    keep = (end != NAT_NS) & np.isfinite(close) & (close > 0)   # invalid price == missing bar
    df = pd.DataFrame({"ticker": tick[keep], "end": end[keep], "close": close[keep]})
    dup = df.duplicated(["ticker", "end"], keep=False)
    if dup.any():
        # Exact duplicate observations are harmless to collapse; conflicting closes make the
        # endpoint ambiguous and must be resolved by the data provider instead.
        conflict = df[dup].groupby(["ticker", "end"])["close"].nunique() > 1
        if conflict.any():
            raise ValueError(f"prices_df has {int(conflict.sum())} duplicated (ticker, bar) key(s) "
                             "with conflicting closes; resolve before building events.")
        df = df.drop_duplicates(["ticker", "end"])
    series = {}
    for t, sub in df.groupby("ticker", sort=False):
        series[t] = pd.Series(sub["close"].to_numpy(), index=sub["end"].to_numpy()).sort_index()
    return _PriceBook(series=series, bar_minutes=int(bar_minutes))


def _lookup(book: _PriceBook, tickers: np.ndarray, ns: np.ndarray) -> np.ndarray:
    # Exact timestamp lookup is intentional: never substitute a nearby bar for a missing one.
    out = np.full(len(ns), np.nan)
    for t in np.unique(tickers):
        s = book.series.get(t)
        if s is None:
            continue
        m = tickers == t
        out[m] = s.reindex(ns[m]).to_numpy(dtype=float)
    return out


# --------------------------------------------------------------------------------------
# Trading-minute grid
# --------------------------------------------------------------------------------------


@dataclass
class TradingGrid:
    """Every regular-session bar slot, concatenated across sessions (UTC ns)."""
    bar_minutes: int
    start_ns: np.ndarray
    end_ns: np.ndarray
    session_idx: np.ndarray
    sessions: pd.DataFrame
    session_first_pos: np.ndarray

    def __len__(self) -> int:
        return len(self.start_ns)


def build_trading_grid(calendar: NYSECalendar, start, end, bar_minutes: int = 1) -> TradingGrid:
    # Concatenating only regular-session slots lets an integer offset represent trading time:
    # stepping past a session close skips overnight, weekends, and holidays automatically.
    sess = calendar.sessions(start, end).reset_index(drop=True)
    bar_ns = bar_minutes * NS_PER_MIN
    length = sess["close_ns"].to_numpy() - sess["open_ns"].to_numpy()
    if len(sess) and (length % bar_ns).any():
        raise ValueError(f"bar_minutes={bar_minutes} does not divide every session length")
    n_bars = (length // bar_ns).astype("int64")
    total = int(n_bars.sum())
    first_pos = np.cumsum(n_bars) - n_bars
    sidx = np.repeat(np.arange(len(sess)), n_bars)
    k = np.arange(total) - np.repeat(first_pos, n_bars)
    start_ns = sess["open_ns"].to_numpy()[sidx] + k * bar_ns if total else np.empty(0, dtype="int64")
    return TradingGrid(bar_minutes=bar_minutes, start_ns=start_ns.astype("int64"),
                       end_ns=(start_ns + bar_ns).astype("int64"), session_idx=sidx,
                       sessions=sess, session_first_pos=first_pos)


# --------------------------------------------------------------------------------------
# Corporate-action checks
# --------------------------------------------------------------------------------------


def _detect_split_positions(book: _PriceBook, grid: TradingGrid, tickers, tol: float,
                            min_move: float) -> dict:
    """Grid positions where consecutive closes resemble a split ratio (a safeguard)."""
    cand = np.array([1.0 / k for k in SPLIT_FACTORS] + [float(k) for k in SPLIT_FACTORS])
    out = {}
    for t in tickers:
        s = book.series.get(t)
        if s is None or len(grid) == 0:
            continue
        vals = s.reindex(grid.end_ns).to_numpy(dtype=float)
        idx = np.flatnonzero(~np.isnan(vals))
        if len(idx) < 2:
            continue
        v = vals[idx]
        r = v[1:] / v[:-1]
        big = np.abs(r - 1.0) >= min_move
        hit = np.zeros(len(r), dtype=bool)
        if big.any():
            hit[big] = (np.abs(r[big][:, None] / cand[None, :] - 1.0).min(axis=1) < tol)
        if hit.any():
            out[t] = idx[1:][hit]
    return out


def _corporate_action_positions(actions: pd.DataFrame | None, grid: TradingGrid) -> dict:
    out: dict = {}
    if actions is None or len(actions) == 0 or len(grid.sessions) == 0:
        return out
    _require_columns(actions, ["ticker", "effective_date"], "corporate_actions")
    sd = grid.sessions["session_date"].to_numpy(dtype="datetime64[ns]").view("int64")
    for t, sub in actions.groupby(actions["ticker"].astype(str)):
        eff = pd.to_datetime(sub["effective_date"]).dt.tz_localize(None).dt.normalize()
        # Map an effective date to that session's first grid slot, or the next session if the
        # date is closed, so window checks use the same positions as return endpoints.
        k = np.searchsorted(sd, eff.to_numpy(dtype="datetime64[ns]").view("int64"), side="left")
        k = k[k < len(sd)]
        if len(k):
            out[t] = np.sort(grid.session_first_pos[k])
    return out


def _spans(positions: np.ndarray | None, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """True where some position s satisfies p < s <= q."""
    if positions is None or len(positions) == 0:
        return np.zeros(len(p), dtype=bool)
    return (np.searchsorted(positions, q, side="right") - np.searchsorted(positions, p, side="right")) > 0


# --------------------------------------------------------------------------------------
# Overlap / duplicate coverage
# --------------------------------------------------------------------------------------


def _overlap_info(pos: np.ndarray, tickers: np.ndarray, max_steps: int):
    """Overlap of return windows, counted exactly on the trading-minute grid.

    Return intervals (p, p+H] of two events intersect iff |p_i - p_j| < H.
    """
    n = len(pos)
    n_over = np.zeros(n, dtype=int)
    shared = np.zeros(n, dtype=bool)
    cluster = np.full(n, None, dtype=object)
    for t in np.unique(tickers):
        idx = np.flatnonzero((pos >= 0) & (tickers == t))
        if len(idx) == 0:
            continue
        order = np.argsort(pos[idx], kind="stable")
        ids = idx[order]
        sp = pos[ids]
        # Search the inclusive integer-position range that satisfies |p_i - p_j| < H;
        # identical entry positions count as overlapping articles as well.
        lo = np.searchsorted(sp, sp - max_steps + 1, side="left")
        hi = np.searchsorted(sp, sp + max_steps - 1, side="right")
        n_over[ids] = hi - lo - 1
        # Connected components chain consecutive windows that overlap; this is the grouping
        # key for dependence-aware summaries, not a claim that every pair in a cluster overlaps.
        cid = np.cumsum(np.concatenate([[0], (np.diff(sp) >= max_steps).astype(int)])) + 1
        cluster[ids] = [f"{t}_{c:05d}" for c in cid]
        _, inv, cnt = np.unique(sp, return_inverse=True, return_counts=True)
        shared[ids] = cnt[inv] > 1
    return n_over, shared, cluster


# --------------------------------------------------------------------------------------
# Matching + dataset construction
# --------------------------------------------------------------------------------------


def _missing_reason(book: _PriceBook, tickers: np.ndarray, ns: np.ndarray, price: np.ndarray,
                    eligible: np.ndarray):
    """Classify a missing exact bar as absent ticker, outside file coverage, or an internal gap."""
    for i in np.flatnonzero(eligible & np.isnan(price)):
        s = book.series.get(tickers[i])
        if s is None or len(s) == 0:
            yield i, "no_price_data"
        elif ns[i] > s.index[-1]:
            yield i, "beyond_price_data"
        elif ns[i] < s.index[0]:
            yield i, "before_price_data"
        else:
            yield i, "missing_bar"


def _run(news_df, prices_df, horizons, benchmark, calendar, bar_minutes, delay_threshold_minutes,
         corporate_actions, detect_splits, split_tolerance, split_min_move) -> dict:
    """Build shared intermediate event state used by both public alignment functions.

    The sequence is deliberate: normalize availability, map it to a regular-session grid,
    retrieve exact stock/benchmark closes, then calculate horizons and attach quality flags.
    """
    horizons = _validate_horizons(horizons)
    _require_columns(news_df, ["article_id", "ticker", "published_at_utc"], "news_df")
    news = news_df.reset_index(drop=True)
    n = len(news)
    book = _prepare_prices(prices_df, bar_minutes)
    bm = book.bar_minutes
    steps = {}
    for h in horizons:
        if h % bm:
            raise ValueError(f"horizon {h} min is not a multiple of the {bm}-min bar length; "
                             "horizons would be approximated, which this module does not do silently.")
        steps[h] = h // bm
    max_steps = max(steps.values())
    cal = calendar or NYSECalendar()
    flags = [[] for _ in range(n)]

    def add(mask, token):
        for i in np.flatnonzero(mask):
            flags[i].append(token)

    # ---- availability timestamp: use first-seen when available, otherwise publisher time ----
    ticker = news["ticker"].astype(str).to_numpy()
    pub = _to_utc_ns(news["published_at_utc"])
    seen = _to_utc_ns(news["first_seen_at_utc"]) if "first_seen_at_utc" in news.columns \
        else np.full(n, NAT_NS, dtype="int64")
    pub_ok, seen_ok = pub != NAT_NS, seen != NAT_NS
    avail = np.where(seen_ok, seen, pub)
    valid = avail != NAT_NS
    source = np.where(seen_ok, "first_seen", np.where(pub_ok, "published", None))
    add(~valid, "missing_timestamp")
    add(valid & ~seen_ok, "publisher_time_only")
    add(seen_ok & ~pub_ok, "published_time_missing")
    add(seen_ok & pub_ok & (seen < pub), "first_seen_before_published")
    add(seen_ok & pub_ok & ((seen - pub) > delay_threshold_minutes * NS_PER_MIN), "delayed_availability")

    # ---- trading-minute grid: pad the news-date range to cover weekends and forward horizons ----
    if valid.any():
        lo = pd.Timestamp(int(avail[valid].min()), unit="ns").normalize() - pd.Timedelta(days=7)
        hi = pd.Timestamp(int(avail[valid].max()), unit="ns").normalize() + \
            pd.Timedelta(days=14 + 7 * math.ceil(max(horizons) / 390))
    else:
        lo, hi = pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-01")
    grid = build_trading_grid(cal, lo, hi, bm)
    G, S = len(grid), len(grid.sessions)

    # ---- release state and entry bar: search by bar start so no pre-release bar is eligible ----
    pos = np.full(n, -1, dtype="int64")
    if G:
        p = np.searchsorted(grid.start_ns, avail, side="left")
        ok = valid & (p < G)
        pos[ok] = p[ok]
    add(valid & (pos < 0), "no_session_after_release")
    posc = np.clip(pos, 0, max(G - 1, 0))

    state = np.where(valid, "non_trading_day", "unknown").astype(object)
    in_sess = np.zeros(n, dtype=bool)
    j = np.full(n, -1)
    if S:
        open_ns = grid.sessions["open_ns"].to_numpy()
        close_ns = grid.sessions["close_ns"].to_numpy()
        j = np.searchsorted(open_ns, avail, side="right") - 1
        in_sess = valid & (j >= 0) & (avail < close_ns[np.clip(j, 0, S - 1)])
        local_day = pd.DatetimeIndex(np.where(valid, avail, 0).view("datetime64[ns]")) \
            .tz_localize("UTC").tz_convert(NY_TZ).tz_localize(None).normalize()
        day_ns = local_day.as_unit("ns").asi8
        sd = grid.sessions["session_date"].to_numpy(dtype="datetime64[ns]").view("int64")
        k = np.searchsorted(sd, day_ns)
        kc = np.clip(k, 0, S - 1)
        trading_day = valid & (k < S) & (sd[kc] == day_ns)
        state = np.select(
            [~valid, in_sess, trading_day & (avail < open_ns[kc]), trading_day & (avail >= close_ns[kc])],
            ["unknown", "intraday", "pre_open", "post_close"], default="non_trading_day").astype(object)
    add(valid & (state != "intraday"), "off_hours_release")
    entry_sess = np.where(pos >= 0, grid.session_idx[posc], -1) if G else np.full(n, -1)
    add(in_sess & (pos >= 0) & (entry_sess != j), "late_session_rollover")

    has_pos = pos >= 0
    entry_start = np.where(has_pos, grid.start_ns[posc], NAT_NS) if G else np.full(n, NAT_NS)
    entry_end = np.where(has_pos, grid.end_ns[posc], NAT_NS) if G else np.full(n, NAT_NS)
    if S:
        sdates = grid.sessions["session_date"].to_numpy(dtype="datetime64[ns]")
        entry_date = np.where(entry_sess >= 0, sdates[np.clip(entry_sess, 0, S - 1)],
                              np.datetime64("NaT", "ns"))
    else:
        entry_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    bench = np.full(n, benchmark, dtype=object)
    # Stock and benchmark are observed at identical bar ends; a missing side stays NaN and is
    # reported separately rather than shifting one instrument to a different timestamp.
    entry_px = _lookup(book, ticker, entry_end)
    entry_mpx = _lookup(book, bench, entry_end)
    for arr, tk, who in ((entry_px, ticker, "stock"), (entry_mpx, bench, "market")):
        for i, why in _missing_reason(book, tk, entry_end, arr, has_pos):
            flags[i].append(f"{why}_entry_{who}")

    # ---- corporate actions ----
    uniq_t = sorted(set(ticker.tolist()) | {benchmark})
    split_pos = _detect_split_positions(book, grid, uniq_t, split_tolerance, split_min_move) \
        if detect_splits else {}
    action_pos = _corporate_action_positions(corporate_actions, grid)

    def spans_by_ticker(table, tk, p, q):
        out = np.zeros(n, dtype=bool)
        for t in np.unique(tk):
            m = tk == t
            out[m] = _spans(table.get(t), p[m], q[m])
        return out

    # ---- forward horizons: grid offsets count regular-session bars, not wall-clock minutes ----
    res = {}
    for h in horizons:
        q = pos + steps[h]
        qvalid = has_pos & (q < G)
        qc = np.clip(q, 0, max(G - 1, 0))
        end_ts = np.where(qvalid, grid.end_ns[qc], NAT_NS) if G else np.full(n, NAT_NS)
        add(has_pos & ~qvalid, f"beyond_calendar_end_{h}m")
        px = _lookup(book, ticker, end_ts)
        mpx = _lookup(book, bench, end_ts)
        for arr, tk, who in ((px, ticker, "stock"), (mpx, bench, "market")):
            for i, why in _missing_reason(book, tk, end_ts, arr, qvalid):
                flags[i].append(f"{why}_end_{h}m_{who}")
        r_s = np.asarray(calculate_return(entry_px, px))
        r_m = np.asarray(calculate_return(entry_mpx, mpx))
        for who, tk, r in (("stock", ticker, r_s), ("market", bench, r_m)):
            bad = spans_by_ticker(split_pos, tk, pos, q) & has_pos & qvalid
            add(bad, f"suspected_unadjusted_split_{h}m_{who}")
            r[bad] = np.nan                       # an artificial discontinuity is not a return
            add(spans_by_ticker(action_pos, tk, pos, q) & has_pos & qvalid,
                f"corporate_action_in_window_{h}m_{who}")
        res[h] = dict(end_ts=end_ts, px=px, mpx=mpx, r=r_s, mr=r_m,
                      ar=np.asarray(calculate_abnormal_return(r_s, r_m)))

    # ---- quality/context metadata used by downstream filtering and inference ----
    if S and G:
        early = grid.sessions["early_close"].to_numpy().astype(int)
        cum = np.cumsum(early)
        q_max = np.clip(pos + max_steps, 0, G - 1)
        s0, s1 = entry_sess.clip(0), grid.session_idx[q_max]
        touches = has_pos & ((cum[s1] - cum[s0] + early[s0]) > 0)
        add(touches, "early_close_in_window")
    n_over, shared, cluster = _overlap_info(pos, ticker, max_steps)
    add(shared, "shared_entry_bar")
    ids = news["article_id"]
    add(ids.duplicated(keep=False).to_numpy(), "duplicate_article_id")

    quality = ["; ".join(dict.fromkeys(f)) or "ok" for f in flags]
    return dict(news=news, n=n, ticker=ticker, benchmark=benchmark, horizons=horizons, bar_minutes=bm,
                pub=pub, seen=seen, avail=avail, source=source, state=state, entry_date=entry_date,
                entry_start=entry_start, entry_end=entry_end, entry_px=entry_px, entry_mpx=entry_mpx,
                res=res, overlap=(n_over > 0) | shared, n_over=n_over, cluster=cluster, quality=quality,
                pos=pos)


_OPTIONS = dict(benchmark="SPY", calendar=None, bar_minutes=None, delay_threshold_minutes=60,
                corporate_actions=None, detect_splits=True, split_tolerance=0.05, split_min_move=0.35)


def match_news_to_prices(news_df: pd.DataFrame, prices_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Align news with first eligible post-news market bars.

    One output row per input row (order preserved). Columns: ``article_id``, ``ticker``,
    ``event_timestamp_utc``, ``availability_source``, ``release_session_state``
    (intraday / pre_open / post_close / non_trading_day), ``entry_session_date``,
    ``entry_bar_start_utc``, ``entry_timestamp_utc`` (bar end = time ``P_t0`` is observed),
    ``entry_price``, ``market_entry_price``, ``data_quality_flag``.

    This is the alignment-only view for callers that need entry timestamps/prices before
    computing or joining later-stage features. Keyword options are those of
    :func:`build_event_dataset` (``benchmark``, ``calendar``, ``bar_minutes``, ``horizons`` ...);
    horizons only size the calendar grid here.
    """
    opts = {**_OPTIONS, "horizons": DEFAULT_HORIZONS, **kwargs}
    r = _run(news_df, prices_df, **opts)
    return pd.DataFrame({
        "article_id": r["news"]["article_id"].to_numpy(),
        "ticker": r["ticker"],
        "event_timestamp_utc": _ns_to_utc(r["avail"]),
        "availability_source": pd.array(r["source"], dtype="string"),
        "release_session_state": pd.array(r["state"], dtype="string"),
        "entry_session_date": r["entry_date"],
        "entry_bar_start_utc": _ns_to_utc(r["entry_start"]),
        "entry_timestamp_utc": _ns_to_utc(r["entry_end"]),
        "entry_price": r["entry_px"],
        "market_entry_price": r["entry_mpx"],
        "data_quality_flag": r["quality"],
    })


def _opt_col(news: pd.DataFrame, name: str) -> pd.Series:
    if name in news.columns:
        return news[name].astype("string").reset_index(drop=True)
    return pd.Series(pd.NA, index=range(len(news)), dtype="string")


def build_event_dataset(news_df: pd.DataFrame, prices_df: pd.DataFrame,
                        horizons=DEFAULT_HORIZONS, *, benchmark: str = "SPY",
                        calendar: NYSECalendar | None = None, bar_minutes: int | None = None,
                        delay_threshold_minutes: float = 60, corporate_actions: pd.DataFrame | None = None,
                        detect_splits: bool = True, split_tolerance: float = 0.05,
                        split_min_move: float = 0.35) -> pd.DataFrame:
    """Construct the event-level dataset of forward returns for trading-minute horizons.

    Parameters
    ----------
    news_df : required ``article_id``, ``ticker``, ``published_at_utc``; optional
        ``first_seen_at_utc`` (preferred for entry when present), ``headline``, ``source``,
        ``novelty_label``, ``direction_label``, ``event_type``. These labels may come from a
        separate RAG/classification step; absent labels stay NA and do not block event building.
        Join upstream/downstream on ``article_id`` rather than generating labels in this module.
    prices_df : ``ticker``, ``close`` and ``bar_start_utc`` and/or ``bar_end_utc``
        (``volume`` ignored). Must contain every news ticker and ``benchmark``. Bars outside
        regular hours are ignored. Prices must be split/dividend-consistent.
    horizons : forward horizons in regular-session minutes (multiples of the bar length).
    benchmark : benchmark ticker (default SPY), read at the same timestamps as the stock.
    calendar : :class:`NYSECalendar`; default is the built-in NYSE rules.
    bar_minutes : bar length; inferred from ``bar_end - bar_start`` when both columns exist.
    delay_threshold_minutes : ``first_seen - published`` above this is flagged.
    corporate_actions : optional DataFrame (``ticker``, ``effective_date``); windows spanning
        an action get an informational flag.
    detect_splits : flag windows spanning an unadjusted split-like jump and set that
        component's return to NaN.

    Returns one row per input row in input order. Missing returns are NaN (never 0) and the
    reason is in ``data_quality_flag`` ("ok" when clean; otherwise ``"; "``-joined tokens).
    Returns are fractions (0.01 = +1%). All timestamps are UTC.
    """
    r = _run(news_df, prices_df, horizons, benchmark, calendar, bar_minutes, delay_threshold_minutes,
             corporate_actions, detect_splits, split_tolerance, split_min_move)
    news, hs = r["news"], r["horizons"]
    cols: dict = {
        "article_id": news["article_id"].to_numpy(),
        "ticker": r["ticker"],
        "event_timestamp_utc": _ns_to_utc(r["avail"]),
        "entry_timestamp_utc": _ns_to_utc(r["entry_end"]),
        "novelty_label": _opt_col(news, "novelty_label"),
        "direction_label": _opt_col(news, "direction_label"),
        "event_type": _opt_col(news, "event_type"),
    }
    for h in hs:
        cols[f"return_{h}m"] = r["res"][h]["r"]
    for h in hs:
        cols[f"market_return_{h}m"] = r["res"][h]["mr"]
    for h in hs:
        cols[f"abnormal_return_{h}m"] = r["res"][h]["ar"]
    cols["overlapping_event_flag"] = r["overlap"]
    cols["data_quality_flag"] = r["quality"]
    # ---- passthrough and traceability fields for validation, grouping, and downstream joins ----
    cols["headline"] = _opt_col(news, "headline")
    cols["source"] = _opt_col(news, "source")
    cols["published_at_utc"] = _ns_to_utc(r["pub"])
    cols["first_seen_at_utc"] = _ns_to_utc(r["seen"])
    cols["availability_source"] = pd.array(r["source"], dtype="string")
    cols["release_session_state"] = pd.array(r["state"], dtype="string")
    cols["entry_session_date"] = r["entry_date"]
    cols["entry_bar_start_utc"] = _ns_to_utc(r["entry_start"])
    cols["entry_price"] = r["entry_px"]
    cols["market_entry_price"] = r["entry_mpx"]
    for h in hs:
        cols[f"price_{h}m"] = r["res"][h]["px"]
        cols[f"market_price_{h}m"] = r["res"][h]["mpx"]
        cols[f"exit_timestamp_{h}m_utc"] = _ns_to_utc(r["res"][h]["end_ts"])
        cols[f"usable_{h}m"] = ~np.isnan(r["res"][h]["ar"])
    cols["n_overlapping_events"] = r["n_over"]
    cols["overlap_cluster_id"] = pd.array(r["cluster"], dtype="string")
    out = pd.DataFrame(cols)
    out.attrs["metadata"] = dict(horizons=hs, benchmark=benchmark, bar_minutes=r["bar_minutes"],
                                 entry_convention="close of first complete bar with bar_start >= availability",
                                 horizon_convention="exact regular-session trading minutes along bar ends")
    return out
