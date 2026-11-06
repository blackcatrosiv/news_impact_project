"""Event alignment and dataset construction (QA cases 1-7 of the brief, plus extras)."""
import numpy as np
import pandas as pd
import pytest

from helpers import make_bars, make_news, ny
from news_impact import build_event_dataset, match_news_to_prices, save_event_dataset

REQUIRED = ["article_id", "ticker", "event_timestamp_utc", "entry_timestamp_utc", "novelty_label",
            "direction_label", "event_type", "return_60m", "return_390m", "market_return_60m",
            "market_return_390m", "abnormal_return_60m", "abnormal_return_390m",
            "overlapping_event_flag", "data_quality_flag"]

D3 = ["2026-11-04", "2026-11-05", "2026-11-06"]
EXAMPLE = {  # the brief's worked example: entry 10:02 ET (bar 10:01-10:02 after 10:00:30 news)
    ("NVDA", "2026-11-04 10:02"): 100, ("SPY", "2026-11-04 10:02"): 500,
    ("NVDA", "2026-11-04 11:02"): 101, ("SPY", "2026-11-04 11:02"): 502,
    ("NVDA", "2026-11-05 10:02"): 102, ("SPY", "2026-11-05 10:02"): 504,
}


def one(events: pd.DataFrame) -> pd.Series:
    assert len(events) == 1
    return events.iloc[0]


# ---- 1. ordinary intraday article -------------------------------------------------------

def test_intraday_article_reproduces_worked_example():
    news = make_news({"id": "NVDA_001", "at": "2026-11-04 10:00:30"})
    ev = one(build_event_dataset(news, make_bars(D3, overrides=EXAMPLE)))
    assert ev["entry_timestamp_utc"] == ny("2026-11-04 10:02")
    assert ev["exit_timestamp_60m_utc"] == ny("2026-11-04 11:02")
    assert ev["exit_timestamp_390m_utc"] == ny("2026-11-05 10:02")     # next session, same clock minute
    assert ev["return_60m"] == pytest.approx(0.010)
    assert ev["market_return_60m"] == pytest.approx(0.004)
    assert ev["abnormal_return_60m"] == pytest.approx(0.006)
    assert ev["return_390m"] == pytest.approx(0.020)
    assert ev["market_return_390m"] == pytest.approx(0.008)
    assert ev["abnormal_return_390m"] == pytest.approx(0.012)
    assert ev["data_quality_flag"] == "ok"
    assert ev["release_session_state"] == "intraday" and ev["availability_source"] == "first_seen"


@pytest.mark.parametrize("at,entry_end,state", [
    ("2026-11-04 10:00:30", "2026-11-04 10:02", "intraday"),      # mid-bar news: 10:00 bar is contaminated
    ("2026-11-04 10:00:00", "2026-11-04 10:01", "intraday"),      # exactly on a bar start: that bar is eligible
    ("2026-11-04 09:30:00", "2026-11-04 09:31", "intraday"),      # open
    ("2026-11-04 09:29:59", "2026-11-04 09:31", "pre_open"),
])
def test_entry_is_first_complete_bar_starting_at_or_after_availability(at, entry_end, state):
    ev = one(build_event_dataset(make_news({"id": "a", "at": at}), make_bars(D3)))
    assert ev["entry_timestamp_utc"] == ny(entry_end)
    assert ev["entry_bar_start_utc"] == ny(entry_end) - pd.Timedelta(minutes=1)
    assert ev["release_session_state"] == state


def test_entry_never_uses_a_bar_that_started_before_availability():
    news = make_news(*[{"id": f"a{s}", "at": f"2026-11-04 10:00:{s:02d}"} for s in (0, 1, 30, 59)])
    ev = build_event_dataset(news, make_bars(D3))
    assert (ev["entry_bar_start_utc"] >= ev["event_timestamp_utc"]).all()
    assert (ev["entry_timestamp_utc"] > ev["event_timestamp_utc"]).all()


# ---- 2. after-hours / pre-open --------------------------------------------------------------

def test_after_hours_enters_next_session_and_excludes_overnight_gap():
    over = {("NVDA", "2026-11-04 15:59"): 100,      # last pre-news close
            ("NVDA", "2026-11-05 09:31"): 110,      # next-day entry price, after a +10% overnight gap
            ("NVDA", "2026-11-05 10:31"): 111}      # entry + 60 trading minutes
    news = make_news({"id": "ah", "at": "2026-11-04 17:00:00"})
    ev = one(build_event_dataset(news, make_bars(D3, overrides=over, extra_hours=True)))
    assert ev["release_session_state"] == "post_close"
    assert ev["entry_timestamp_utc"] == ny("2026-11-05 09:31")
    assert ev["entry_price"] == 110
    assert ev["return_60m"] == pytest.approx(111 / 110 - 1)
    assert ev["return_60m"] != pytest.approx(111 / 100 - 1)           # gap is not credited to the investor
    assert "off_hours_release" in ev["data_quality_flag"]


def test_news_in_last_minute_rolls_to_next_session():
    ev = one(build_event_dataset(make_news({"id": "x", "at": "2026-11-04 15:59:30"}), make_bars(D3)))
    assert ev["release_session_state"] == "intraday"
    assert ev["entry_timestamp_utc"] == ny("2026-11-05 09:31")
    assert "late_session_rollover" in ev["data_quality_flag"]


def test_pre_open_news_enters_at_first_bar_of_same_day():
    ev = one(build_event_dataset(make_news({"id": "x", "at": "2026-11-04 08:00:00"}),
                                 make_bars(D3, extra_hours=True)))
    assert ev["release_session_state"] == "pre_open"
    assert ev["entry_timestamp_utc"] == ny("2026-11-04 09:31")


# ---- 3. weekend / holiday -----------------------------------------------------------------

def test_weekend_news_enters_monday():
    dates = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]
    ev = one(build_event_dataset(make_news({"id": "w", "at": "2026-11-07 12:00:00"}), make_bars(dates)))
    assert ev["release_session_state"] == "non_trading_day"
    assert ev["entry_timestamp_utc"] == ny("2026-11-09 09:31")
    assert ev["entry_session_date"] == pd.Timestamp("2026-11-09")


def test_holiday_news_and_early_close_window():
    """Thanksgiving news -> Fri Nov 27 (13:00 close) -> 390 trading minutes ends Mon Nov 30 12:31."""
    dates = ["2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01"]
    over = {("NVDA", "2026-11-27 09:31"): 100, ("SPY", "2026-11-27 09:31"): 500,
            ("NVDA", "2026-11-30 12:31"): 105, ("SPY", "2026-11-30 12:31"): 505}
    bars = make_bars(dates, overrides=over, early_close={"2026-11-27"})
    ev = one(build_event_dataset(make_news({"id": "tg", "at": "2026-11-26 12:00:00"}), bars))
    assert ev["release_session_state"] == "non_trading_day"
    assert ev["entry_timestamp_utc"] == ny("2026-11-27 09:31")
    assert ev["exit_timestamp_390m_utc"] == ny("2026-11-30 12:31")
    assert ev["return_390m"] == pytest.approx(0.05)
    assert ev["abnormal_return_390m"] == pytest.approx(0.05 - 0.01)
    assert "early_close_in_window" in ev["data_quality_flag"]


# ---- 4. missing observations --------------------------------------------------------------

def test_missing_benchmark_endpoint_is_nan_with_flag_not_zero():
    bars = make_bars(D3, overrides=EXAMPLE, drop=[("SPY", "2026-11-04 11:02")])
    ev = one(build_event_dataset(make_news({"id": "m", "at": "2026-11-04 10:00:30"}), bars))
    assert ev["return_60m"] == pytest.approx(0.01)                      # stock side still fine
    assert np.isnan(ev["market_return_60m"]) and np.isnan(ev["abnormal_return_60m"])
    assert not ev["usable_60m"] and ev["usable_390m"]
    assert ev["abnormal_return_390m"] == pytest.approx(0.012)           # other horizon unaffected
    assert "missing_bar_end_60m_market" in ev["data_quality_flag"]


def test_missing_entry_bar_blanks_every_horizon_for_that_ticker():
    bars = make_bars(D3, overrides=EXAMPLE, drop=[("NVDA", "2026-11-04 10:02")])
    ev = one(build_event_dataset(make_news({"id": "m", "at": "2026-11-04 10:00:30"}), bars))
    assert np.isnan(ev["return_60m"]) and np.isnan(ev["return_390m"])
    assert np.isnan(ev["abnormal_return_60m"]) and np.isnan(ev["abnormal_return_390m"])
    assert ev["market_return_60m"] == pytest.approx(0.004)
    assert "missing_bar_entry_stock" in ev["data_quality_flag"]


def test_window_running_past_price_data_is_flagged():
    ev = one(build_event_dataset(make_news({"id": "e", "at": "2026-11-04 10:00:30"}),
                                 make_bars(["2026-11-04"], overrides={k: v for k, v in EXAMPLE.items() if "11-04" in k[1]})))
    assert ev["return_60m"] == pytest.approx(0.01)
    assert np.isnan(ev["return_390m"])
    assert "beyond_price_data_end_390m_stock" in ev["data_quality_flag"]


def test_unknown_ticker_and_missing_timestamp_rows_are_kept_as_nan():
    news = make_news({"id": "ok", "at": "2026-11-04 10:00:30"},
                     {"id": "noprice", "at": "2026-11-04 10:00:30", "ticker": "AAPL"},
                     {"id": "nots", "utc": "2026-11-04 15:00:30", "first_seen_at_utc": None})
    news.loc[2, "published_at_utc"] = pd.NaT
    ev = build_event_dataset(news, make_bars(D3))
    assert list(ev["article_id"]) == ["ok", "noprice", "nots"]
    assert "no_price_data_entry_stock" in ev.loc[1, "data_quality_flag"]
    assert ev.loc[2, "data_quality_flag"].startswith("missing_timestamp")
    assert ev.loc[[1, 2], ["return_60m", "abnormal_return_390m"]].isna().all().all()
    assert ev.loc[2, "entry_timestamp_utc"] is pd.NaT


# ---- 5. repeated coverage ---------------------------------------------------------------

def test_duplicate_and_overlapping_coverage_is_flagged_and_ids_kept():
    dates = [f"2026-11-{d:02d}" for d in (4, 5, 6, 9, 10, 11, 12, 13)]
    news = make_news({"id": "A", "at": "2026-11-04 10:00:30"},
                     {"id": "B", "at": "2026-11-04 10:00:45"},       # same entry bar as A
                     {"id": "C", "at": "2026-11-04 10:30:00"},       # inside A's window
                     {"id": "D", "at": "2026-11-10 10:00:00"})       # days later
    ev = build_event_dataset(news, make_bars(dates)).set_index("article_id")
    assert ev.loc[["A", "B", "C"], "overlapping_event_flag"].all()
    assert not ev.loc["D", "overlapping_event_flag"]
    assert "shared_entry_bar" in ev.loc["A", "data_quality_flag"] and "shared_entry_bar" in ev.loc["B", "data_quality_flag"]
    assert "shared_entry_bar" not in ev.loc["C", "data_quality_flag"]
    assert ev.loc["A", "n_overlapping_events"] == 2
    assert ev.loc["A", "overlap_cluster_id"] == ev.loc["B", "overlap_cluster_id"] == ev.loc["C", "overlap_cluster_id"]
    assert ev.loc["D", "overlap_cluster_id"] != ev.loc["A", "overlap_cluster_id"]


def test_windows_that_only_touch_are_not_overlapping():
    # entries exactly 390 trading minutes apart: return intervals (p, p+390] do not intersect
    news = make_news({"id": "E", "at": "2026-11-04 10:00:30"}, {"id": "F", "at": "2026-11-05 10:00:30"})
    ev = build_event_dataset(news, make_bars(D3 + ["2026-11-09"]))
    assert not ev["overlapping_event_flag"].any()
    assert ev["overlap_cluster_id"].nunique() == 2
    news = make_news({"id": "E", "at": "2026-11-04 10:00:30"}, {"id": "G", "at": "2026-11-05 10:00:00"})
    ev = build_event_dataset(news, make_bars(D3 + ["2026-11-09"]))      # G enters one minute earlier
    assert ev["overlapping_event_flag"].all()


def test_duplicate_article_ids_are_flagged_not_dropped():
    news = make_news({"id": "A", "at": "2026-11-04 10:00:30"}, {"id": "A", "at": "2026-11-05 10:00:30"})
    ev = build_event_dataset(news, make_bars(D3 + ["2026-11-09"]))
    assert len(ev) == 2 and ev["data_quality_flag"].str.contains("duplicate_article_id").all()


# ---- 6. timezones ------------------------------------------------------------------------

@pytest.mark.parametrize("pub_utc,dates,entry_ny,exit390_ny", [
    ("2026-03-06T15:00:30Z", ["2026-03-06", "2026-03-09", "2026-03-10"], "2026-03-06 10:02", "2026-03-09 10:02"),  # EST->EDT
    ("2026-03-09T14:00:30Z", ["2026-03-09", "2026-03-10", "2026-03-11"], "2026-03-09 10:02", "2026-03-10 10:02"),  # EDT
    ("2026-10-30T14:00:30Z", ["2026-10-30", "2026-11-02", "2026-11-03"], "2026-10-30 10:02", "2026-11-02 10:02"),  # EDT->EST
])
def test_dst_changes_use_new_york_session_logic_but_store_utc(pub_utc, dates, entry_ny, exit390_ny):
    news = make_news({"id": "dst", "utc": pub_utc})
    ev = one(build_event_dataset(news, make_bars(dates)))
    assert ev["entry_timestamp_utc"] == ny(entry_ny)
    assert ev["exit_timestamp_390m_utc"] == ny(exit390_ny)
    assert str(ev["entry_timestamp_utc"].tz) == "UTC"


def test_timezone_inputs_naive_is_utc_aware_is_converted():
    bars = make_bars(D3)
    a = build_event_dataset(pd.DataFrame({"article_id": ["a"], "ticker": ["NVDA"],
                                          "published_at_utc": ["2026-11-04T10:00:30-05:00"]}), bars)
    b = build_event_dataset(pd.DataFrame({"article_id": ["a"], "ticker": ["NVDA"],
                                          "published_at_utc": ["2026-11-04 15:00:30"]}), bars)       # naive == UTC
    assert a.loc[0, "entry_timestamp_utc"] == b.loc[0, "entry_timestamp_utc"] == ny("2026-11-04 10:02")


# ---- 7. corporate actions ---------------------------------------------------------------

def test_unadjusted_split_does_not_become_a_return():
    fill = {"NVDA": lambda d: 1000.0 if d == "2026-11-04" else 100.0}     # 10:1 split, prices NOT adjusted
    news = make_news({"id": "pre", "at": "2026-11-04 10:00:30"}, {"id": "post", "at": "2026-11-05 10:00:30"})
    ev = build_event_dataset(news, make_bars(D3 + ["2026-11-09"], fill=fill)).set_index("article_id")
    assert np.isnan(ev.loc["pre", "return_390m"])                          # window spans the split
    assert "suspected_unadjusted_split_390m_stock" in ev.loc["pre", "data_quality_flag"]
    assert ev.loc["pre", "return_60m"] == pytest.approx(0.0)               # same-day window is clean
    assert ev.loc["post", "return_390m"] == pytest.approx(0.0)
    assert "split" not in ev.loc["post", "data_quality_flag"]


def test_explicit_corporate_action_table_flags_window_but_keeps_returns():
    actions = pd.DataFrame({"ticker": ["NVDA"], "effective_date": ["2026-11-05"]})
    ev = one(build_event_dataset(make_news({"id": "x", "at": "2026-11-04 10:00:30"}),
                                 make_bars(D3), corporate_actions=actions))
    assert "corporate_action_in_window_390m_stock" in ev["data_quality_flag"]
    assert "corporate_action_in_window_60m_stock" not in ev["data_quality_flag"]
    assert not np.isnan(ev["return_390m"])


# ---- first_seen vs published --------------------------------------------------------------

def test_first_seen_preferred_and_delay_flagged():
    news = make_news({"id": "d", "at": "2026-11-04 09:00:00", "first_seen_at_utc": ny("2026-11-04 10:00:30")})
    ev = one(build_event_dataset(news, make_bars(D3)))
    assert ev["availability_source"] == "first_seen"
    assert ev["event_timestamp_utc"] == ny("2026-11-04 10:00:30")
    assert ev["entry_timestamp_utc"] == ny("2026-11-04 10:02")
    assert "delayed_availability" in ev["data_quality_flag"]


def test_publisher_time_only_is_documented_in_flag():
    ev = one(build_event_dataset(make_news({"id": "p", "at": "2026-11-04 10:00:30"}, first_seen=False), make_bars(D3)))
    assert ev["availability_source"] == "published" and "publisher_time_only" in ev["data_quality_flag"]


def test_first_seen_before_published_is_flagged():
    news = make_news({"id": "p", "at": "2026-11-04 10:00:30", "first_seen_at_utc": ny("2026-11-04 09:50:00")})
    assert "first_seen_before_published" in one(build_event_dataset(news, make_bars(D3)))["data_quality_flag"]


# ---- bar frequency / extended hours / inputs --------------------------------------------

def test_five_minute_bars_give_exact_60_and_390_minute_horizons():
    over = {("NVDA", "2026-11-04 10:10"): 100, ("NVDA", "2026-11-04 11:10"): 101, ("NVDA", "2026-11-05 10:10"): 102,
            ("SPY", "2026-11-04 10:10"): 500, ("SPY", "2026-11-04 11:10"): 502, ("SPY", "2026-11-05 10:10"): 504}
    bars = make_bars(D3, overrides=over, bar_minutes=5)
    ev = one(build_event_dataset(make_news({"id": "5m", "at": "2026-11-04 10:00:30"}), bars))
    assert ev["entry_timestamp_utc"] == ny("2026-11-04 10:10")             # bar 10:05-10:10 is first starting after news
    assert ev["abnormal_return_60m"] == pytest.approx(0.006)
    assert ev["abnormal_return_390m"] == pytest.approx(0.012)
    with pytest.raises(ValueError, match="multiple"):
        build_event_dataset(make_news({"id": "5m", "at": "2026-11-04 10:00:30"}), bars, horizons=(7,))


def test_extended_hours_bars_are_ignored():
    news = make_news({"id": "x", "at": "2026-11-04 10:00:30"})
    clean = one(build_event_dataset(news, make_bars(D3, overrides=EXAMPLE)))
    noisy = one(build_event_dataset(news, make_bars(D3, overrides=EXAMPLE, extra_hours=True)))
    assert noisy["abnormal_return_390m"] == clean["abnormal_return_390m"]
    after = one(build_event_dataset(make_news({"id": "y", "at": "2026-11-04 16:30:00"}), make_bars(D3, extra_hours=True)))
    assert after["entry_timestamp_utc"] == ny("2026-11-05 09:31") and after["entry_price"] == 100


def test_conflicting_duplicate_bars_raise_but_identical_duplicates_do_not():
    bars = make_bars(D3)
    same = pd.concat([bars, bars.iloc[[10]]], ignore_index=True)
    build_event_dataset(make_news({"id": "x", "at": "2026-11-04 10:00:30"}), same)
    bad = bars.iloc[[10]].copy()
    bad["close"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        build_event_dataset(make_news({"id": "x", "at": "2026-11-04 10:00:30"}), pd.concat([bars, bad]))


def test_schema_labels_not_fabricated_custom_horizons_and_roundtrip(tmp_path):
    news = make_news({"id": "a", "at": "2026-11-04 10:00:30", "novelty_label": "mixed"},
                     {"id": "b", "at": "2026-11-05 10:00:30"})
    ev = build_event_dataset(news, make_bars(D3 + ["2026-11-09"]))
    assert set(REQUIRED) <= set(ev.columns) and len(ev) == 2
    assert ev["event_timestamp_utc"].dt.tz is not None
    assert ev.loc[0, "novelty_label"] == "mixed" and pd.isna(ev.loc[1, "novelty_label"])
    assert ev["direction_label"].isna().all() and ev["event_type"].isna().all()
    custom = build_event_dataset(news, make_bars(D3 + ["2026-11-09"]), horizons=(30, 120))
    assert {"return_30m", "abnormal_return_120m", "market_return_30m"} <= set(custom.columns)
    assert "return_60m" not in custom.columns
    for suffix in ("parquet", "csv"):
        p = save_event_dataset(ev, tmp_path / f"events.{suffix}")
        back = pd.read_parquet(p) if suffix == "parquet" else pd.read_csv(p)
        np.testing.assert_allclose(back["abnormal_return_60m"], ev["abnormal_return_60m"])


def test_match_news_to_prices_exposes_entry_alignment():
    m = match_news_to_prices(make_news({"id": "a", "at": "2026-11-04 17:00:00"}), make_bars(D3))
    assert m.loc[0, "entry_timestamp_utc"] == ny("2026-11-05 09:31")
    assert m.loc[0, "release_session_state"] == "post_close"
    assert m.loc[0, "entry_price"] == 100 and m.loc[0, "market_entry_price"] == 500
