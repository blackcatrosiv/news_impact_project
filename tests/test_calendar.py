import datetime as dt

import pandas as pd
import pytest

from helpers import make_bars
from news_impact import NYSECalendar, check_calendar_against_prices
from news_impact.trading_calendar import nyse_early_closes, nyse_holidays


def D(s):
    return dt.date.fromisoformat(s)


@pytest.mark.parametrize("day", [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",      # 2026 (Jul 4 is a Saturday)
    "2024-03-29", "2025-04-18", "2025-01-09", "2022-06-20", "2021-12-24",
])
def test_closed_days(day):
    assert not NYSECalendar().is_session(day)


@pytest.mark.parametrize("day", ["2026-11-27", "2026-12-24", "2026-11-04", "2021-12-31", "2022-12-30", "2026-07-02"])
def test_open_days(day):
    # 2021-12-31 and 2022-12-30: a Saturday Jan 1 is not observed on the prior Friday.
    assert NYSECalendar().is_session(day)


def test_early_closes():
    assert D("2026-11-27") in nyse_early_closes(2026, nyse_holidays(2026))
    assert D("2025-12-24") in nyse_early_closes(2025, nyse_holidays(2025))
    assert D("2025-07-03") in nyse_early_closes(2025, nyse_holidays(2025))
    assert D("2026-07-03") not in nyse_early_closes(2026, nyse_holidays(2026))   # full holiday instead
    assert D("2026-12-24") in nyse_early_closes(2026, nyse_holidays(2026))
    s = NYSECalendar().sessions("2026-11-26", "2026-11-30").set_index("session_date")
    assert bool(s.loc["2026-11-27", "early_close"]) and not bool(s.loc["2026-11-30", "early_close"])


def test_session_times_follow_dst():
    s = NYSECalendar().sessions("2026-03-06", "2026-03-09").set_index("session_date")
    to_utc = lambda ns: pd.Timestamp(ns, unit="ns", tz="UTC")
    assert to_utc(s.loc["2026-03-06", "open_ns"]) == pd.Timestamp("2026-03-06 14:30", tz="UTC")   # EST
    assert to_utc(s.loc["2026-03-09", "open_ns"]) == pd.Timestamp("2026-03-09 13:30", tz="UTC")   # EDT
    assert to_utc(s.loc["2026-03-09", "close_ns"]) == pd.Timestamp("2026-03-09 20:00", tz="UTC")


def test_extra_holidays_override():
    cal = NYSECalendar(extra_holidays=("2026-11-04",))
    assert not cal.is_session("2026-11-04")


def test_check_calendar_against_prices_reports_gaps():
    bars = make_bars(["2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"], early_close={"2026-11-27"})
    assert check_calendar_against_prices(bars) == {"in_data_not_calendar": [], "in_calendar_not_data": []}
    bars_missing_day = bars[~bars["bar_end_utc"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").eq("2026-11-25")]
    res = check_calendar_against_prices(bars_missing_day)
    assert res["in_calendar_not_data"] == [D("2026-11-25")]
