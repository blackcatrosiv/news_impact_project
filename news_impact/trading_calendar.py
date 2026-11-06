"""Rule-based NYSE regular-session calendar.

Only what the event study needs: which dates are trading sessions, when each
session opens and closes (in UTC, DST-aware), and which sessions close early.

Scope and limits
----------------
* Regular hours only: 09:30-16:00 America/New_York (13:00 close on early-close days).
* Holiday rules follow the current NYSE schedule (Juneteenth from 2022, MLK from 1998).
  Intended for roughly 2010 onward.
* A short list of well-known unscheduled closures is built in; it is NOT exhaustive.
  Pass ``extra_holidays`` / ``extra_early_closes`` to add others, and run
  :func:`check_calendar_against_prices` to compare the calendar with the bars you hold.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.easter import easter

NY_TZ = "America/New_York"

#: Unscheduled full-day closures (national days of mourning, 9/11, Hurricane Sandy).
SPECIAL_CLOSURES = (
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
    "2004-06-11",  # President Reagan
    "2007-01-02",  # President Ford
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # President H.W. Bush
    "2025-01-09",  # President Carter
)

SESSION_COLUMNS = ["session_date", "open_ns", "close_ns", "early_close"]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th ``weekday`` (Mon=0) of a month."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date, saturday_to_friday: bool = True) -> dt.date | None:
    """Weekend holidays move to Friday (Sat) or Monday (Sun)."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1) if saturday_to_friday else None
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(year: int) -> dict[dt.date, str]:
    """Scheduled NYSE full-day closures that fall in ``year``."""
    h: dict[dt.date, str] = {}

    def put(d: dt.date | None, name: str) -> None:
        if d is not None:
            h[d] = name

    # A Saturday Jan 1 is NOT observed on the previous Friday.
    put(_observed(dt.date(year, 1, 1), saturday_to_friday=False), "New Year's Day")
    if year >= 1998:
        put(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    put(_nth_weekday(year, 2, 0, 3), "Washington's Birthday")
    put(easter(year) - dt.timedelta(days=2), "Good Friday")
    put(_last_weekday(year, 5, 0), "Memorial Day")
    if year >= 2022:
        put(_observed(dt.date(year, 6, 19)), "Juneteenth")
    put(_observed(dt.date(year, 7, 4)), "Independence Day")
    put(_nth_weekday(year, 9, 0, 1), "Labor Day")
    put(_nth_weekday(year, 11, 3, 4), "Thanksgiving Day")
    put(_observed(dt.date(year, 12, 25)), "Christmas Day")
    return h


def nyse_early_closes(year: int, holidays: Iterable[dt.date] = ()) -> set[dt.date]:
    """Scheduled 13:00 closes: day after Thanksgiving, Jul 3 and Dec 24 when Mon-Thu."""
    hol = set(holidays)
    out = {_nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1)}
    for month, day in ((7, 3), (12, 24)):
        d = dt.date(year, month, day)
        if d.weekday() < 4 and d not in hol:
            out.add(d)
    return out


def _as_date(x) -> dt.date:
    return pd.Timestamp(x).date()


@dataclass(frozen=True)
class NYSECalendar:
    """Regular-session calendar. Immutable, so instances are safe to share."""

    extra_holidays: tuple = ()
    extra_early_closes: tuple = ()
    include_special_closures: bool = True

    def _closed_dates(self, y0: int, y1: int) -> set[dt.date]:
        closed: set[dt.date] = set()
        for y in range(y0 - 1, y1 + 2):
            closed.update(nyse_holidays(y))
        if self.include_special_closures:
            closed.update(_as_date(d) for d in SPECIAL_CLOSURES)
        closed.update(_as_date(d) for d in self.extra_holidays)
        return closed

    def _early_dates(self, y0: int, y1: int, closed: set[dt.date]) -> set[dt.date]:
        early: set[dt.date] = set()
        for y in range(y0 - 1, y1 + 2):
            early.update(nyse_early_closes(y, closed))
        early.update(_as_date(d) for d in self.extra_early_closes)
        return early - closed

    def is_session(self, d) -> bool:
        d = _as_date(d)
        if d.weekday() >= 5:
            return False
        return d not in self._closed_dates(d.year, d.year)

    def sessions(self, start, end) -> pd.DataFrame:
        """Sessions with start <= date <= end.

        Columns: ``session_date`` (naive midnight), ``open_ns`` / ``close_ns``
        (UTC epoch nanoseconds, int64) and ``early_close`` (bool).
        """
        d0, d1 = _as_date(start), _as_date(end)
        if d1 < d0:
            return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
                SESSION_COLUMNS, ["datetime64[ns]", "int64", "int64", "bool"])})
        closed = self._closed_dates(d0.year, d1.year)
        early = self._early_dates(d0.year, d1.year, closed)
        days = [d for d in pd.bdate_range(d0, d1).date if d not in closed]
        dates = pd.DatetimeIndex(pd.to_datetime(days)).as_unit("ns")
        is_early = np.array([d in early for d in days], dtype=bool)
        open_local = dates + pd.to_timedelta(9 * 60 + 30, unit="min")
        close_local = dates + pd.to_timedelta(np.where(is_early, 13 * 60, 16 * 60), unit="min")
        # 09:30 / 13:00 / 16:00 never fall in a DST gap or overlap, so localizing is unambiguous.
        open_ns = open_local.tz_localize(NY_TZ).tz_convert("UTC").as_unit("ns").asi8
        close_ns = close_local.tz_localize(NY_TZ).tz_convert("UTC").as_unit("ns").asi8
        return pd.DataFrame({
            "session_date": dates,
            "open_ns": open_ns.astype("int64"),
            "close_ns": close_ns.astype("int64"),
            "early_close": is_early,
        })


def check_calendar_against_prices(prices_df: pd.DataFrame, calendar: NYSECalendar | None = None,
                                  reference_ticker: str | None = None) -> dict:
    """Compare the calendar's sessions with the dates that actually have regular-hours bars.

    Returns ``{"in_data_not_calendar": [...], "in_calendar_not_data": [...]}`` (dates).
    Dates in the second list are either unlisted closures or gaps in your data; either way
    event windows touching them are flagged/NaN rather than silently wrong.
    """
    cal = calendar or NYSECalendar()
    df = prices_df
    if reference_ticker is not None:
        df = df[df["ticker"] == reference_ticker]
    col = "bar_end_utc" if "bar_end_utc" in df.columns else "bar_start_utc"
    ts = pd.to_datetime(df[col], utc=True).dt.tz_convert(NY_TZ)
    local_time = ts.dt.hour * 60 + ts.dt.minute
    regular = ts[(local_time > 9 * 60 + 30) & (local_time <= 16 * 60)]
    data_dates = set(regular.dt.date.unique())
    if not data_dates:
        return {"in_data_not_calendar": [], "in_calendar_not_data": []}
    sess = cal.sessions(min(data_dates), max(data_dates))
    cal_dates = set(pd.DatetimeIndex(sess["session_date"]).date)
    return {
        "in_data_not_calendar": sorted(data_dates - cal_dates),
        "in_calendar_not_data": sorted(cal_dates - data_dates),
    }
