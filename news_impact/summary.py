"""Distribution summaries and group comparisons for the event dataset (Task 3).

Everything here describes *association* between news timing and subsequent returns. None of
it is evidence of causality, of predictive value, or of tradable profit.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_QUANTILES = (0.05, 0.25, 0.75, 0.95)


def _qname(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def _naive_ci(x: np.ndarray, alpha: float):
    """AR_bar +/- t_{1-alpha/2, N-1} * s / sqrt(N). Assumes independent observations."""
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return x.mean() - t * se, x.mean() + t * se


def _cluster_ci(x: np.ndarray, clusters: np.ndarray, alpha: float):
    """CI for the mean with cluster-robust SE (observations in a cluster may be dependent).

    Var(mean) = G/(G-1) * sum_g (sum_{i in g}(x_i - xbar))^2 / N^2, t with G-1 df. With one
    observation per cluster this equals the naive interval. Needs G >= 2 clusters.
    """
    n = len(x)
    codes, uniques = pd.factorize(clusters)
    g = len(uniques)
    if g < 2:
        return np.nan, np.nan
    resid = x - x.mean()
    sums = np.bincount(codes, weights=resid, minlength=g)
    se = np.sqrt(g / (g - 1) * np.sum(sums ** 2)) / n
    t = stats.t.ppf(1 - alpha / 2, g - 1)
    return x.mean() - t * se, x.mean() + t * se


def _describe(x: np.ndarray, clusters, alpha: float, quantiles) -> dict:
    n = len(x)
    out = {"n": n}
    if n == 0:
        nan = np.nan
        out.update(mean=nan, median=nan, std=nan, mean_abs=nan, fraction_positive=nan)
        out.update({_qname(q): nan for q in quantiles})
        out.update(mean_ci_low=nan, mean_ci_high=nan, mean_ci_low_cluster=nan, mean_ci_high_cluster=nan,
                   mean_abs_ci_low=nan, mean_abs_ci_high=nan, mean_abs_ci_low_cluster=nan,
                   mean_abs_ci_high_cluster=nan, n_clusters=0)
        return out
    ax = np.abs(x)
    out.update(mean=x.mean(), median=np.median(x), std=x.std(ddof=1) if n > 1 else np.nan,
               mean_abs=ax.mean(), fraction_positive=float((x > 0).mean()))
    for q, v in zip(quantiles, np.quantile(x, quantiles)):
        out[_qname(q)] = v
    out["mean_ci_low"], out["mean_ci_high"] = _naive_ci(x, alpha)
    out["mean_abs_ci_low"], out["mean_abs_ci_high"] = _naive_ci(ax, alpha)
    if clusters is not None:
        out["mean_ci_low_cluster"], out["mean_ci_high_cluster"] = _cluster_ci(x, clusters, alpha)
        out["mean_abs_ci_low_cluster"], out["mean_abs_ci_high_cluster"] = _cluster_ci(ax, clusters, alpha)
        out["n_clusters"] = int(pd.Series(clusters).nunique())
    else:
        out.update(mean_ci_low_cluster=np.nan, mean_ci_high_cluster=np.nan,
                   mean_abs_ci_low_cluster=np.nan, mean_abs_ci_high_cluster=np.nan, n_clusters=np.nan)
    return out


def _horizons_for(events_df: pd.DataFrame, measure: str):
    pat = re.compile(rf"^{re.escape(measure)}_(\d+)m$")
    return sorted(int(m.group(1)) for c in events_df.columns if (m := pat.match(c)))


def summarize_event_impacts(events_df: pd.DataFrame, group_columns=None, *, horizons=None,
                            measure: str = "abnormal_return", quantiles=DEFAULT_QUANTILES,
                            cluster_column: str | None = "auto", exclude_overlapping: bool = False,
                            alpha: float = 0.05) -> pd.DataFrame:
    """Report event-return distributions, optionally by news categories.

    Parameters
    ----------
    group_columns : column name(s) to group by, e.g. ``"novelty_label"`` supplied by a
        downstream RAG classifier. ``None`` gives one overall row per horizon. Missing labels
        form their own group (never filled in), so an unlabeled event is not silently assigned.
    measure : ``"abnormal_return"`` (default), ``"return"`` (raw stock) or ``"market_return"``.
    cluster_column : column defining dependence clusters for the cluster-robust CI. ``"auto"``
        uses ``overlap_cluster_id`` if present, else ``entry_session_date``; ``None`` disables.
    exclude_overlapping : drop events with ``overlapping_event_flag`` (sensitivity analysis).

    Returns one row per (group, horizon): ``n`` (non-missing observations), ``n_missing``, mean,
    median, std (ddof=1), ``mean_abs`` (mean absolute value), ``fraction_positive``, quantiles,
    naive t-interval for the mean and for the mean absolute value (valid only if observations
    are independent) and cluster-robust versions. Values are fractions (0.01 = 1%).
    """
    # Label columns are optional; summarize overall outcomes until downstream classifications
    # are joined, then pass their column name here for category-level descriptions.
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    group_columns = list(group_columns or [])
    hs = list(horizons) if horizons is not None else _horizons_for(events_df, measure)
    if not hs:
        raise ValueError(f"no '{measure}_<h>m' columns found in events_df")
    df = events_df
    if exclude_overlapping:
        df = df[~df["overlapping_event_flag"].astype(bool)]
    if cluster_column == "auto":
        cluster_column = next((c for c in ("overlap_cluster_id", "entry_session_date")
                               if c in df.columns), None)

    rows = []
    # dropna=False keeps articles awaiting labels visible as a separate group.
    groups = df.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), df)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        for h in hs:
            col = f"{measure}_{h}m"
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            ok = ~np.isnan(vals)
            cl = sub[cluster_column].to_numpy()[ok] if cluster_column else None
            if cl is not None and pd.isna(cl).any():       # unplaced events: each its own cluster
                cl = np.where(pd.isna(cl), [f"_solo{i}" for i in range(len(cl))], cl)
            row = dict(zip(group_columns, key))
            row.update(horizon_min=h, measure=measure, n_missing=int((~ok).sum()))
            row.update(_describe(vals[ok], cl, alpha, quantiles))
            rows.append(row)
    cols_first = group_columns + ["horizon_min", "measure", "n", "n_missing"]
    out = pd.DataFrame(rows)
    return out[cols_first + [c for c in out.columns if c not in cols_first]]


def compare_groups(events_df: pd.DataFrame, group_column: str, group_a, group_b, horizon: int, *,
                   measure: str = "abnormal_return", use_abs: bool = False, alpha: float = 0.05) -> dict:
    """Welch comparison ``delta = mean(group_a) - mean(group_b)`` (e.g. new minus old news).

    With ``use_abs=True`` compares mean absolute returns instead. Assumes independent
    observations within and across groups; overlapping/duplicate coverage makes the interval
    too narrow, so treat it as descriptive and check with ``exclude_overlapping=True``.
    """
    # This comparison is descriptive and assumes independent rows; use clustered summaries
    # when overlapping events make that assumption implausible.
    col = f"{measure}_{horizon}m"
    def vals(g):
        v = pd.to_numeric(events_df.loc[events_df[group_column] == g, col], errors="coerce").dropna()
        return v.abs().to_numpy() if use_abs else v.to_numpy()
    a, b = vals(group_a), vals(group_b)
    out = dict(group_a=group_a, group_b=group_b, horizon_min=horizon, measure=measure + ("_abs" if use_abs else ""),
               n_a=len(a), n_b=len(b), mean_a=a.mean() if len(a) else np.nan,
               mean_b=b.mean() if len(b) else np.nan)
    out["delta"] = out["mean_a"] - out["mean_b"]
    if len(a) < 2 or len(b) < 2:
        out.update(se=np.nan, ci_low=np.nan, ci_high=np.nan, t_stat=np.nan, p_value=np.nan, df=np.nan)
        return out
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        t_stat = out["delta"] / se if se > 0 else np.nan
    tcrit = stats.t.ppf(1 - alpha / 2, dof) if np.isfinite(dof) else np.nan
    out.update(se=se, ci_low=out["delta"] - tcrit * se, ci_high=out["delta"] + tcrit * se, t_stat=t_stat,
               p_value=2 * stats.t.sf(abs(t_stat), dof) if np.isfinite(t_stat) and np.isfinite(dof) else np.nan,
               df=dof)
    return out
