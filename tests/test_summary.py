import numpy as np
import pandas as pd
import pytest
from scipy import stats

from news_impact import compare_groups, summarize_event_impacts

AR = np.array([0.01, -0.02, 0.03, 0.00, -0.01, 0.04, np.nan, 0.02])


def events(**extra):
    df = pd.DataFrame({"abnormal_return_60m": AR, "abnormal_return_390m": AR * 2,
                       "overlapping_event_flag": [False] * 6 + [False, True]})
    for k, v in extra.items():
        df[k] = v
    return df


def test_descriptive_statistics_match_numpy_and_exclude_nan():
    row = summarize_event_impacts(events()).query("horizon_min == 60").iloc[0]
    x = AR[~np.isnan(AR)]
    assert row["n"] == 7 and row["n_missing"] == 1
    assert row["mean"] == pytest.approx(x.mean())
    assert row["median"] == pytest.approx(np.median(x))
    assert row["std"] == pytest.approx(x.std(ddof=1))
    assert row["mean_abs"] == pytest.approx(np.abs(x).mean())
    assert row["fraction_positive"] == pytest.approx((x > 0).mean())
    assert row["q05"] == pytest.approx(np.quantile(x, 0.05)) and row["q95"] == pytest.approx(np.quantile(x, 0.95))


def test_naive_ci_matches_formula():
    row = summarize_event_impacts(events()).query("horizon_min == 60").iloc[0]
    x = AR[~np.isnan(AR)]
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    assert row["mean_ci_low"] == pytest.approx(x.mean() - half)
    assert row["mean_ci_high"] == pytest.approx(x.mean() + half)


def test_cluster_ci_equals_naive_when_every_event_is_its_own_cluster_and_widens_when_clustered():
    solo = summarize_event_impacts(events(overlap_cluster_id=list("abcdefgh")), cluster_column="overlap_cluster_id")
    r = solo.query("horizon_min == 60").iloc[0]
    assert r["mean_ci_low_cluster"] == pytest.approx(r["mean_ci_low"])
    # identical returns duplicated within clusters carry no extra information -> wider than naive
    dup = pd.DataFrame({"abnormal_return_60m": np.repeat([0.01, -0.02, 0.03, 0.02], 5),
                        "overlap_cluster_id": np.repeat(list("abcd"), 5)})
    r = summarize_event_impacts(dup, cluster_column="overlap_cluster_id").iloc[0]
    assert (r["mean_ci_high_cluster"] - r["mean_ci_low_cluster"]) > (r["mean_ci_high"] - r["mean_ci_low"])
    assert r["n_clusters"] == 4


def test_group_by_label_keeps_missing_labels_as_their_own_group():
    # AR60 by row: new: .01, -.02, NaN | old: .03, .00, -.01, .02 | unlabeled: .04
    labels = ["new", "new", "old", "old", "old", None, "new", "old"]
    out = summarize_event_impacts(events(novelty_label=labels), "novelty_label")
    h60 = out[out["horizon_min"] == 60]
    h60 = h60.set_index(h60["novelty_label"].fillna("NA"))
    assert h60.loc["new", "n"] == 2 and h60.loc["new", "n_missing"] == 1
    assert h60.loc["old", "n"] == 4 and h60.loc["NA", "n"] == 1
    assert h60.loc["new", "mean"] == pytest.approx(-0.005)
    assert h60.loc["old", "mean_abs"] == pytest.approx(np.mean([0.03, 0.0, 0.01, 0.02]))


def test_exclude_overlapping_sensitivity():
    out = summarize_event_impacts(events(), exclude_overlapping=True).query("horizon_min == 60").iloc[0]
    assert out["n"] + out["n_missing"] == 7


def test_other_measures_and_horizon_discovery():
    df = pd.DataFrame({"return_60m": [0.01, 0.02], "return_390m": [0.0, 0.1]})
    out = summarize_event_impacts(df, measure="return")
    assert sorted(out["horizon_min"]) == [60, 390]
    with pytest.raises(ValueError):
        summarize_event_impacts(df, measure="abnormal_return")


def test_compare_groups_matches_scipy_welch():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.004, 0.01, 40), rng.normal(0.0, 0.015, 55)
    df = pd.DataFrame({"abnormal_return_60m": np.concatenate([a, b]),
                       "novelty_label": ["new"] * 40 + ["old"] * 55})
    res = compare_groups(df, "novelty_label", "new", "old", 60)
    t = stats.ttest_ind(a, b, equal_var=False)
    assert res["delta"] == pytest.approx(a.mean() - b.mean())
    assert res["p_value"] == pytest.approx(t.pvalue)
    assert res["ci_low"] < res["delta"] < res["ci_high"]
    absres = compare_groups(df, "novelty_label", "new", "old", 60, use_abs=True)
    assert absres["delta"] == pytest.approx(np.abs(a).mean() - np.abs(b).mean())
