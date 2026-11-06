import numpy as np
import pandas as pd
import pytest

from news_impact import calculate_abnormal_return, calculate_return


def test_brief_examples():
    assert calculate_return(100.0, 101.0) == pytest.approx(0.01)
    assert calculate_return(500.0, 502.0) == pytest.approx(0.004)
    assert calculate_abnormal_return(0.01, 0.004) == pytest.approx(0.006)


def test_worked_numerical_example_both_horizons():
    r60, m60 = calculate_return(100, 101), calculate_return(500, 502)
    r390, m390 = calculate_return(100, 102), calculate_return(500, 504)
    assert calculate_abnormal_return(r60, m60) == pytest.approx(0.006)
    assert calculate_abnormal_return(r390, m390) == pytest.approx(0.012)


def test_negative_and_zero_moves():
    assert calculate_return(100, 90) == pytest.approx(-0.10)
    assert calculate_return(100, 100) == 0.0


def test_vectorised_and_series_keep_shape_and_index():
    out = calculate_return(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    np.testing.assert_allclose(out, [0.10, -0.05])
    s = calculate_return(pd.Series([100.0, 50.0], index=["a", "b"]), pd.Series([101.0, 55.0], index=["a", "b"]))
    assert list(s.index) == ["a", "b"]
    assert s["b"] == pytest.approx(0.10)


@pytest.mark.parametrize("start,end", [(np.nan, 101), (100, np.nan), (0, 5), (-1, 5), (100, 0), (None, 1)])
def test_invalid_prices_give_nan_not_zero_or_inf(start, end):
    assert np.isnan(calculate_return(start, end))


def test_abnormal_propagates_nan():
    assert np.isnan(calculate_abnormal_return(np.nan, 0.01))
    assert np.isnan(calculate_abnormal_return(0.01, np.nan))
    out = calculate_abnormal_return(np.array([0.01, np.nan]), np.array([0.004, 0.0]))
    assert out[0] == pytest.approx(0.006) and np.isnan(out[1])
