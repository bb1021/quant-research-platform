import numpy as np
import pandas as pd
import pytest

from src.metrics import annualized_volatility, max_drawdown, sharpe_ratio


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_sharpe_ratio_positive_for_positive_mean_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.01, 0.0])
    assert sharpe_ratio(returns) > 0
    assert annualized_volatility(returns) > 0


def test_sharpe_ratio_nan_when_no_variance():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns))

