"""
Unit test suite for Multi-Asset Trading Daemon (Precision Shield Edition).
"""
import pytest
import numpy as np
import pandas as pd
from scripts.live_trading_daemon import MultiAssetTradingDaemon
from scripts.train_super_alpha_model import engineer_18_alpha_features, ALPHA_FEATURES


@pytest.fixture
def mock_daemon():
    daemon = MultiAssetTradingDaemon(
        symbols=["EURUSD", "XAGUSD"],
        timeframe="H1",
        mode="paper",
        base_risk_pct=0.0015,
    )
    return daemon


def test_daemon_18_alpha_feature_engineering():
    np.random.seed(42)
    n = 150
    df_h1 = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": 1.1000 + np.random.normal(0, 0.001, n),
        "high": 1.1020 + np.random.normal(0, 0.001, n),
        "low": 1.0980 + np.random.normal(0, 0.001, n),
        "close": 1.1000 + np.random.normal(0, 0.001, n),
        "tick_volume": np.random.randint(100, 1000, n),
    })
    feat_df = engineer_18_alpha_features(df_h1)

    assert len(feat_df) == n - 50
    for col in ALPHA_FEATURES:
        assert col in feat_df.columns, f"Missing feature: {col}"
        assert not feat_df[col].isna().any(), f"NaN in feature: {col}"


def test_daemon_initialization(mock_daemon):
    assert "EURUSD" in mock_daemon.symbols
    assert mock_daemon.timeframe == "H1"
    assert mock_daemon.mode == "paper"
    assert mock_daemon.base_risk_pct == 0.0015
    assert mock_daemon.state["status"] == "INITIALIZING"
