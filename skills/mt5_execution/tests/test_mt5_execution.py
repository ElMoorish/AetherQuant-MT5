"""
Unit test suite for MT5 Execution, Risk Management, and Order Routing.
"""
import pytest
import pandas as pd
import numpy as np

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.risk_manager import RiskManager
from skills.mt5_execution.scripts.order_router import OrderRouter


@pytest.fixture
def client():
    return MT5Client()


@pytest.fixture
def risk_manager(client):
    return RiskManager(client, default_risk_pct=0.01)


@pytest.fixture
def order_router(client):
    return OrderRouter(client)


def test_mt5_client_offline_rates(client):
    df = client.get_rates("EURUSD", timeframe="M5", count=100)
    assert not df.empty
    assert len(df) == 100
    assert "log_return" in df.columns
    assert "close" in df.columns
    # Check stationarity calculation
    assert not df["log_return"].isna().any()


def test_risk_manager_atr_calculation(risk_manager):
    data = {
        "high": [1.10, 1.12, 1.11, 1.13, 1.15],
        "low": [1.08, 1.09, 1.07, 1.10, 1.12],
        "close": [1.09, 1.11, 1.08, 1.12, 1.14],
    }
    df = pd.DataFrame(data)
    atr = risk_manager.calculate_atr(df, period=3)
    assert len(atr) == 5
    assert atr.iloc[-1] > 0


def test_risk_manager_dynamic_lot_sizing(risk_manager):
    # Test sizing with $100,000 equity, 1% risk ($1000), 200 points SL
    lot_size = risk_manager.calculate_lot_size(
        symbol="EURUSD",
        sl_points=200,
        risk_pct=0.01,
        account_equity=100000.0
    )
    assert lot_size > 0
    assert isinstance(lot_size, float)
    assert lot_size <= 100.0  # Under max limit


def test_order_router_mandatory_sltp_guardrail(order_router):
    # Rule B: must reject missing or non-positive SL/TP
    with pytest.raises(ValueError, match="Rule B Violation"):
        order_router.send_market_order(
            symbol="EURUSD",
            order_type="BUY",
            volume=0.1,
            sl_points=0,  # Invalid SL
            tp_points=200
        )

    with pytest.raises(ValueError, match="Rule B Violation"):
        order_router.send_market_order(
            symbol="EURUSD",
            order_type="SELL",
            volume=0.1,
            sl_points=100,
            tp_points=-50  # Invalid TP
        )


def test_order_router_valid_order_execution(order_router):
    result = order_router.send_market_order(
        symbol="EURUSD",
        order_type="BUY",
        volume=0.5,
        sl_points=150,
        tp_points=300
    )
    assert result is not None
    assert "retcode" in result
    assert result["retcode"] in [10009, 10008]  # TRADE_RETCODE_DONE / PLACED
    assert result["volume"] == 0.5
