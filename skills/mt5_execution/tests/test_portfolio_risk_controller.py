"""
Unit tests for Institutional Portfolio Risk Controller.
"""
import pytest
import pandas as pd
import numpy as np
from skills.mt5_execution.scripts.portfolio_risk_controller import PortfolioRiskController


@pytest.fixture
def risk_ctrl():
    return PortfolioRiskController(
        max_portfolio_risk_pct=0.0075,
        base_trade_risk_pct=0.0025,
        max_drawdown_limit_pct=0.0150,
        correlation_threshold=0.60,
    )


def test_circuit_breaker_triggers_on_drawdown(risk_ctrl):
    # Healthy account
    halted, dd = risk_ctrl.evaluate_circuit_breaker(equity=9950.0, balance=10000.0)
    assert not halted
    assert dd == 0.0050  # 0.50% drawdown

    # Breached account (2.0% drawdown >= 1.50% limit)
    halted, dd = risk_ctrl.evaluate_circuit_breaker(equity=9800.0, balance=10000.0)
    assert halted
    assert dd == 0.0200


def test_correlation_discount_applied_on_usd_drivers(risk_ctrl):
    # Active Long EURUSD (USD Short)
    open_pos = [{"symbol": "EURUSD.x", "type": "BUY", "volume": 0.14, "price_open": 1.1600, "sl": 1.1580, "contract_size": 100000.0}]

    # Candidate Long XAGUSD (also USD Short)
    decision = risk_ctrl.calculate_permitted_risk(
        candidate_symbol="XAGUSD.x",
        candidate_direction="BUY",
        open_positions=open_pos,
        equity=10000.0,
        balance=10000.0
    )

    assert decision["permitted"] is True
    assert decision["correlation_discount_applied"] is True
    # Base risk 0.25% * 0.50 discount = 0.125%
    assert decision["risk_pct"] == 0.00125


def test_portfolio_risk_budget_exhaustion(risk_ctrl):
    # 3 active trades taking 0.25% each = 0.75% total risk (cap)
    open_pos = [
        {"symbol": "EURUSD.x", "type": "BUY", "volume": 0.14, "price_open": 1.1600, "sl": 1.1582, "contract_size": 100000.0},
        {"symbol": "NAS100.x", "type": "BUY", "volume": 0.01, "price_open": 20000.0, "sl": 19750.0, "contract_size": 10.0},
        {"symbol": "WTI.x", "type": "BUY", "volume": 0.19, "price_open": 75.00, "sl": 73.70, "contract_size": 100.0},
    ]

    # Candidate 4th trade on USDJPY
    decision = risk_ctrl.calculate_permitted_risk(
        candidate_symbol="USDJPY.x",
        candidate_direction="BUY",
        open_positions=open_pos,
        equity=10000.0,
        balance=10000.0
    )

    # Must be rejected because portfolio risk budget (0.75%) is full
    assert decision["permitted"] is False
    assert "EXHAUSTED" in decision["reason"]
