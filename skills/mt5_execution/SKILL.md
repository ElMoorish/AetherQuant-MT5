---
name: mt5-execution
description: >-
  MetaTrader 5 (MT5) execution engine, high-frequency tick/bar ingestion, 
  dynamic ATR position sizing, hard risk management (mandatory SL/TP), 
  and resilient order routing with retry handlers.
---

# MetaTrader 5 Execution & Risk Management Skill

This skill defines the standard API boundaries, risk guardrails, and execution workflows for interfacing with MetaTrader 5 (MT5) terminals via the official Python API.

---

## 1. Core API Boundaries & Responsibilities

The execution system is partitioned into three discrete layers:
1. **`MT5Client` (`scripts/mt5_client.py`)**: Terminal lifecycle management (`initialize`, `shutdown`), symbol subscription, tick/bar extraction, and account telemetry.
2. **`RiskManager` (`scripts/risk_manager.py`)**: Dynamic lot sizing based on account equity, Average True Range (ATR), risk-per-trade percentage, and broker margin constraints.
3. **`OrderRouter` (`scripts/order_router.py`)**: Resilient trade execution, mandatory Stop Loss (SL) and Take Profit (TP) enforcement, slippage control, dynamic trailing stop tracking, and order retry loops.

---

## 2. Hard Risk Management Guardrails (Rule B Compliance)

- **Mandatory SL & TP**: Any trade request missing explicit `sl` or `tp` is rejected before reaching `MetaTrader5.order_send()`.
- **Dynamic Lot Sizing Formula**:
  $$\text{Risk Amount} = \text{Account Equity} \times \text{Risk Fraction}$$
  $$\text{Stop Distance (points)} = \frac{|\text{Entry Price} - \text{SL}|}{\text{Point Size}}$$
  $$\text{Lot Size} = \frac{\text{Risk Amount}}{\text{Stop Distance} \times \text{Tick Value}} \implies \text{clamped to } [\text{vol\_min}, \text{vol\_max}] \text{ aligned to } \text{vol\_step}$$
- **Execution Retry Protocol**:
  Orders encountering transient broker errors (`TRADE_RETCODE_REQUOTE`, `TRADE_RETCODE_CONNECTION`, `TRADE_RETCODE_PRICE_OFF`) are retried up to $N$ times with exponential backoff and updated price quotes.

---

## 3. Standard Usage Workflow

```python
from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.risk_manager import RiskManager
from skills.mt5_execution.scripts.order_router import OrderRouter

# 1. Initialize Terminal Connection
client = MT5Client()
if client.connect():
    # 2. Extract Data
    df = client.get_rates("EURUSD", timeframe="M5", count=500)
    
    # 3. Size Position
    risk = RiskManager(client)
    lots = risk.calculate_lot_size("EURUSD", risk_pct=0.01, sl_points=250)
    
    # 4. Route Order with Hard SL/TP
    router = OrderRouter(client)
    result = router.send_market_order(
        symbol="EURUSD",
        order_type="BUY",
        volume=lots,
        sl_points=250,
        tp_points=500
    )
```

---

## 4. Module Directory Structure

```text
skills/mt5_execution/
├── SKILL.md
├── scripts/
│   ├── __init__.py
│   ├── mt5_client.py     # Terminal connection, rates, ticks
│   ├── risk_manager.py   # Equity & ATR-based position sizing
│   └── order_router.py   # Safe order routing with SL/TP & retry loops
└── tests/
    ├── __init__.py
    └── test_mt5_execution.py # Pytest test suite with mocked MT5
```
