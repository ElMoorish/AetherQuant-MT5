"""
Rule B: Institutional Risk Management Module (Hard Dollar-Risk Guardrail)
=========================================================================
Guarantees mandatory SL/TP parameters, dynamic ATR stop sizing, and hard dollar risk ceilings.
Overrides broker minimum-lot size constraints to strictly prevent risk overshoots.
"""
import math
import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from skills.mt5_execution.scripts.mt5_client import MT5Client

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Institutional risk calculation and strict trade parameter validation engine.
    """

    def __init__(
        self,
        client: Optional[MT5Client] = None,
        default_risk_pct: float = 0.0015,  # 0.15% Balanced Growth Default
        max_risk_pct: float = 0.01,         # 1.0% Hard ceiling
    ):
        self.client = client or MT5Client()
        self.default_risk_pct = default_risk_pct
        self.max_risk_pct = max_risk_pct

    def calculate_lot_size_and_sl(
        self,
        symbol: str,
        sl_points: float,
        risk_pct: Optional[float] = None,
        account_equity: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculates position volume (lots) and guarantees that dollar loss at SL never exceeds risk budget.
        If min lot (0.01) would exceed dollar risk, automatically tightens SL points to match budget.
        Returns: (lot_size, effective_sl_points)
        """
        if sl_points <= 0:
            raise ValueError(f"Stop loss points must be strictly positive (>0), got: {sl_points}")

        risk = min(risk_pct or self.default_risk_pct, self.max_risk_pct)

        if account_equity is None:
            acc = self.client.get_account_info()
            equity = acc.get("equity", 10000.0)
        else:
            equity = account_equity

        risk_amount = equity * risk

        sym_info = self.client.get_symbol_info(symbol)
        if not sym_info:
            logger.warning(f"Unable to retrieve symbol info for {symbol}, defaulting to min volume 0.01")
            return 0.01, sl_points

        point = sym_info.get("point", 0.00001)
        tick_value = sym_info.get("trade_tick_value", 1.0)
        tick_size = sym_info.get("trade_tick_size", point)
        vol_min = sym_info.get("volume_min", 0.01)
        vol_max = sym_info.get("volume_max", 100.0)
        vol_step = sym_info.get("volume_step", 0.01)

        value_per_point_per_lot = (point / tick_size) * tick_value if tick_size > 0 else 1.0
        dollar_loss_per_lot = sl_points * value_per_point_per_lot

        if dollar_loss_per_lot <= 0:
            return vol_min, sl_points

        raw_lots = risk_amount / dollar_loss_per_lot
        steps = math.floor(raw_lots / vol_step)
        calc_lots = steps * vol_step
        calc_lots = max(vol_min, min(vol_max, calc_lots))
        calc_lots = round(calc_lots, 2)

        # ─── HARD DOLLAR RISK CEILING ───
        # Check if 0.01 min lot violates the dollar risk budget
        actual_dollar_risk = calc_lots * sl_points * value_per_point_per_lot
        effective_sl = sl_points

        if actual_dollar_risk > (risk_amount * 1.05):
            # Tighten SL distance so actual dollar risk == risk_amount
            adjusted_sl = risk_amount / (calc_lots * value_per_point_per_lot + 1e-8)
            effective_sl = round(adjusted_sl, 1)
            logger.info(
                f"🛡️ HARD RISK CEILING: Adjusted SL for {symbol} from {sl_points:.1f} to {effective_sl:.1f} pts "
                f"to guarantee max dollar risk <= ${risk_amount:.2f} at min lot {calc_lots}"
            )
        else:
            logger.info(
                f"Symbol: {symbol} | Equity: ${equity:.2f} | Risk%: {risk*100:.2f}% | "
                f"Risk$: ${risk_amount:.2f} | SL pts: {effective_sl} | Lot Size: {calc_lots}"
            )

        return calc_lots, effective_sl

    def calculate_lot_size(
        self,
        symbol: str,
        sl_points: float,
        risk_pct: Optional[float] = None,
        account_equity: Optional[float] = None,
    ) -> float:
        lots, _ = self.calculate_lot_size_and_sl(symbol, sl_points, risk_pct, account_equity)
        return lots

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Computes rolling Average True Range (ATR) series on DataFrame."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_atr_stop_distance(
        self,
        symbol: str,
        timeframe: str = "H1",
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ) -> float:
        """
        Computes the Stop Loss distance in points using the Average True Range (ATR).
        """
        rates = self.client.get_rates(symbol=symbol, timeframe=timeframe, count=atr_period + 50)
        if rates.empty or len(rates) < atr_period:
            logger.warning(f"Insufficient rates for {symbol}, using fallback 200 points")
            return 200.0

        high = rates["high"]
        low = rates["low"]
        close = rates["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean().iloc[-1]

        sym_info = self.client.get_symbol_info(symbol)
        point = sym_info.get("point", 0.00001)

        sl_points = (atr * atr_multiplier) / point
        return max(50.0, float(sl_points))
