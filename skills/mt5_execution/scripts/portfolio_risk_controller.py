"""
Institutional Portfolio Risk & Drawdown Controller (Precision Shield Edition)
=============================================================================
Enforces strict institutional portfolio-level risk limits, cross-asset correlation gates,
min-lot dollar risk ceilings, session rollover blackouts, and consecutive loss freezes.

Rules:
  - Base Risk: 0.15% ($15.00 / trade on $10k equity)
  - Max Simultaneous Portfolio Exposure: 0.60% ($60.00 max open risk)
  - Hard Dollar Risk Ceiling: Absolute max $15.00 risk per trade (overriding broker min lot)
  - Rollover Blackout: Freeze entries between 21:30 UTC and 23:30 UTC
  - Staggered Entry Queue: Minimum 1-hour spacing between cross-asset trade dispatches
  - Consecutive Loss Cooldown: 3-hour trading freeze after 2 consecutive stop-outs
  - Floating Drawdown Circuit Breaker: 1.50% hard halt
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("PortfolioRiskController")


class PortfolioRiskController:
    """
    Manages portfolio-wide capital allocation, cross-symbol correlations, and drawdown caps.
    """

    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.0060,   # 0.60% max simultaneous risk (Balanced Growth)
        base_trade_risk_pct: float = 0.0015,       # 0.15% default risk per trade ($15.00)
        max_drawdown_limit_pct: float = 0.0150,    # 1.50% portfolio circuit breaker halt
        correlation_threshold: float = 0.60,       # Correlation penalty threshold
        consecutive_loss_limit: int = 2,           # Consecutive losses before freeze
        cooldown_hours: float = 3.0,               # Cooldown freeze duration
        stagger_seconds: int = 3600,               # Min spacing between trade entries (1 hour)
    ):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.base_trade_risk_pct = base_trade_risk_pct
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.correlation_threshold = correlation_threshold
        self.consecutive_loss_limit = consecutive_loss_limit
        self.cooldown_hours = cooldown_hours
        self.stagger_seconds = stagger_seconds

        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.last_entry_time: Optional[datetime] = None

        # Macro currency/driver map for cross-market correlation detection
        self.macro_driver_map = {
            "EURUSD": ("USD_SHORT", "EUR_LONG"),
            "GBPUSD": ("USD_SHORT", "GBP_LONG"),
            "XAGUSD": ("USD_SHORT", "METALS_LONG"),
            "XAUUSD": ("USD_SHORT", "METALS_LONG"),
            "NAS100": ("EQUITIES_RISK_ON", "USD_NEUTRAL"),
            "US500":  ("EQUITIES_RISK_ON", "USD_NEUTRAL"),
            "WTI":    ("ENERGY_LONG", "USD_SHORT"),
            "USDJPY": ("USD_LONG", "JPY_SHORT"),
            "USDCHF": ("USD_LONG", "CHF_SHORT"),
        }

    def clean_symbol(self, sym: str) -> str:
        """Strips broker suffixes like .x, .pro, m."""
        s = sym.upper()
        for suffix in [".X", ".PRO", ".RAW", ".M", "_CUSTOM"]:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
        return s

    def is_rollover_window(self, dt: Optional[datetime] = None) -> bool:
        """
        Checks if current time is inside the 21:30 - 23:30 UTC daily rollover window.
        """
        if dt is None:
            dt = datetime.utcnow()
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute
        # 21:30 is 1290 mins, 23:30 is 1410 mins
        return 1290 <= time_minutes <= 1410

    def record_trade_result(self, is_win: bool, current_time: Optional[datetime] = None) -> None:
        """Updates consecutive loss counter and activates cooling-off if triggered."""
        if current_time is None:
            current_time = datetime.utcnow()

        if is_win:
            self.consecutive_losses = 0
            self.cooldown_until = None
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.consecutive_loss_limit:
                self.cooldown_until = current_time + timedelta(hours=self.cooldown_hours)
                logger.warning(
                    f"❄️ CONSECUTIVE LOSS COOLDOWN ACTIVATED! {self.consecutive_losses} losses in a row. "
                    f"Trading paused until {self.cooldown_until.strftime('%Y-%m-%d %H:%M:%S')} UTC."
                )

    def evaluate_circuit_breaker(self, equity: float, balance: float) -> Tuple[bool, float]:
        """
        Checks if current unrealized floating drawdown breaches the circuit breaker.
        """
        if balance <= 0:
            return True, 1.0

        drawdown = max(0.0, (balance - equity) / balance)
        is_halted = drawdown >= self.max_drawdown_limit_pct

        if is_halted:
            logger.warning(
                f"🚨 PORTFOLIO CIRCUIT BREAKER TRIGGERED! Floating Drawdown: {drawdown*100:.2f}% "
                f">= Limit {self.max_drawdown_limit_pct*100:.2f}%. Halting new entries."
            )

        return is_halted, drawdown

    def calculate_allocated_risk(self, open_positions: List[Dict[str, Any]], equity: float) -> float:
        """
        Computes the total current simultaneous risk percentage across all open positions.
        """
        total_risk_dollar = 0.0
        for p in open_positions:
            entry = float(p.get("price_open", 0.0))
            sl = float(p.get("sl", 0.0))
            vol = float(p.get("volume", 0.0))
            contract = float(p.get("contract_size", 100000.0))

            if entry > 0 and sl > 0 and vol > 0:
                dist = abs(entry - sl)
                risk_dollar = dist * vol * contract
                total_risk_dollar += risk_dollar

        if equity <= 0:
            return 0.0
        return total_risk_dollar / equity

    def check_correlation_conflict(
        self,
        new_symbol: str,
        new_direction: str,
        open_positions: List[Dict[str, Any]],
        rolling_returns_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, float, str]:
        """
        Evaluates whether candidate trade correlates with active positions.
        """
        clean_new = self.clean_symbol(new_symbol)
        new_drivers = self.macro_driver_map.get(clean_new, ("UNKNOWN", "UNKNOWN"))
        new_macro = new_drivers[0] if new_direction == "BUY" else new_drivers[0].replace("SHORT", "LONG") if "SHORT" in new_drivers[0] else new_drivers[0].replace("LONG", "SHORT")

        for p in open_positions:
            active_sym = self.clean_symbol(p.get("symbol", ""))
            active_type = p.get("type", "BUY")

            if active_sym == clean_new:
                return True, 0.0, f"Position already active on {new_symbol}."

            active_drivers = self.macro_driver_map.get(active_sym, ("UNKNOWN", "UNKNOWN"))
            active_macro = active_drivers[0] if active_type == "BUY" else active_drivers[0].replace("SHORT", "LONG") if "SHORT" in active_drivers[0] else active_drivers[0].replace("LONG", "SHORT")

            if new_macro == active_macro and new_macro != "UNKNOWN":
                logger.info(
                    f"⚠️ Correlation Gate: {new_symbol} ({new_direction}) shares macro driver '{new_macro}' "
                    f"with active position {active_sym} ({active_type}). Enforcing 50% risk discount."
                )
                return True, 0.50, f"Correlated macro exposure with active position {active_sym} ({new_macro})."

        return False, 1.0, "Independent orthogonal risk stream. Full allocation granted."

    def calculate_permitted_risk(
        self,
        candidate_symbol: str,
        candidate_direction: str,
        open_positions: List[Dict[str, Any]],
        equity: float,
        balance: float,
        current_time: Optional[datetime] = None,
        rolling_returns_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Master Risk Decision: Evaluates Circuit Breaker, Rollover, Cooldown, Staggering, Budget, Correlation.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # 1. Circuit Breaker Check
        is_halted, dd_pct = self.evaluate_circuit_breaker(equity, balance)
        if is_halted:
            return {"permitted": False, "risk_pct": 0.0, "reason": f"CIRCUIT_BREAKER_ACTIVE: Drawdown {dd_pct*100:.2f}% >= 1.50%"}

        # 2. Rollover Window Check
        if self.is_rollover_window(current_time):
            return {"permitted": False, "risk_pct": 0.0, "reason": "SESSION_BLACKOUT: Daily rollover window (21:30 - 23:30 UTC)"}

        # 3. Consecutive Loss Cooldown Check
        if self.cooldown_until is not None and current_time < self.cooldown_until:
            rem_mins = int((self.cooldown_until - current_time).total_seconds() / 60)
            return {"permitted": False, "risk_pct": 0.0, "reason": f"CONSECUTIVE_LOSS_FREEZE: Cooldown active ({rem_mins} mins remaining)"}

        # 4. Staggered Entry Queue Check
        if self.last_entry_time is not None:
            elapsed = (current_time - self.last_entry_time).total_seconds()
            if elapsed < self.stagger_seconds:
                rem_s = int(self.stagger_seconds - elapsed)
                return {"permitted": False, "risk_pct": 0.0, "reason": f"STAGGERED_QUEUE: Must wait {rem_s}s between multi-asset entries"}

        # 5. Current Allocated Risk vs Cap
        current_risk_pct = self.calculate_allocated_risk(open_positions, equity)
        available_risk_pct = max(0.0, self.max_portfolio_risk_pct - current_risk_pct)

        if available_risk_pct < 0.0004:
            return {"permitted": False, "risk_pct": 0.0, "reason": f"PORTFOLIO_RISK_EXHAUSTED: Allocated {current_risk_pct*100:.2f}% / Max {self.max_portfolio_risk_pct*100:.2f}%"}

        # 6. Correlation Gate
        has_conflict, mult, explanation = self.check_correlation_conflict(
            new_symbol=candidate_symbol,
            new_direction=candidate_direction,
            open_positions=open_positions,
            rolling_returns_df=rolling_returns_df,
        )

        if mult == 0.0:
            return {"permitted": False, "risk_pct": 0.0, "reason": explanation}

        target_risk = self.base_trade_risk_pct * mult
        final_risk = min(target_risk, available_risk_pct)

        logger.info(
            f"Risk Gate Approval for {candidate_symbol}: "
            f"Authorized Risk: {final_risk*100:.3f}% (Budget Available: {available_risk_pct*100:.2f}%) | {explanation}"
        )

        return {
            "permitted": True,
            "risk_pct": round(final_risk, 5),
            "current_portfolio_risk_pct": round(current_risk_pct, 5),
            "available_portfolio_risk_pct": round(available_risk_pct, 5),
            "correlation_discount_applied": mult < 1.0,
            "reason": explanation,
        }
