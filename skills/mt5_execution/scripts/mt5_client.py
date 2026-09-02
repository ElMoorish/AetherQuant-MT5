"""
MetaTrader 5 Client Interface.
Handles terminal lifecycle, historical data retrieval, tick streaming, and account telemetry.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

# Timeframe mapping helper
TIMEFRAME_MAP = {
    "M1": 1 if not MT5_AVAILABLE else mt5.TIMEFRAME_M1,
    "M5": 5 if not MT5_AVAILABLE else mt5.TIMEFRAME_M5,
    "M15": 15 if not MT5_AVAILABLE else mt5.TIMEFRAME_M15,
    "M30": 30 if not MT5_AVAILABLE else mt5.TIMEFRAME_M30,
    "H1": 16385 if not MT5_AVAILABLE else mt5.TIMEFRAME_H1,
    "H4": 16388 if not MT5_AVAILABLE else mt5.TIMEFRAME_H4,
    "D1": 16408 if not MT5_AVAILABLE else mt5.TIMEFRAME_D1,
}


class MT5Client:
    """Encapsulates MetaTrader 5 terminal connectivity, market data ingestion, and account telemetry."""

    def __init__(self, path: Optional[str] = None, portable: bool = False):
        self.path = path
        self.portable = portable
        self.connected = False

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        """Initializes connection to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 python package is not installed. Running in mock/offline mode.")
            self.connected = False
            return False

        init_kwargs: Dict[str, Any] = {}
        if self.path:
            init_kwargs["path"] = self.path
        if self.portable:
            init_kwargs["portable"] = self.portable

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            self.connected = False
            return False

        if login and password and server:
            authorized = mt5.login(login=login, password=password, server=server)
            if not authorized:
                logger.error(f"MT5 login failed for account {login} on {server}, error: {mt5.last_error()}")
                self.connected = False
                return False

        self.connected = True
        logger.info("Connected to MetaTrader 5 terminal successfully.")
        return True

    def disconnect(self) -> None:
        """Shuts down MT5 terminal connection."""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader 5.")

    def get_account_info(self) -> Dict[str, Any]:
        """Retrieves current account balance, equity, leverage, and margin details."""
        if not MT5_AVAILABLE or not self.connected:
            return {
                "balance": 100000.0,
                "equity": 100000.0,
                "margin": 0.0,
                "free_margin": 100000.0,
                "margin_level": 0.0,
                "leverage": 100,
                "currency": "USD",
            }

        acc = mt5.account_info()
        if acc is None:
            logger.error(f"Failed to fetch account_info, error: {mt5.last_error()}")
            return {}

        return {
            "login": acc.login,
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "margin_level": acc.margin_level,
            "leverage": acc.leverage,
            "currency": acc.currency,
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves specification properties for a given market symbol."""
        if not MT5_AVAILABLE or not self.connected:
            # Standard EURUSD defaults for offline/testing
            return {
                "symbol": symbol,
                "point": 0.00001 if "JPY" not in symbol else 0.001,
                "digits": 5 if "JPY" not in symbol else 3,
                "spread": 10,
                "ask": 1.08500,
                "bid": 1.08490,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_value": 1.0,
                "trade_tick_size": 0.00001,
                "trade_contract_size": 100000.0,
            }

        resolved_symbol = self._resolve_symbol(symbol)
        # Ensure symbol is selected in MarketWatch
        if not mt5.symbol_select(resolved_symbol, True):
            logger.warning(f"Could not select {resolved_symbol} in MarketWatch, returning default spec.")
            return {
                "symbol": resolved_symbol,
                "point": 0.00001,
                "digits": 5,
                "spread": 10,
                "ask": 1.08500,
                "bid": 1.08490,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_value": 1.0,
                "trade_tick_size": 0.00001,
                "trade_contract_size": 100000.0,
            }

        info = mt5.symbol_info(resolved_symbol)
        if info is None:
            logger.warning(f"symbol_info({resolved_symbol}) returned None, using default spec.")
            return {
                "symbol": resolved_symbol,
                "point": 0.00001,
                "digits": 5,
                "spread": 10,
                "ask": 1.08500,
                "bid": 1.08490,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_value": 1.0,
                "trade_tick_size": 0.00001,
                "trade_contract_size": 100000.0,
            }

        return {
            "symbol": info.name,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "ask": info.ask,
            "bid": info.bid,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "trade_contract_size": info.trade_contract_size,
        }

    def _resolve_symbol(self, symbol: str) -> str:
        """Finds matching broker symbol name (e.g. EURUSD -> EURUSD.x)."""
        if not MT5_AVAILABLE or not self.connected:
            return symbol

        # Check direct symbol
        info = mt5.symbol_info(symbol)
        if info is not None:
            return symbol

        # Check common broker suffixes
        suffixes = [".x", "m", ".raw", ".pro", "_i", ".ecn"]
        for sfx in suffixes:
            alt_sym = f"{symbol}{sfx}"
            if mt5.symbol_info(alt_sym) is not None:
                return alt_sym

        # Search symbols
        all_syms = mt5.symbols_get()
        if all_syms:
            for s in all_syms:
                if symbol.lower() in s.name.lower():
                    return s.name

        return symbol

    def get_rates(
        self,
        symbol: str,
        timeframe: str = "M5",
        count: int = 1000,
        start_pos: int = 0
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV bar data and computes stationary log returns.
        Strict 100% Real Broker Data (Zero Silent Fallbacks).
        """
        if not MT5_AVAILABLE or not self.connected:
            raise ConnectionError(f"MT5 terminal is disconnected! Cannot retrieve live rates for {symbol}.")

        resolved_symbol = self._resolve_symbol(symbol)
        mt5.symbol_select(resolved_symbol, True)

        tf = TIMEFRAME_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(resolved_symbol, tf, start_pos, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Failed to fetch rates for {symbol} (resolved: {resolved_symbol}, {timeframe}), "
                f"error: {mt5.last_error()}."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # Compute stationary log returns (Rule A compliance)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        return df

    def get_ticks(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Fetches raw tick stream (Ask, Bid, Last, Flags)."""
        if not MT5_AVAILABLE or not self.connected:
            dates = pd.date_range(end=datetime.now(), periods=count, freq="100ms")
            return pd.DataFrame({
                "time": dates,
                "bid": np.full(count, 1.08490),
                "ask": np.full(count, 1.08500),
                "last": np.full(count, 1.08495),
                "volume": np.ones(count),
                "flags": np.zeros(count),
            })

        ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            logger.error(f"Failed to fetch ticks for {symbol}, error: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
