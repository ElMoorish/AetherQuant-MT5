"""
Order Router & Execution Engine (Precision Routing Edition)
============================================================
Enforces Rule B guardrails: Mandatory hard Stop Loss (SL) and Take Profit (TP),
slippage bounds, optimal pullback limit order placement, dynamic trailing stops, and robust retry loops.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from .mt5_client import MT5Client, MT5_AVAILABLE

if MT5_AVAILABLE:
    import MetaTrader5 as mt5
else:
    mt5 = None

logger = logging.getLogger(__name__)


class OrderRouter:
    """Routes orders safely to MT5 terminal with validation, SL/TP enforcement, and retries."""

    def __init__(self, client: MT5Client, max_retries: int = 3, retry_delay_sec: float = 0.5):
        self.client = client
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    def send_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl_points: float,
        tp_points: float,
        slippage_points: int = 20,
        magic_number: int = 10101,
        comment: str = "KDENSE_AI_EXEC"
    ) -> Dict[str, Any]:
        """
        Executes a market BUY or SELL order with mandatory hard SL and TP.

        Raises:
            ValueError: If sl_points <= 0 or tp_points <= 0 (Rule B enforcement).
        """
        if sl_points <= 0 or tp_points <= 0:
            raise ValueError(
                f"Rule B Violation: Every order MUST specify positive SL and TP. Got sl_points={sl_points}, tp_points={tp_points}"
            )

        order_type_upper = order_type.upper()
        if order_type_upper not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid order type: {order_type}. Must be 'BUY' or 'SELL'.")

        if not MT5_AVAILABLE or not self.client.connected:
            logger.warning(f"Simulating market {order_type_upper} order on {symbol} (offline mode).")
            return {
                "retcode": 10009,
                "deal": 123456,
                "order": 654321,
                "volume": volume,
                "price": 1.08500 if order_type_upper == "BUY" else 1.08490,
                "sl": 1.08250 if order_type_upper == "BUY" else 1.08740,
                "tp": 1.09000 if order_type_upper == "BUY" else 1.07990,
                "comment": f"MOCK_DONE: {comment}",
            }

        resolved_symbol = self.client._resolve_symbol(symbol) if hasattr(self.client, "_resolve_symbol") else symbol
        sym_info = self.client.get_symbol_info(resolved_symbol)
        if not sym_info:
            return {"retcode": -1, "error": f"Failed to retrieve symbol specifications for {symbol}"}

        point = sym_info["point"]
        digits = sym_info["digits"]

        for attempt in range(1, self.max_retries + 1):
            tick = mt5.symbol_info_tick(resolved_symbol)
            if tick is None:
                logger.warning(f"[Attempt {attempt}/{self.max_retries}] Tick info unavailable for {resolved_symbol}")
                time.sleep(self.retry_delay_sec)
                continue

            if order_type_upper == "BUY":
                price = tick.ask
                sl_price = round(price - (sl_points * point), digits)
                tp_price = round(price + (tp_points * point), digits)
                action_type = mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                sl_price = round(price + (sl_points * point), digits)
                tp_price = round(price - (tp_points * point), digits)
                action_type = mt5.ORDER_TYPE_SELL

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": resolved_symbol,
                "volume": volume,
                "type": action_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": slippage_points,
                "magic": magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                logger.error(f"[Attempt {attempt}/{self.max_retries}] order_send returned None. Error: {mt5.last_error()}")
                time.sleep(self.retry_delay_sec)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Order executed successfully: {order_type_upper} {volume} {symbol} @ {price} | "
                    f"SL: {sl_price} | TP: {tp_price} | Ticket: {result.order}"
                )
                return {
                    "retcode": result.retcode,
                    "deal": result.deal,
                    "order": result.order,
                    "volume": result.volume,
                    "price": result.price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "comment": result.comment,
                }
            elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF]:
                logger.warning(f"[Attempt {attempt}/{self.max_retries}] Price requote ({result.retcode}). Retrying...")
                time.sleep(self.retry_delay_sec)
            else:
                logger.error(f"Order failed permanently with retcode {result.retcode}: {result.comment}")
                return {"retcode": result.retcode, "error": result.comment}

        return {"retcode": -1, "error": f"Exceeded max retries ({self.max_retries}) executing {order_type_upper} on {symbol}"}

    def send_limit_pullback_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl_points: float,
        tp_points: float,
        pullback_points: float,
        expiration_hours: float = 4.0,
        magic_number: int = 10101,
        comment: str = "PULLBACK_LIMIT"
    ) -> Dict[str, Any]:
        """
        Places a Limit Order at a favorable pullback price with mandatory hard SL and TP.
        For BUY: Buy Limit = Ask - pullback_points (buys on dip)
        For SELL: Sell Limit = Bid + pullback_points (sells on rally)
        """
        if sl_points <= 0 or tp_points <= 0:
            raise ValueError(f"Rule B Violation: sl_points={sl_points}, tp_points={tp_points} must be > 0")

        order_type_upper = order_type.upper()
        if order_type_upper not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid order type: {order_type}. Must be 'BUY' or 'SELL'.")

        if not MT5_AVAILABLE or not self.client.connected:
            return {
                "retcode": 10009,
                "order": 789101,
                "volume": volume,
                "price": 1.08300,
                "sl": 1.08100,
                "tp": 1.08800,
                "comment": f"MOCK_LIMIT: {comment}"
            }

        resolved_symbol = self.client._resolve_symbol(symbol) if hasattr(self.client, "_resolve_symbol") else symbol
        sym_info = self.client.get_symbol_info(resolved_symbol)
        if not sym_info:
            return {"retcode": -1, "error": f"Symbol specs unavailable for {symbol}"}

        point = sym_info["point"]
        digits = sym_info["digits"]

        tick = mt5.symbol_info_tick(resolved_symbol)
        if tick is None:
            return {"retcode": -1, "error": f"Tick unavailable for {resolved_symbol}"}

        if order_type_upper == "BUY":
            limit_price = round(tick.ask - (pullback_points * point), digits)
            sl_price = round(limit_price - (sl_points * point), digits)
            tp_price = round(limit_price + (tp_points * point), digits)
            action_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            limit_price = round(tick.bid + (pullback_points * point), digits)
            sl_price = round(limit_price + (sl_points * point), digits)
            tp_price = round(limit_price - (tp_points * point), digits)
            action_type = mt5.ORDER_TYPE_SELL_LIMIT

        exp_time = int((datetime.utcnow() + timedelta(hours=expiration_hours)).timestamp())

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": resolved_symbol,
            "volume": volume,
            "type": action_type,
            "price": limit_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": exp_time,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else mt5.last_error()
            logger.warning(f"Limit order failed ({err}). Falling back to market order.")
            return self.send_market_order(symbol, order_type, volume, sl_points, tp_points, magic_number=magic_number, comment=comment)

        logger.info(f"Limit Order Placed: {order_type_upper} LIMIT {volume} {symbol} @ {limit_price} | Ticket: {result.order}")
        return {
            "retcode": result.retcode,
            "order": result.order,
            "volume": result.volume,
            "price": limit_price,
            "sl": sl_price,
            "tp": tp_price,
            "comment": result.comment,
        }

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modifies the Stop Loss of an open position to an exact price."""
        if not MT5_AVAILABLE or not self.client.connected:
            return True

        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return False

        pos = positions[0]
        sym_info = self.client.get_symbol_info(pos.symbol)
        digits = sym_info["digits"] if sym_info else 5
        rounded_sl = round(new_sl, digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": rounded_sl,
            "tp": pos.tp,
        }
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🛡️ POSITION SL MODIFIED | Ticket #{ticket} ({pos.symbol}) -> New SL: {rounded_sl}")
            return True
        else:
            err = res.comment if res else mt5.last_error()
            logger.warning(f"Failed to modify SL for Ticket #{ticket}: {err}")
            return False

    def update_trailing_stop(self, ticket: int, symbol: str, trailing_distance_points: float, step_points: float = 10.0) -> bool:
        if not MT5_AVAILABLE or not self.client.connected:
            return True

        resolved_symbol = self.client._resolve_symbol(symbol) if hasattr(self.client, "_resolve_symbol") else symbol
        sym_info = self.client.get_symbol_info(resolved_symbol)
        if not sym_info:
            return False

        point = sym_info["point"]
        digits = sym_info["digits"]

        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return False

        pos = positions[0]
        tick = mt5.symbol_info_tick(resolved_symbol)
        if not tick:
            return False

        cur_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        dist_price = trailing_distance_points * point
        step_price = step_points * point

        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = round(cur_price - dist_price, digits)
            if new_sl > pos.price_open and new_sl > (pos.sl + step_price):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": resolved_symbol,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                res = mt5.order_send(request)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Trailing SL updated for ticket {ticket} -> {new_sl}")
                    return True
        elif pos.type == mt5.POSITION_TYPE_SELL:
            new_sl = round(cur_price + dist_price, digits)
            if (pos.sl == 0 or new_sl < pos.sl - step_price) and new_sl < pos.price_open:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": resolved_symbol,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                res = mt5.order_send(request)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Trailing SL updated for ticket {ticket} -> {new_sl}")
                    return True

        return False

    def close_position(self, ticket: int) -> Dict[str, Any]:
        """
        Closes an open position at the current market price with retry handling.
        """
        if not MT5_AVAILABLE or not self.client.connected:
            return {"retcode": 10009, "comment": f"MOCK_CLOSED ticket {ticket}"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return {"retcode": -1, "error": f"Position ticket {ticket} not found."}

        pos = positions[0]
        sym_info = self.client.get_symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick or not sym_info:
            return {"retcode": -1, "error": f"Symbol or tick unavailable for {pos.symbol}"}

        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "MODEL_DYNAMIC_EXIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(1, self.max_retries + 1):
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position #{ticket} ({pos.symbol}) closed successfully @ {price:.5f}")
                return {"retcode": 10009, "deal": result.deal, "order": result.order}
            else:
                comment = result.comment if result else "None"
                logger.warning(f"[Attempt {attempt}/{self.max_retries}] Close failed: {comment}. Retrying...")
                time.sleep(self.retry_delay_sec)

        return {"retcode": -1, "error": f"Failed to close position #{ticket}"}

