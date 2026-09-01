"""
Autonomous Multi-Asset MT5 Live Execution Daemon (Balanced Growth Shield Edition)
================================================================================
Universe: EURUSD, XAGUSD, NAS100, WTI
Architecture: MultiAssetSuperPatchTST (RevIN + 18 Alpha Indicators)
Risk Controls:
  - Base Risk: 0.15% ($15.00 per trade on $10,000 equity)
  - Hard Dollar-Risk Ceiling: Enforced on broker min-lot constraints
  - Max Simultaneous Portfolio Exposure: 0.60% ($60.00 max open risk)
  - Rollover Blackout: Prohibit entries during 21:30 - 23:30 UTC
  - Staggered Queue: Minimum 1 hour between multi-asset entries
  - Consecutive Loss Cooldown: 3-hour freeze after 2 consecutive losses
  - Floating Drawdown Circuit Breaker: 1.50% hard halt
  - Magic Number: 10101
"""
import sys, io, os, time, json, signal, argparse, warnings, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*SwigPy.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.risk_manager import RiskManager
from skills.mt5_execution.scripts.order_router import OrderRouter
from skills.mt5_execution.scripts.portfolio_risk_controller import PortfolioRiskController
from scripts.train_super_alpha_model import SuperPatchTST, engineer_18_alpha_features, ALPHA_FEATURES

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

LOG_FILE = ROOT / "scripts/live_daemon.log"
STATE_FILE = ROOT / "scripts/daemon_state.json"
DEFAULT_CKPT = str(ROOT / "checkpoints/multi_asset/best_multi_asset_epoch=12_val_loss=-0.0248.ckpt")


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        FlushFileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("MultiAssetDaemon")


class MultiAssetTradingDaemon:
    """
    Autonomous Concurrent Multi-Asset Execution Daemon with Balanced Growth Risk Controls.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "H1",
        mode: str = "live-demo",
        base_risk_pct: float = 0.0015,
        max_portfolio_risk_pct: float = 0.0060,
        checkpoint_path: str = DEFAULT_CKPT,
        magic_number: int = 10101,
    ):
        self.symbols = [s.upper() for s in (symbols or ["EURUSD", "XAGUSD", "NAS100", "WTI"])]
        self.timeframe = timeframe.upper()
        self.mode = mode.lower()
        self.base_risk_pct = base_risk_pct
        self.checkpoint_path = checkpoint_path
        self.magic_number = magic_number

        self.client = MT5Client()
        self.risk_mgr = RiskManager(client=self.client, default_risk_pct=self.base_risk_pct)
        self.router = OrderRouter(client=self.client)
        self.portfolio_ctrl = PortfolioRiskController(
            max_portfolio_risk_pct=max_portfolio_risk_pct,
            base_trade_risk_pct=base_risk_pct,
            max_drawdown_limit_pct=0.0150,
            correlation_threshold=0.60,
            consecutive_loss_limit=2,
            cooldown_hours=3.0,
            stagger_seconds=3600,  # 1 hour spacing
        )

        self.model: Optional[SuperPatchTST] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.running = False
        self.last_bar_times: Dict[str, str] = {}
        self.state: Dict[str, Any] = {
            "status": "INITIALIZING",
            "tier": "Balanced Growth (Precision Shield)",
            "mode": self.mode,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "base_risk_pct": self.base_risk_pct,
            "max_portfolio_risk_pct": max_portfolio_risk_pct,
            "magic_number": self.magic_number,
            "model_architecture": "MultiAssetSuperPatchTST (Balanced Growth: 0.15% Risk / <1.14% MaxDD)",
            "last_update": datetime.now().isoformat(),
            "active_positions": [],
            "signals": {},
            "performance": {"total_trades": 0, "wins": 0, "losses": 0},
        }

    def connect(self) -> bool:
        connected = self.client.connect()
        if connected:
            acc = self.client.get_account_info()
            logger.info(
                f"MT5 Connected | Account #{acc.get('login')} | "
                f"Balance: {acc.get('currency', 'USD')} {acc.get('balance', 0):,.2f} | "
                f"Leverage: 1:{acc.get('leverage', 0)}"
            )
        return connected

    def load_model(self) -> bool:
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}. Falling back.")
            fallback = str(ROOT / "checkpoints/super_alpha/best_super_patchtst_epoch=09_val_loss=-0.0435.ckpt")
            self.checkpoint_path = fallback

        try:
            self.model = SuperPatchTST.load_from_checkpoint(
                self.checkpoint_path,
                seq_len=96,
                patch_len=16,
                stride=8,
                input_dim=len(ALPHA_FEATURES),
                output_dim=5,
                d_model=128,
                nhead=8,
                num_layers=4,
                learning_rate=3e-4,
                dropout=0.15,
            )
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Multi-Asset Model loaded successfully on {self.device.upper()} from {Path(self.checkpoint_path).name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_open_positions(self) -> List[Dict[str, Any]]:
        positions = []
        if self.client.connected and MT5_AVAILABLE:
            all_pos = mt5.positions_get()
            if all_pos:
                for p in all_pos:
                    if p.magic == self.magic_number:
                        sym_info = mt5.symbol_info(p.symbol)
                        contract = sym_info.trade_contract_size if sym_info else 100000.0
                        positions.append({
                            "ticket": p.ticket,
                            "symbol": p.symbol,
                            "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                            "volume": p.volume,
                            "price_open": p.price_open,
                            "price_current": p.price_current,
                            "sl": p.sl,
                            "tp": p.tp,
                            "profit": p.profit,
                            "contract_size": contract,
                            "magic": p.magic,
                        })
        return positions

    def evaluate_symbol(self, symbol: str, open_positions: List[Dict[str, Any]], equity: float, balance: float) -> Optional[Dict[str, Any]]:
        res_sym = self.client._resolve_symbol(symbol)
        raw_h1 = self.client.get_rates(symbol=res_sym, timeframe=self.timeframe, count=250)
        raw_h4 = self.client.get_rates(symbol=res_sym, timeframe="H4", count=100)

        if len(raw_h1) < 96:
            return None

        feat_df = engineer_18_alpha_features(raw_h1, raw_h4)
        scaler = RobustScaler()
        X = scaler.fit_transform(feat_df[ALPHA_FEATURES].values)

        x_t = torch.tensor(X[-96:], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(x_t).cpu().numpy()[0]

        mean_pred = float(np.mean(preds))
        signal_type = "BUY" if mean_pred > 0.00003 else ("SELL" if mean_pred < -0.00003 else "FLAT")

        bar_time = str(feat_df["time"].iloc[-1]) if "time" in feat_df else datetime.now().isoformat()
        
        # Get live tick
        tick = mt5.symbol_info_tick(res_sym) if (self.client.connected and MT5_AVAILABLE) else None
        current_price = float((tick.ask + tick.bid) / 2.0) if tick else 1.0

        decision = {
            "symbol": symbol,
            "res_symbol": res_sym,
            "bar_time": bar_time,
            "price": current_price,
            "signal": signal_type,
            "forecast_return": round(mean_pred, 6),
            "confidence": round(abs(mean_pred), 6),
        }

        # Check existing positions for this symbol
        active_for_sym = [p for p in open_positions if self.portfolio_ctrl.clean_symbol(p["symbol"]) == symbol]

        if signal_type != "FLAT" and not active_for_sym:
            # Evaluate Portfolio Risk Controller with Precision Safeguards
            risk_decision = self.portfolio_ctrl.calculate_permitted_risk(
                candidate_symbol=res_sym,
                candidate_direction=signal_type,
                open_positions=open_positions,
                equity=equity,
                balance=balance,
                current_time=datetime.now(timezone.utc).replace(tzinfo=None),
            )

            decision["risk_evaluation"] = risk_decision

            if risk_decision["permitted"]:
                auth_risk_pct = risk_decision["risk_pct"]
                raw_sl_pts = self.risk_mgr.calculate_atr_stop_distance(res_sym, self.timeframe, atr_period=14)
                
                # Sizing with Hard Dollar-Risk Ceiling
                lot_size, effective_sl_pts = self.risk_mgr.calculate_lot_size_and_sl(
                    res_sym, sl_points=raw_sl_pts, risk_pct=auth_risk_pct, account_equity=equity
                )
                tp_points = effective_sl_pts * 2.5

                decision["lot_size"] = lot_size
                decision["sl_points"] = effective_sl_pts
                decision["tp_points"] = tp_points
                decision["risk_pct_used"] = auth_risk_pct

                if self.mode == "live-demo":
                    exec_res = self.router.send_market_order(
                        symbol=symbol,
                        order_type=signal_type,
                        volume=lot_size,
                        sl_points=effective_sl_pts,
                        tp_points=tp_points,
                        magic_number=self.magic_number,
                        comment=f"BAL_{symbol}"
                    )
                    decision["execution"] = exec_res
                    self.portfolio_ctrl.last_entry_time = datetime.now(timezone.utc).replace(tzinfo=None)
                    logger.info(f"🚀 LIVE ORDER: {signal_type} {lot_size} {symbol} | SL: {effective_sl_pts}pts | Ticket: {exec_res.get('order')}")
                else:
                    decision["execution"] = {"mode": "PAPER", "status": "SIMULATED", "action": signal_type, "volume": lot_size}
                    self.portfolio_ctrl.last_entry_time = datetime.now(timezone.utc).replace(tzinfo=None)
                    logger.info(f"[PAPER] Simulated {signal_type} {lot_size} {symbol} @ {current_price}")
            else:
                logger.info(f"Entry Gate Blocked for {symbol}: {risk_decision['reason']}")

        return decision

    def run_cycle(self) -> Dict[str, Any]:
        acc = self.client.get_account_info() if self.client.connected else {}
        equity = float(acc.get("equity", 10000.0))
        balance = float(acc.get("balance", 10000.0))
        open_pos = self.get_open_positions()

        cycle_results = {}
        for sym in self.symbols:
            res = self.evaluate_symbol(sym, open_pos, equity, balance)
            if res:
                cycle_results[sym] = res

        # Update telemetry
        self.state["last_update"] = datetime.now().isoformat()
        self.state["active_positions"] = open_pos
        self.state["equity"] = equity
        self.state["balance"] = balance
        self.state["signals"] = cycle_results
        self.save_state()

        return cycle_results

    def save_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def run(self, poll_interval: int = 15) -> None:
        self.running = True
        logger.info("=" * 75)
        logger.info(f"  STARTING MULTI-ASSET TRADING DAEMON [BALANCED GROWTH SHIELD EDITION]")
        logger.info(f"  Universe: {', '.join(self.symbols)} | Timeframe: {self.timeframe}")
        logger.info(f"  Base Risk: {self.base_risk_pct*100:.2f}% ($15/trade) | Portfolio Cap: {self.portfolio_ctrl.max_portfolio_risk_pct*100:.2f}%")
        logger.info(f"  Rollover Blackout: 21:30 - 23:30 UTC | Entry Staggering: 1 Hour")
        logger.info("=" * 75)

        self.connect()
        self.load_model()
        self.state["status"] = "RUNNING"
        self.state["mode"] = self.mode
        self.save_state()

        try:
            while self.running:
                rates = self.client.get_rates(symbol="EURUSD", timeframe=self.timeframe, count=2)
                if len(rates) > 0 and "time" in rates:
                    cur_bar = str(rates["time"].iloc[-1])
                    if cur_bar != self.last_bar_times.get("EURUSD"):
                        logger.info(f"New multi-asset bar detected: {cur_bar}")
                        self.run_cycle()
                        self.last_bar_times["EURUSD"] = cur_bar

                # Trailing stops management
                if self.mode == "live-demo" and self.client.connected:
                    for p in self.get_open_positions():
                        atr_pts = self.risk_mgr.calculate_atr_stop_distance(p["symbol"], self.timeframe)
                        self.router.update_trailing_stop(p["ticket"], p["symbol"], atr_pts * 1.5, step_points=10.0)

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Daemon interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        self.state["status"] = "STOPPED"
        self.save_state()
        self.client.disconnect()
        logger.info("Multi-Asset Trading Daemon safely stopped.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Autonomous MT5 Trading Daemon")
    parser.add_argument("--mode", type=str, choices=["paper", "live-demo"], default="live-demo")
    parser.add_argument("--base-risk", type=float, default=0.0015)
    parser.add_argument("--portfolio-risk-cap", type=float, default=0.0060)
    parser.add_argument("--magic", type=int, default=10101)
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()

    daemon = MultiAssetTradingDaemon(
        mode=args.mode,
        base_risk_pct=args.base_risk,
        max_portfolio_risk_pct=args.portfolio_risk_cap,
        magic_number=args.magic,
    )

    def _sig_handler(sig, frame):
        logger.info("Termination signal received. Shutting down...")
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    daemon.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
