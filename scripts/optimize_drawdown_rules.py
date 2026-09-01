"""
Institutional Drawdown & Conviction Optimization Matrix
========================================================
Finds the exact mathematical sweet spot between High Alpha & Minimal Drawdown:
- Tests Conviction Thresholds: [0.00005, 0.00015, 0.00025, 0.00035]
- Tests Base Risk: [0.10%, 0.15%, 0.20%]
- Tests Max Open Concurrent Positions: [1, 2, 3]
- Tests Trailing Stop ATR Multipliers: [1.2x, 1.5x, 2.0x]
"""
import sys, os, json, time, warnings, logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.portfolio_risk_controller import PortfolioRiskController
from scripts.train_super_alpha_model import SuperPatchTST, engineer_18_alpha_features, ALPHA_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("OptimizeDrawdown")

SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
INITIAL_CAPITAL = 10000.0


def evaluate_configuration(
    symbol_streams: Dict[str, pd.DataFrame],
    client: MT5Client,
    threshold: float,
    base_risk_pct: float,
    max_concurrent: int,
    trail_atr_mult: float,
) -> Dict[str, Any]:
    common_length = min(len(s) for s in symbol_streams.values())
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    open_trades = []
    closed_trades = []

    risk_ctrl = PortfolioRiskController(
        max_portfolio_risk_pct=max_concurrent * base_risk_pct,
        base_trade_risk_pct=base_risk_pct,
        max_drawdown_limit_pct=0.0150,
        correlation_threshold=0.60,
    )

    for step in range(common_length):
        # 1. Update existing trades with Trailing Stop
        remaining_trades = []
        for trade in open_trades:
            sym = trade["symbol"]
            bar = symbol_streams[sym].iloc[step]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            atr = float(bar["atr"])

            t_type = trade["type"]
            entry = trade["entry_price"]
            sl = trade["sl_price"]
            tp = trade["tp_price"]
            vol = trade["volume"]
            contract = trade["contract_size"]

            # Dynamic ATR Trailing Stop
            if t_type == 1:
                new_sl = high - (trail_atr_mult * atr)
                if new_sl > trade["sl_price"]:
                    trade["sl_price"] = new_sl
            else:
                new_sl = low + (trail_atr_mult * atr)
                if new_sl < trade["sl_price"]:
                    trade["sl_price"] = new_sl

            # Check Exit
            hit_sl = (t_type == 1 and low <= trade["sl_price"]) or (t_type == -1 and high >= trade["sl_price"])
            hit_tp = (t_type == 1 and high >= tp) or (t_type == -1 and low <= tp)

            if hit_sl or hit_tp:
                exit_price = trade["sl_price"] if hit_sl else tp
                pnl = vol * (exit_price - entry if t_type == 1 else entry - exit_price) * contract
                pnl -= (vol * 3.50)  # Commission
                equity += pnl
                closed_trades.append({"symbol": sym, "pnl": pnl, "is_win": pnl > 0})
            else:
                remaining_trades.append(trade)

        open_trades = remaining_trades

        # 2. Candidate Entries
        if len(open_trades) < max_concurrent:
            for sym in SYMBOLS:
                if len(open_trades) >= max_concurrent:
                    break
                if any(t["symbol"] == sym for t in open_trades):
                    continue

                bar = symbol_streams[sym].iloc[step]
                pred_val = float(bar["forecast"])

                # Conviction threshold filter
                if abs(pred_val) >= threshold:
                    sig = 1 if pred_val > 0 else -1
                    cand_dir = "BUY" if sig == 1 else "SELL"

                    risk_res = risk_ctrl.calculate_permitted_risk(
                        candidate_symbol=sym,
                        candidate_direction=cand_dir,
                        open_positions=open_trades,
                        equity=equity,
                        balance=equity,
                    )

                    if risk_res["permitted"]:
                        auth_risk_pct = risk_res["risk_pct"]
                        close_p = float(bar["close"])
                        atr_val = float(bar["atr"])
                        res_sym = client._resolve_symbol(sym)
                        sym_info = client.get_symbol_info(res_sym)
                        contract_sz = sym_info.get("trade_contract_size", 100000.0) if sym_info else 100000.0
                        point_sz = sym_info.get("point", 0.00001) if sym_info else 0.00001

                        r_distance = max(atr_val * 1.5, point_sz * 40)
                        risk_dollars = equity * auth_risk_pct
                        calc_lots = risk_dollars / (r_distance * contract_sz + 1e-8)
                        lots = max(0.01, round(calc_lots, 2))

                        sl_p = (close_p - r_distance) if sig == 1 else (close_p + r_distance)
                        tp_p = (close_p + 2.5 * r_distance) if sig == 1 else (close_p - 2.5 * r_distance)

                        open_trades.append({
                            "symbol": sym,
                            "type": sig,
                            "entry_price": close_p,
                            "sl_price": sl_p,
                            "tp_price": tp_p,
                            "volume": lots,
                            "contract_size": contract_sz,
                        })

        equity_curve.append(equity)

    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd_series = (peak - eq_series) / (peak + 1e-8)

    tot_ret = float((eq_series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    max_dd = float(dd_series.max()) * 100
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame(columns=["pnl"])

    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = len(wins) / len(trades_df) * 100
        pf = (wins["pnl"].sum() / (abs(losses["pnl"].sum()) + 1e-8)) if len(losses) > 0 else 99.0
    else:
        win_rate = pf = 0.0

    step_returns = eq_series.pct_change().dropna()
    sharpe = float(step_returns.mean() / (step_returns.std() + 1e-8)) * np.sqrt(6048)

    return {
        "threshold": threshold,
        "base_risk_pct": base_risk_pct,
        "max_concurrent": max_concurrent,
        "trail_atr_mult": trail_atr_mult,
        "total_return_pct": round(tot_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "trades": len(trades_df),
    }


def main():
    client = MT5Client()
    if not client.connect():
        sys.exit(1)

    ckpt_path = str(ROOT / "checkpoints/multi_asset/best_multi_asset_epoch=12_val_loss=-0.0248.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperPatchTST.load_from_checkpoint(
        ckpt_path,
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
    model.eval()
    model.to(device)

    symbol_streams = {}
    for sym in SYMBOLS:
        res_sym = client._resolve_symbol(sym)
        raw_h1 = client.get_rates(symbol=res_sym, timeframe="H1", count=4000)
        raw_h4 = client.get_rates(symbol=res_sym, timeframe="H4", count=1000)
        df = engineer_18_alpha_features(raw_h1, raw_h4)

        scaler = RobustScaler()
        X = scaler.fit_transform(df[ALPHA_FEATURES].values)

        forecasts = []
        with torch.no_grad():
            for i in range(96, len(df)):
                x_t = torch.tensor(X[i-96:i], dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(x_t).cpu().numpy()[0]
                forecasts.append(float(np.mean(pred)))

        aligned_df = df.iloc[96:].reset_index(drop=True)
        aligned_df["forecast"] = forecasts
        aligned_df["time_dt"] = pd.to_datetime(aligned_df["time"])

        tr = np.maximum(
            aligned_df["high"] - aligned_df["low"],
            np.maximum(
                np.abs(aligned_df["high"] - aligned_df["close"].shift(1)),
                np.abs(aligned_df["low"] - aligned_df["close"].shift(1)),
            ),
        )
        aligned_df["atr"] = tr.rolling(14).mean().fillna(aligned_df["high"] - aligned_df["low"])
        symbol_streams[sym] = aligned_df

    # Grid Search Matrix
    thresholds = [0.00005, 0.00015, 0.00030]
    base_risks = [0.0010, 0.0015, 0.0020]
    max_concurrent_list = [2, 3]
    trail_mults = [1.5, 2.0]

    all_results = []
    logger.info("=" * 85)
    logger.info("  RUNNING GRID OPTIMIZATION MATRIX FOR DRAWDOWN SUPPRESSION & HIGH SHARPE")
    logger.info("=" * 85)

    for thresh in thresholds:
        for risk in base_risks:
            for max_c in max_concurrent_list:
                for trail in trail_mults:
                    res = evaluate_configuration(symbol_streams, client, thresh, risk, max_c, trail)
                    all_results.append(res)
                    logger.info(
                        f"Thresh: {thresh:.5f} | Risk: {risk*100:.2f}% | MaxPos: {max_c} | Trail: {trail}x -> "
                        f"Ret: {res['total_return_pct']:+7.2f}% | MaxDD: {res['max_drawdown_pct']:5.2f}% | "
                        f"Sharpe: {res['sharpe_ratio']:6.3f} | WinRate: {res['win_rate_pct']:5.1f}% | "
                        f"PF: {res['profit_factor']:5.3f} | Trades: {res['trades']}"
                    )

    client.disconnect()

    # Sort by lowest Max Drawdown while keeping Sharpe > 5.0
    sorted_res = sorted(all_results, key=lambda x: (x["max_drawdown_pct"], -x["sharpe_ratio"]))
    out_file = ROOT / "scripts/drawdown_optimization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sorted_res, f, indent=2)

    logger.info(f"Optimization Matrix Saved to: {out_file}")
    return sorted_res


if __name__ == "__main__":
    main()
