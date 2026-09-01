"""
Multi-Asset Portfolio Aggregation & Drawdown Analyzer
=====================================================
Calculates the exact synchronized portfolio equity curve and drawdown metrics
when trading EURUSD, XAGUSD, NAS100, and WTI concurrently under varying risk budgets:
- 0.05% per trade ($5.00)
- 0.08% per trade ($8.00)
- 0.10% per trade ($10.00)
- 0.15% per trade ($15.00)
- 0.20% per trade ($20.00)
- 0.25% per trade ($25.00)
"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from scripts.train_super_alpha_model import SuperPatchTST, engineer_18_alpha_features, ALPHA_FEATURES

SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
INITIAL_CAPITAL = 10000.0


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

    # 1. Ingest synchronized data
    asset_pnls = {}
    for sym in SYMBOLS:
        res_sym = client._resolve_symbol(sym)
        raw_h1 = client.get_rates(symbol=res_sym, timeframe="H1", count=4000)
        raw_h4 = client.get_rates(symbol=res_sym, timeframe="H4", count=1000)
        df = engineer_18_alpha_features(raw_h1, raw_h4)

        scaler = RobustScaler()
        X = scaler.fit_transform(df[ALPHA_FEATURES].values)

        # Generate signals on out-of-sample split (last 30% of data)
        n = len(df)
        test_start = int(n * 0.70)

        signals = []
        with torch.no_grad():
            for i in range(test_start, len(df)):
                x_t = torch.tensor(X[i-96:i], dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(x_t).cpu().numpy()[0]
                mean_p = float(np.mean(pred))
                sig = 1 if mean_p > 0.00003 else (-1 if mean_p < -0.00003 else 0)
                signals.append(sig)

        signals = np.array(signals)
        test_df = df.iloc[test_start:].reset_index(drop=True)
        raw_ret = test_df["log_return"].values[:len(signals)]

        pos_chg = np.abs(np.diff(np.concatenate([[0], signals])))
        cost = 0.00015 * pos_chg  # Commission + spread
        unlevered_pnl = (signals * raw_ret) - cost
        asset_pnls[sym] = unlevered_pnl

    client.disconnect()

    # Synchronize length
    min_len = min(len(p) for p in asset_pnls.values())
    pnl_matrix = np.column_stack([asset_pnls[s][:min_len] for s in SYMBOLS])

    # Correlation Matrix between asset strategy returns
    ret_df = pd.DataFrame(pnl_matrix, columns=SYMBOLS)
    corr_matrix = ret_df.corr().round(3).to_dict()

    # Test varying risk allocations
    risk_levels = [0.0005, 0.0008, 0.0010, 0.0015, 0.0020, 0.0025]
    portfolio_results = []

    print("\n" + "=" * 95)
    print("  📊 SYNCHRONIZED MULTI-ASSET PORTFOLIO SIMULATION (4 MARKETS COMBINED)")
    print("=" * 95)
    print(f"{'Risk/Trade':^12} | {'Net Return':^14} | {'Max Drawdown':^15} | {'Sharpe':^10} | {'Win Rate':^10} | {'Profit Factor':^14}")
    print("-" * 95)

    for r in risk_levels:
        # Scale factor relative to 0.25% baseline (multiplier = r * 10000 * 2.0 / 0.0025)
        # Each trade risks r% of equity
        leverage_mult = (r / 0.0025) * 2.0  # Normalized leverage scale
        
        # Portfolio combined return per bar (equal weight across 4 assets)
        port_step_pnl = np.sum(pnl_matrix * (INITIAL_CAPITAL * leverage_mult / len(SYMBOLS)), axis=1)

        eq_curve = INITIAL_CAPITAL + np.cumsum(port_step_pnl)
        peak = np.maximum.accumulate(eq_curve)
        dd_curve = (peak - eq_curve) / (peak + 1e-8)

        total_ret = float((eq_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        max_dd = float(np.max(dd_curve)) * 100
        
        wins = port_step_pnl[port_step_pnl > 0]
        losses = port_step_pnl[port_step_pnl < 0]
        wr = float(len(wins) / (len(wins) + len(losses) + 1e-8)) * 100
        pf = float(wins.sum() / (abs(losses.sum()) + 1e-8))
        sharpe = float(port_step_pnl.mean() / (port_step_pnl.std() + 1e-8)) * np.sqrt(6048)

        row = {
            "risk_per_trade_pct": round(r * 100, 2),
            "max_portfolio_exposure_pct": round(r * 100 * 4, 2),
            "net_return_pct": round(total_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 3),
        }
        portfolio_results.append(row)

        print(
            f"{r*100:6.2f}% ($ {r*INITIAL_CAPITAL:4.1f}) | "
            f"{total_ret:+10.2f}%     | "
            f"{max_dd:10.2f}%      | "
            f"{sharpe:8.3f}   | "
            f"{wr:7.1f}%   | "
            f"{pf:10.3f}"
        )

    print("=" * 95)

    summary = {
        "bars_tested": min_len,
        "symbols": SYMBOLS,
        "strategy_correlation_matrix": corr_matrix,
        "portfolio_risk_curves": portfolio_results,
    }

    out_file = ROOT / "scripts/portfolio_risk_curves.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
