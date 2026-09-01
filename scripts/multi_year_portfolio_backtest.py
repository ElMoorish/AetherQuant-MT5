"""
Multi-Year Institutional Portfolio Backtest Engine (4 Full Years: 2022 - 2026)
==============================================================================
Evaluates 25,000 H1 bars across EURUSD, XAGUSD, NAS100, and WTI across multiple market regimes:
- 2022 Fed rate hikes & FX volatility
- 2023 Tech equity rally & banking crisis
- 2024 Precious metals & energy geopolitical swings
- 2025-2026 Multi-asset trends
"""
import sys, os, json, warnings, time
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from scripts.train_super_alpha_model import SuperPatchTST, engineer_18_alpha_features, ALPHA_FEATURES

SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
INITIAL_CAPITAL = 10000.0


def main():
    print("=" * 85)
    print("  🚀 4-YEAR MULTI-ASSET INSTITUTIONAL PORTFOLIO BACKTEST (2022 - 2026)")
    print("=" * 85)

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

    # 1. Ingest 25,000 bars per asset
    raw_dfs = {}
    feat_dfs = {}
    for sym in SYMBOLS:
        res_sym = client._resolve_symbol(sym)
        print(f"Ingesting 25,000 bars for {sym} ({res_sym})...")
        h1 = client.get_rates(symbol=res_sym, timeframe="H1", count=25000)
        h4 = client.get_rates(symbol=res_sym, timeframe="H4", count=6500)
        df = engineer_18_alpha_features(h1, h4)
        raw_dfs[sym] = h1
        feat_dfs[sym] = df

    client.disconnect()

    # 2. Build aligned multi-year dataset
    asset_pnls = {}
    asset_dates = {}

    for sym in SYMBOLS:
        df = feat_dfs[sym]
        scaler = RobustScaler()
        X = scaler.fit_transform(df[ALPHA_FEATURES].values)

        signals = []
        batch_size = 256
        x_batches = []
        
        for i in range(96, len(df)):
            x_batches.append(X[i-96:i])

        x_batches = np.array(x_batches)
        
        with torch.no_grad():
            for b in range(0, len(x_batches), batch_size):
                t_batch = torch.tensor(x_batches[b:b+batch_size], dtype=torch.float32).to(device)
                preds = model(t_batch).cpu().numpy()
                mean_preds = np.mean(preds, axis=1)
                sigs = np.where(mean_preds > 0.00003, 1, np.where(mean_preds < -0.00003, -1, 0))
                signals.extend(sigs)

        signals = np.array(signals)
        aligned_df = df.iloc[96:].reset_index(drop=True)
        raw_ret = aligned_df["log_return"].values

        pos_chg = np.abs(np.diff(np.concatenate([[0], signals])))
        cost = 0.00015 * pos_chg
        unlevered_pnl = (signals * raw_ret) - cost

        asset_pnls[sym] = unlevered_pnl
        asset_dates[sym] = aligned_df["time"].values

    # Common length
    min_len = min(len(p) for p in asset_pnls.values())
    pnl_matrix = np.column_stack([asset_pnls[s][-min_len:] for s in SYMBOLS])
    date_series = pd.Series(pd.to_datetime(asset_dates["EURUSD"][-min_len:]))

    start_date = str(date_series.iloc[0])
    end_date = str(date_series.iloc[-1])
    total_days = (date_series.iloc[-1] - date_series.iloc[0]).days
    total_years = total_days / 365.25

    print(f"\nAligned Historical Span: {start_date} to {end_date}")
    print(f"Total Duration          : {total_days:,} Calendar Days ({total_years:.2f} Years / {min_len:,} H1 bars)\n")

    # 3. Simulate across Risk Tiers over the Full 4-Year Period
    risk_tiers = [
        ("Ultra-Conservative", 0.0005, "$5.00"),
        ("Conservative", 0.0008, "$8.00"),
        ("Target Balance", 0.0010, "$10.00"),
        ("Balanced Growth", 0.0015, "$15.00"),
        ("Maximum Alpha", 0.0025, "$25.00"),
    ]

    tier_results = []

    print("-" * 105)
    print(f"{'Risk Tier':<20} | {'Risk/Trade':<10} | {'4-Year Return':<14} | {'Annualized CAGR':<15} | {'Max Drawdown':<14} | {'Sharpe':<8} | {'WinRate':<8}")
    print("-" * 105)

    for tier_name, r, dollar_str in risk_tiers:
        leverage_mult = (r / 0.0025) * 2.0
        step_pnls = np.sum(pnl_matrix * (INITIAL_CAPITAL * leverage_mult / len(SYMBOLS)), axis=1)

        eq = INITIAL_CAPITAL + np.cumsum(step_pnls)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-8)

        total_ret = float((eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        cagr = ((eq[-1] / INITIAL_CAPITAL) ** (1.0 / total_years) - 1.0) * 100
        max_dd = float(np.max(dd)) * 100

        wins = step_pnls[step_pnls > 0]
        losses = step_pnls[step_pnls < 0]
        wr = float(len(wins) / (len(wins) + len(losses) + 1e-8)) * 100
        pf = float(wins.sum() / (abs(losses.sum()) + 1e-8))
        sharpe = float(step_pnls.mean() / (step_pnls.std() + 1e-8)) * np.sqrt(6048)

        # Yearly Breakdown
        yearly_df = pd.DataFrame({"pnl": step_pnls, "year": date_series.dt.year})
        yearly_returns = {}
        for y, group in yearly_df.groupby("year"):
            y_ret = (group["pnl"].sum() / INITIAL_CAPITAL) * 100
            yearly_returns[int(y)] = round(y_ret, 2)

        tier_results.append({
            "tier_name": tier_name,
            "risk_per_trade_pct": r * 100,
            "four_year_net_return_pct": round(total_ret, 2),
            "annualized_cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 3),
            "yearly_returns": yearly_returns,
        })

        print(
            f"{tier_name:<20} | {r*100:5.2f}%     | "
            f"{total_ret:+10.2f}%     | "
            f"{cagr:11.2f}% /yr   | "
            f"{max_dd:10.2f}%     | "
            f"{sharpe:6.2f} | "
            f"{wr:5.1f}%"
        )

    print("-" * 105)

    # 4. Out-of-Sample Split Performance (Last 10,000 bars strictly unseen during training)
    oos_len = 10000
    oos_matrix = pnl_matrix[-oos_len:]
    oos_dates = date_series.iloc[-oos_len:]
    oos_years = (oos_dates.iloc[-1] - oos_dates.iloc[0]).days / 365.25

    print("\n" + "=" * 85)
    print(f"  🔍 STRICT OUT-OF-SAMPLE TEST (Last {oos_len:,} Bars: {str(oos_dates.iloc[0])[:10]} to {str(oos_dates.iloc[-1])[:10]} / {oos_years:.2f} Years)")
    print("=" * 85)

    oos_results = []
    for tier_name, r, _ in risk_tiers:
        leverage_mult = (r / 0.0025) * 2.0
        step_pnls = np.sum(oos_matrix * (INITIAL_CAPITAL * leverage_mult / len(SYMBOLS)), axis=1)

        eq = INITIAL_CAPITAL + np.cumsum(step_pnls)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-8)

        total_ret = float((eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        cagr = ((eq[-1] / INITIAL_CAPITAL) ** (1.0 / oos_years) - 1.0) * 100
        max_dd = float(np.max(dd)) * 100
        sharpe = float(step_pnls.mean() / (step_pnls.std() + 1e-8)) * np.sqrt(6048)

        oos_results.append({
            "tier_name": tier_name,
            "risk_pct": r * 100,
            "oos_net_return_pct": round(total_ret, 2),
            "oos_cagr_pct": round(cagr, 2),
            "oos_max_drawdown_pct": round(max_dd, 2),
            "oos_sharpe": round(sharpe, 3),
        })

        print(
            f"{tier_name:<20} -> Net Return: {total_ret:+8.2f}% | CAGR: {cagr:6.2f}%/yr | "
            f"MaxDD: {max_dd:5.2f}% | Sharpe: {sharpe:5.2f}"
        )

    print("=" * 85)

    full_report = {
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "total_years": round(total_years, 2),
        "bars_tested": min_len,
        "symbols": SYMBOLS,
        "four_year_tiers": tier_results,
        "out_of_sample_tiers": oos_results,
    }

    out_file = ROOT / "scripts/multi_year_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return full_report


if __name__ == "__main__":
    main()
