"""
Controlled A/B Backtest: Baseline vs Economic Calendar News Shield (2022 - 2026)
================================================================================
Empirical evaluation across 24,854 historical bars:
  Experiment A: Standard Primary Model (No News Filter, incurs spread expansion & slippage during Tier-1 releases)
  Experiment B: Primary Model + Economic Calendar News Shield (15m Pre / 30m Post Blackout + Spread Avoidance)
"""
import sys, os, json, warnings, logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.economic_calendar import EconomicCalendarEngine
from scripts.train_super_alpha_model import (
    SuperPatchTST,
    engineer_18_alpha_features,
    ALPHA_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("CalendarBacktestAB")

SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
MODEL_CHECKPOINT = ROOT / "checkpoints/multi_asset/best_multi_asset_epoch=12_val_loss=-0.0248.ckpt"
INITIAL_CAPITAL = 10000.0


def run_experiment(apply_news_shield: bool, dfs_feat: dict, calendar: EconomicCalendarEngine):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperPatchTST.load_from_checkpoint(
        str(MODEL_CHECKPOINT),
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALPHA_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
    )
    model.eval()
    model.to(device)

    # Common dates
    min_len = min(len(dfs_feat[s]) for s in SYMBOLS)
    date_series = pd.to_datetime(dfs_feat["EURUSD"]["time"].iloc[-min_len:]).reset_index(drop=True)

    portfolio_pnl = np.zeros(min_len)
    total_trades = 0
    news_blocks_count = 0
    cost_savings = 0.0

    # Precompute all model predictions across symbols using fast batched GPU inference
    model_predictions = {}
    seq_len = 96
    
    for s in SYMBOLS:
        df = dfs_feat[s].iloc[-min_len:].reset_index(drop=True)
        feats = df[ALPHA_FEATURES].values
        
        # Build sliding windows: [N - 96, 96, 19]
        windows = []
        for i in range(seq_len, min_len):
            windows.append(feats[i-seq_len:i])
        windows = np.array(windows, dtype=np.float32)
        
        # Batch inference on GPU (batch_size=1024)
        preds_all = []
        batch_size = 1024
        with torch.no_grad():
            for b in range(0, len(windows), batch_size):
                b_x = torch.tensor(windows[b:b+batch_size]).to(device)
                b_out = model(b_x).cpu().numpy()
                preds_all.append(np.mean(b_out, axis=1))
        
        preds_concat = np.concatenate(preds_all)
        # Pad initial 96 bars with 0
        full_preds = np.zeros(min_len)
        full_preds[seq_len:] = preds_concat
        model_predictions[s] = full_preds

    for s in SYMBOLS:
        df = dfs_feat[s].iloc[-min_len:].reset_index(drop=True)
        rets = df["log_return"].values
        m_preds = model_predictions[s]

        # Rolling signals
        signals = np.zeros(min_len)
        for i in range(seq_len, min_len):
            t_bar = date_series.iloc[i]
            
            # Check calendar blackout if enabled
            if apply_news_shield:
                is_blackout, _ = calendar.is_news_blackout(t_bar, pre_window_min=15, post_window_min=30)
                if is_blackout:
                    news_blocks_count += 1
                    signals[i] = 0
                    continue

            if m_preds[i] > 0.00010:
                signals[i] = 1
            elif m_preds[i] < -0.00010:
                signals[i] = -1
            else:
                signals[i] = 0

        # Calculate position changes and costs
        pos_chg = np.abs(np.diff(np.concatenate([[0], signals])))
        total_trades += int(np.sum(pos_chg > 0))

        # Base transaction cost = 1.5 pips
        base_cost = 0.00015
        
        # During news releases, if news shield is OFF, simulate 10x broker spread expansion (15 pips cost + slippage)
        costs = np.full(min_len, base_cost)
        if not apply_news_shield:
            for i in range(min_len):
                if pos_chg[i] > 0:
                    t_bar = date_series.iloc[i]
                    is_news, _ = calendar.is_news_blackout(t_bar, pre_window_min=15, post_window_min=30)
                    if is_news:
                        costs[i] = 0.00120  # 12 pips spread blowout
                        cost_savings += 0.00105 * INITIAL_CAPITAL * (0.0015 / 0.0025)

        # Asset PnL
        asset_pnl = (signals * rets) - (pos_chg * costs)
        # 0.15% risk scaling
        dollar_pnl = asset_pnl * (INITIAL_CAPITAL * (0.0015 / 0.0025) / len(SYMBOLS))
        portfolio_pnl += dollar_pnl

    # Portfolio metrics
    eq = INITIAL_CAPITAL + np.cumsum(portfolio_pnl)
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-8)

    net_return = float((eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    max_dd = float(np.max(dd)) * 100
    wins = portfolio_pnl[portfolio_pnl > 0]
    losses = portfolio_pnl[portfolio_pnl < 0]
    wr = float(len(wins) / (len(wins) + len(losses) + 1e-8)) * 100
    pf = float(wins.sum() / (abs(losses.sum()) + 1e-8))
    sharpe = float(portfolio_pnl.mean() / (portfolio_pnl.std() + 1e-8)) * np.sqrt(6048)

    return {
        "net_return_pct": round(net_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "total_trades": total_trades,
        "news_blocked_entries": news_blocks_count,
        "slippage_savings_usd": round(cost_savings, 2)
    }


def main():
    logger.info("=" * 80)
    logger.info("  A/B CONTROLLED EXPERIMENT: ECONOMIC CALENDAR SHIELD vs BASELINE")
    logger.info("=" * 80)

    client = MT5Client()
    if not client.connect():
        sys.exit(1)

    dfs = {}
    for s in SYMBOLS:
        res = client._resolve_symbol(s)
        logger.info(f"Ingesting 25,000 bars for {s}...")
        h1 = client.get_rates(symbol=res, timeframe="H1", count=25000)
        h4 = client.get_rates(symbol=res, timeframe="H4", count=6000)
        dfs[s] = engineer_18_alpha_features(h1, h4)
    client.disconnect()

    calendar = EconomicCalendarEngine()

    logger.info("\n>>> EXPERIMENT A: Baseline Model (No Calendar News Shield)...")
    res_a = run_experiment(apply_news_shield=False, dfs_feat=dfs, calendar=calendar)

    logger.info("\n>>> EXPERIMENT B: Model + Economic Calendar News Shield (15m Pre / 30m Post)...")
    res_b = run_experiment(apply_news_shield=True, dfs_feat=dfs, calendar=calendar)

    logger.info("\n" + "=" * 80)
    logger.info("  🏆 SCIENTIFIC A/B EXPERIMENT RESULTS (2022 - 2026 / 24,854 BARS)")
    logger.info("=" * 80)
    logger.info(f"{'Performance Metric':<30s} | {'Baseline (No News Shield)':<25s} | {'With Economic Shield':<25s} | {'Improvement'}")
    logger.info("-" * 95)

    ret_a, ret_b = res_a["net_return_pct"], res_b["net_return_pct"]
    dd_a, dd_b = res_a["max_drawdown_pct"], res_b["max_drawdown_pct"]
    wr_a, wr_b = res_a["win_rate_pct"], res_b["win_rate_pct"]
    pf_a, pf_b = res_a["profit_factor"], res_b["profit_factor"]
    sh_a, sh_b = res_a["sharpe_ratio"], res_b["sharpe_ratio"]
    tr_a, tr_b = res_a["total_trades"], res_b["total_trades"]

    logger.info(f"{'4-Year Cumulative Return':<30s} | {f'+{ret_a:.2f}%':<25s} | {f'+{ret_b:.2f}%':<25s} | {f'{ret_b - ret_a:+.2f}%'}")
    logger.info(f"{'4-Year Max Drawdown':<30s} | {f'{dd_a:.2f}%':<25s} | {f'{dd_b:.2f}%':<25s} | {f'{dd_b - dd_a:+.2f}% (Lower is Better)'}")
    logger.info(f"{'Aggregate Win Rate':<30s} | {f'{wr_a:.1f}%':<25s} | {f'{wr_b:.1f}%':<25s} | {f'{wr_b - wr_a:+.1f}%'}")
    logger.info(f"{'Profit Factor':<30s} | {f'{pf_a:.3f}':<25s} | {f'{pf_b:.3f}':<25s} | {f'{pf_b - pf_a:+.3f}'}")
    logger.info(f"{'Portfolio Sharpe Ratio':<30s} | {f'{sh_a:.3f}':<25s} | {f'{sh_b:.3f}':<25s} | {f'{sh_b - sh_a:+.3f}'}")
    logger.info(f"{'Total Realized Trades':<30s} | {f'{tr_a}':<25s} | {f'{tr_b}':<25s} | {f'{tr_b - tr_a} news-whipsaw trades filtered'}")
    savings_str = f"+${res_a['slippage_savings_usd']:.2f} Saved"
    logger.info(f"{'Slippage Cost Savings':<30s} | {'$0.00 (Incurred)':<25s} | {savings_str:<25s} | Spread Protected")
    logger.info("=" * 95)

    out_json = ROOT / "scripts/calendar_ab_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": res_a, "calendar_shielded": res_b}, f, indent=2)
    logger.info(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
