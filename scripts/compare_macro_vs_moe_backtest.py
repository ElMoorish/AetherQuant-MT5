"""
4-Year Full-Cycle Comparative Backtest: MacroSuperPatchTST vs Aether-MoE-iTransformer
====================================================================================
Evaluates the frontier Aether-MoE-iTransformer against the MacroSuperPatchTST across 24,854 bars (2022-2026).
"""
import sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.economic_calendar import EconomicCalendarEngine
from scripts.train_macro_super_patchtst import MacroSuperPatchTST, engineer_23_macro_alpha_features, ALL_23_FEATURES
from scripts.train_moe_itransformer import AetherMoEiTransformerV2
from sklearn.preprocessing import RobustScaler

INITIAL_CAPITAL = 10000.0

def find_latest_macro_ckpt():
    ckpt_dir = ROOT / "checkpoints/macro_super_patchtst"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No macro checkpoint found")
    return sorted(ckpts, key=os.path.getmtime)[-1]

def find_latest_moe_ckpt():
    ckpt_dir = ROOT / "checkpoints/moe_itransformer_v2"
    if not ckpt_dir.exists():
        ckpt_dir = ROOT / "checkpoints/moe_itransformer"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No MoE checkpoint found")
    return sorted(ckpts, key=os.path.getmtime)[-1]

def run():
    client = MT5Client()
    if not client.connect():
        print("MT5 connection failed")
        return

    symbols = ["EURUSD", "NAS100", "WTI"]
    feat_dfs = {}
    raw_dfs = {}
    calendar = EconomicCalendarEngine()

    for s in symbols:
        res = client._resolve_symbol(s)
        h1 = client.get_rates(symbol=res, timeframe="H1", count=25000)
        h4 = client.get_rates(symbol=res, timeframe="H4", count=6500)
        feat_dfs[s] = engineer_23_macro_alpha_features(h1, h4, calendar)
        raw_dfs[s] = h1
    client.disconnect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model A (MacroSuperPatchTST)
    macro_ckpt = find_latest_macro_ckpt()
    print(f"Loading MacroSuperPatchTST Checkpoint: {macro_ckpt.name}")
    model_a = MacroSuperPatchTST.load_from_checkpoint(
        str(macro_ckpt),
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALL_23_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
    ).eval().to(device)

    # Load Model B (Aether-MoE-iTransformer v2.0)
    moe_ckpt = find_latest_moe_ckpt()
    print(f"Loading Aether-MoE-iTransformer Checkpoint: {moe_ckpt.name}")
    model_b = AetherMoEiTransformerV2.load_from_checkpoint(
        str(moe_ckpt),
        seq_len=96,
        patch_len=16,
        stride=8,
        num_channels=len(ALL_23_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
    ).eval().to(device)


    min_len = min(len(feat_dfs[s]) for s in symbols)
    aligned_dates = pd.to_datetime(feat_dfs["EURUSD"]["time"].iloc[-min_len+96:]).reset_index(drop=True)
    N = len(aligned_dates)

    # Generate predictions for both models
    preds_a = {}
    preds_b = {}
    for s in symbols:
        df23 = feat_dfs[s].iloc[-min_len:].reset_index(drop=True)
        scaler23 = RobustScaler()
        X23 = scaler23.fit_transform(df23[ALL_23_FEATURES].values)
        seqs23 = np.array([X23[i-96:i] for i in range(96, len(X23))], dtype=np.float32)

        with torch.no_grad():
            # Model A (PatchTST)
            pa = []
            for b in range(0, len(seqs23), 1024):
                bx = torch.tensor(seqs23[b:b+1024]).to(device)
                pa.append(model_a(bx).cpu().numpy()[:, 0])
            preds_a[s] = np.concatenate(pa)

            # Model B (MoE-iTransformer)
            pb = []
            for b in range(0, len(seqs23), 1024):
                bx = torch.tensor(seqs23[b:b+1024]).to(device)
                out, _ = model_b(bx)
                pb.append(out.cpu().numpy()[:, 0])
            preds_b[s] = np.concatenate(pb)

    def simulate(preds_dict):
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0.0
        equity_curve = [equity]
        trades = 0
        wins = 0
        losses = 0

        active_pos = {s: None for s in symbols} # {"side", "entry_price", "bars_held", "sl", "tp"}
        
        atr_dict = {}
        close_dict = {}
        for s in symbols:
            df = raw_dfs[s].iloc[-min_len+96:].reset_index(drop=True)
            close_dict[s] = df["close"].values
            # ATR computation
            h, l, c = df["high"].values, df["low"].values, df["close"].values
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            tr[0] = h[0] - l[0]
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_dict[s] = atr

        for t in range(N - 1):
            for s in symbols:
                p = preds_dict[s][t]
                curr_c = close_dict[s][t]
                next_c = close_dict[s][t+1]
                atr = atr_dict[s][t]
                pos = active_pos[s]

                # Model Dynamic Exit & Risk Logic
                if pos is not None:
                    pos["bars_held"] += 1
                    side = pos["side"]
                    
                    # Target or Reversal Exit
                    should_exit = False
                    if side == "BUY" and (p < -0.00020 or pos["bars_held"] >= 10):
                        should_exit = True
                    elif side == "SELL" and (p > 0.00020 or pos["bars_held"] >= 10):
                        should_exit = True

                    # Price movement
                    ret = (next_c - curr_c) / curr_c if side == "BUY" else (curr_c - next_c) / curr_c
                    pnl = equity * 0.0015 * (ret / (atr / curr_c + 1e-6))
                    
                    equity += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades += 1

                    if should_exit:
                        active_pos[s] = None

                # Entry
                elif pos is None:
                    if p > 0.000300:
                        active_pos[s] = {"side": "BUY", "entry_price": curr_c, "bars_held": 0}
                    elif p < -0.000300:
                        active_pos[s] = {"side": "SELL", "entry_price": curr_c, "bars_held": 0}

            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            equity_curve.append(equity)

        total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        
        eq_series = pd.Series(equity_curve)
        pct_rets = eq_series.pct_change().dropna()
        sharpe = (pct_rets.mean() / (pct_rets.std() + 1e-9)) * np.sqrt(24 * 365) if len(pct_rets) > 0 else 0.0

        return {
            "total_return": total_ret,
            "max_dd": max_dd * 100.0,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "trades": trades,
        }

    res_a = simulate(preds_a)
    res_b = simulate(preds_b)

    print("=" * 95)
    print("  🏆 4-YEAR FRONTIER MODEL SHOWDOWN: MacroSuperPatchTST vs Aether-MoE-iTransformer (2022-2026)")
    print("=" * 95)
    print(f"{'Performance Metric':<30s} | {'MacroSuperPatchTST':<26s} | {'Aether-MoE-iTransformer':<26s} | {'Delta / Gain':<16s}")
    print("-" * 95)
    print(f"{'4-Year Cumulative Return':<30s} | {res_a['total_return']:>+10.2f}%              | {res_b['total_return']:>+10.2f}%              | {res_b['total_return'] - res_a['total_return']:>+10.2f}%")
    print(f"{'4-Year Maximum Drawdown':<30s} | {res_a['max_dd']:>10.2f}%              | {res_b['max_dd']:>10.2f}%              | {res_b['max_dd'] - res_a['max_dd']:>+10.2f}%")
    print(f"{'Win Rate':<30s} | {res_a['win_rate']:>10.1f}%              | {res_b['win_rate']:>10.1f}%              | {res_b['win_rate'] - res_a['win_rate']:>+10.1f}%")
    print(f"{'Portfolio Sharpe Ratio':<30s} | {res_a['sharpe']:>10.2f}               | {res_b['sharpe']:>10.2f}               | {res_b['sharpe'] - res_a['sharpe']:>+10.2f}")
    print(f"{'Total Realized Trades':<30s} | {res_a['trades']:>10,d}               | {res_b['trades']:>10,d}               | {res_b['trades'] - res_a['trades']:>+10,d} trades")
    print("=" * 95)

if __name__ == "__main__":
    run()
