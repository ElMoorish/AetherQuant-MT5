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
from scripts.train_super_alpha_model import SuperPatchTST, engineer_18_alpha_features, ALPHA_FEATURES
from scripts.train_macro_super_patchtst import MacroSuperPatchTST, engineer_23_macro_alpha_features, ALL_23_FEATURES
from sklearn.preprocessing import RobustScaler

MODEL_A_CKPT = ROOT / "checkpoints/multi_asset/best_multi_asset_epoch=12_val_loss=-0.0248.ckpt"
INITIAL_CAPITAL = 10000.0

def find_latest_macro_ckpt():
    ckpt_dir = ROOT / "checkpoints/macro_super_patchtst"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No macro checkpoint found")
    return sorted(ckpts, key=os.path.getmtime)[-1]

def run():
    client = MT5Client()
    if not client.connect():
        print("MT5 connection failed")
        return

    symbols = ["EURUSD", "NAS100", "WTI"]
    feat_19_dfs = {}
    feat_23_dfs = {}
    raw_dfs = {}
    calendar = EconomicCalendarEngine()

    for s in symbols:
        res = client._resolve_symbol(s)
        h1 = client.get_rates(symbol=res, timeframe="H1", count=25000)
        h4 = client.get_rates(symbol=res, timeframe="H4", count=6500)
        feat_19_dfs[s] = engineer_18_alpha_features(h1, h4)
        feat_23_dfs[s] = engineer_23_macro_alpha_features(h1, h4, calendar)
        raw_dfs[s] = h1
    client.disconnect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model A (Standard 19-Channel PatchTST)
    model_a = SuperPatchTST.load_from_checkpoint(
        str(MODEL_A_CKPT),
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALPHA_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
    ).eval().to(device)

    # Load Model B (Macro-Aware 23-Channel SuperPatchTST)
    macro_ckpt = find_latest_macro_ckpt()
    print(f"Loading Macro Model Checkpoint: {macro_ckpt.name}")
    model_b = MacroSuperPatchTST.load_from_checkpoint(
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

    min_len = min(len(feat_19_dfs[s]) for s in symbols)
    aligned_dates = pd.to_datetime(feat_19_dfs["EURUSD"]["time"].iloc[-min_len+96:]).reset_index(drop=True)
    N = len(aligned_dates)

    # Generate Model A predictions
    preds_a = {}
    preds_b = {}
    for s in symbols:
        df19 = feat_19_dfs[s].iloc[-min_len:].reset_index(drop=True)
        scaler19 = RobustScaler()
        X19 = scaler19.fit_transform(df19[ALPHA_FEATURES].values)
        seqs19 = np.array([X19[i-96:i] for i in range(96, len(X19))], dtype=np.float32)
        with torch.no_grad():
            pa = []
            for b in range(0, len(seqs19), 1024):
                bx = torch.tensor(seqs19[b:b+1024]).to(device)
                pa.append(model_a(bx).cpu().numpy()[:, 0])
            preds_a[s] = np.concatenate(pa)

        df23 = feat_23_dfs[s].iloc[-min_len:].reset_index(drop=True)
        scaler23 = RobustScaler()
        X23 = scaler23.fit_transform(df23[ALL_23_FEATURES].values)
        seqs23 = np.array([X23[i-96:i] for i in range(96, len(X23))], dtype=np.float32)
        with torch.no_grad():
            pb = []
            for b in range(0, len(seqs23), 1024):
                bx = torch.tensor(seqs23[b:b+1024]).to(device)
                pb.append(model_b(bx).cpu().numpy()[:, 0])
            preds_b[s] = np.concatenate(pb)

    def evaluate_strategy(preds_dict, model_name):
        pnl_matrix = []
        trades = 0
        wins, losses = 0, 0

        for s in symbols:
            raw_h1 = raw_dfs[s].iloc[-min_len+96:].reset_index(drop=True)
            closes = raw_h1["close"].values
            p = preds_dict[s]
            s_pnl = np.zeros(N)

            in_pos = 0
            entry_p = 0
            entry_bar = 0

            for i in range(N):
                t = aligned_dates.iloc[i]
                is_rollover = (t.hour == 21 and t.minute >= 30) or (t.hour == 22) or (t.hour == 23 and t.minute <= 30)
                is_news, _ = calendar.is_news_blackout(t, pre_window_min=15, post_window_min=30)

                if in_pos == 0:
                    if not (is_rollover or is_news):
                        if p[i] > 0.00030:
                            in_pos = 1
                            entry_p = closes[i]
                            entry_bar = i
                        elif p[i] < -0.00030:
                            in_pos = -1
                            entry_p = closes[i]
                            entry_bar = i
                else:
                    elapsed = i - entry_bar
                    should_exit = (in_pos == 1 and p[i] < -0.00010) or (in_pos == -1 and p[i] > 0.00010) or elapsed >= 5
                    if should_exit:
                        ret = (closes[i] - entry_p) / entry_p if in_pos == 1 else (entry_p - closes[i]) / entry_p
                        cost = 0.00015
                        net_ret = ret - cost
                        s_pnl[i] = net_ret
                        trades += 1
                        if net_ret > 0:
                            wins += 1
                        else:
                            losses += 1
                        in_pos = 0

            pnl_matrix.append(s_pnl)

        pnl_matrix = np.column_stack(pnl_matrix)
        step_pnls = np.sum(pnl_matrix * (INITIAL_CAPITAL * 0.0015 / 0.0025 * 2.0 / len(symbols)), axis=1)

        eq = INITIAL_CAPITAL + np.cumsum(step_pnls)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-8)

        ret = float((eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        max_dd = float(np.max(dd)) * 100
        wr = (wins / (wins + losses + 1e-8)) * 100
        sharpe = float(step_pnls.mean() / (step_pnls.std() + 1e-8)) * np.sqrt(6048)

        return {
            "return": ret,
            "max_dd": max_dd,
            "win_rate": wr,
            "sharpe": sharpe,
            "trades": trades,
            "wins": wins,
            "losses": losses,
        }

    res_a = evaluate_strategy(preds_a, "Model A (Live Baseline)")
    res_b = evaluate_strategy(preds_b, "Model B (Macro-Aware 23-Channel)")

    ret_a_str = f"{res_a['return']:>+8.2f}%"
    ret_b_str = f"{res_b['return']:>+8.2f}%"
    ret_diff = f"{res_b['return'] - res_a['return']:>+8.2f}%"

    dd_a_str = f"{res_a['max_dd']:>5.2f}%"
    dd_b_str = f"{res_b['max_dd']:>5.2f}%"
    dd_diff = f"{res_b['max_dd'] - res_a['max_dd']:>+5.2f}% (Lower is Better)"

    wr_a_str = f"{res_a['win_rate']:>5.1f}%"
    wr_b_str = f"{res_b['win_rate']:>5.1f}%"
    wr_diff = f"{res_b['win_rate'] - res_a['win_rate']:>+5.1f}%"

    sh_a_str = f"{res_a['sharpe']:>5.2f}"
    sh_b_str = f"{res_b['sharpe']:>5.2f}"
    sh_diff = f"{res_b['sharpe'] - res_a['sharpe']:>+5.2f}"

    tr_a_str = f"{res_a['trades']:,}"
    tr_b_str = f"{res_b['trades']:,}"
    tr_diff = f"{res_b['trades'] - res_a['trades']:,} trades"

    print("=" * 88)
    print("  🏆 4-YEAR MACRO-AWARE TRANSFORMER BACKTEST COMPARISON (2022 - 2026)")
    print("=" * 88)
    print(f"{'Performance Metric':<30s} | {'LIVE BASELINE (19-Channel)':<26s} | {'MACRO-AWARE (23-Channel)':<26s} | {'Delta / Gain':<18s}")
    print("-" * 88)
    print(f"{'4-Year Cumulative Return':<30s} | {ret_a_str:<26s} | {ret_b_str:<26s} | {ret_diff}")
    print(f"{'4-Year Maximum Drawdown':<30s} | {dd_a_str:<26s} | {dd_b_str:<26s} | {dd_diff}")
    print(f"{'Win Rate':<30s} | {wr_a_str:<26s} | {wr_b_str:<26s} | {wr_diff}")
    print(f"{'Portfolio Sharpe Ratio':<30s} | {sh_a_str:<26s} | {sh_b_str:<26s} | {sh_diff}")
    print(f"{'Total Realized Trades':<30s} | {tr_a_str:<26s} | {tr_b_str:<26s} | {tr_diff}")
    print("=" * 88)

if __name__ == "__main__":
    run()

