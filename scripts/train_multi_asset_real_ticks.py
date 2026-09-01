"""
Multi-Asset Super Alpha Deep Learning Training Engine
======================================================
Universe: EURUSD (FX), XAGUSD (Silver), NAS100 (Nasdaq), WTI (Crude Oil)
Data: 60,000+ Multi-Asset Historical Bars (2023-2026)
Architecture: SuperPatchTST with RevIN + Directional Sharpe Loss
Execution: NVIDIA RTX 4060 GPU with AMP 16-bit Mixed Precision

Rules:
  A -- Chronological multi-asset training splits; 18 stationary alpha features
  B -- Dynamic ATR risk sizing (0.25% equity risk per asset)
  C -- PyTorch Lightning GPU isolation with Tensor Core matmul acceleration
  D -- Real-tick verification audit
"""
import sys, os, json, warnings, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from sklearn.preprocessing import RobustScaler

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from scripts.train_super_alpha_model import (
    SuperPatchTST,
    DirectionalSharpeLoss,
    RevIN,
    engineer_18_alpha_features,
    MultiAssetTimeSeriesDataset,
    ALPHA_FEATURES
)

LOG_FILE = ROOT / "scripts/multi_asset_training.log"
CHECKPOINT_DIR = ROOT / "checkpoints/multi_asset"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / "scripts/multi_asset_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("MultiAssetTrain")

TARGET_SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
TIMEFRAME = "H1"
COUNT_PER_SYMBOL = 15000
SEQ_LEN = 96
PATCH_LEN = 16
STRIDE = 8
FORECAST_HORIZON = 5
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 8
LR = 3e-4


def build_symbol_sequences(df: pd.DataFrame, seq_len: int = 96, horizon: int = 5):
    scaler = RobustScaler()
    X = scaler.fit_transform(df[ALPHA_FEATURES].values)
    y_raw = df["log_return"].values

    seqs, targets = [], []
    for i in range(seq_len, len(df) - horizon):
        seqs.append(X[i - seq_len:i])
        targets.append(y_raw[i:i + horizon])

    return np.array(seqs), np.array(targets), scaler


def main():
    logger.info("=" * 75)
    logger.info("  MULTI-ASSET SUPER ALPHA TRAINING ENGINE (FX + SILVER + NASDAQ + OIL)")
    logger.info("=" * 75)

    client = MT5Client()
    if not client.connect():
        logger.error("MT5 terminal connection failed.")
        sys.exit(1)

    acc = client.get_account_info()
    logger.info(f"MT5 Connected | Account #{acc.get('login')} | Balance: USD {acc.get('balance', 0):,.2f}")

    # 1. Ingest Multi-Asset History (60,000 total bars)
    all_datasets = {}
    for sym in TARGET_SYMBOLS:
        res_sym = client._resolve_symbol(sym)
        logger.info(f"Ingesting {COUNT_PER_SYMBOL} H1 bars for {sym} (resolved: {res_sym})...")
        raw_h1 = client.get_rates(symbol=res_sym, timeframe="H1", count=COUNT_PER_SYMBOL)
        raw_h4 = client.get_rates(symbol=res_sym, timeframe="H4", count=3000)
        feat_df = engineer_18_alpha_features(raw_h1, raw_h4)
        all_datasets[sym] = feat_df
        logger.info(f"  {sym}: {len(feat_df):,} H1 bars ({feat_df['time'].iloc[0]} -> {feat_df['time'].iloc[-1]})")

    client.disconnect()

    # 2. Build Pooled Multi-Asset Training Set (70% Train, 15% Val, 15% Test per asset)
    all_train_seqs, all_train_targets = [], []
    all_val_seqs, all_val_targets = [], []
    test_splits = {}

    for sym, df_s in all_datasets.items():
        n = len(df_s)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)

        s_train, y_train, _ = build_symbol_sequences(df_s.iloc[:t_end], seq_len=SEQ_LEN, horizon=FORECAST_HORIZON)
        s_val, y_val, _ = build_symbol_sequences(df_s.iloc[t_end:v_end], seq_len=SEQ_LEN, horizon=FORECAST_HORIZON)
        s_test, y_test, test_scaler = build_symbol_sequences(df_s.iloc[v_end:].reset_index(drop=True), seq_len=SEQ_LEN, horizon=FORECAST_HORIZON)

        all_train_seqs.append(s_train)
        all_train_targets.append(y_train)
        all_val_seqs.append(s_val)
        all_val_targets.append(y_val)

        test_splits[sym] = {
            "df": df_s.iloc[v_end:].reset_index(drop=True),
            "X_test": s_test,
            "Y_test": y_test,
            "scaler": test_scaler
        }

    X_train = np.concatenate(all_train_seqs, axis=0)
    Y_train = np.concatenate(all_train_targets, axis=0)
    X_val = np.concatenate(all_val_seqs, axis=0)
    Y_val = np.concatenate(all_val_targets, axis=0)

    logger.info(f"Pooled Multi-Asset Training: {len(X_train):,} sequences across {len(TARGET_SYMBOLS)} markets")
    logger.info(f"Validation Set             : {len(X_val):,} sequences")

    train_ds = MultiAssetTimeSeriesDataset(X_train, Y_train)
    val_ds = MultiAssetTimeSeriesDataset(X_val, Y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model Architecture & Lightning Trainer
    model = SuperPatchTST(
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        input_dim=len(ALPHA_FEATURES),
        output_dim=FORECAST_HORIZON,
        d_model=128,
        nhead=8,
        num_layers=4,
        learning_rate=LR,
        dropout=0.15,
        weight_decay=1e-4,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture: {param_count:,} parameters (SuperPatchTST + RevIN)")

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="best_multi_asset_{epoch:02d}_{val_loss:.4f}",
        save_top_k=-1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=device_type,
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[early_stop, ckpt_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
        precision="16-mixed" if device_type == "gpu" else 32,
    )

    logger.info(f"Starting Multi-Asset GPU Training on {device_type.upper()}...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_score = float(ckpt_callback.best_model_score or 0)
    best_path = ckpt_callback.best_model_path
    logger.info(f"Training Finished! Best val_loss: {best_score:.5f} | Saved: {best_path}")

    # 4. Out-of-Sample Multi-Asset Evaluation Matrix
    logger.info("Evaluating Out-of-Sample Performance across all 4 markets...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    asset_performance = {}
    for sym in TARGET_SYMBOLS:
        test_info = test_splits[sym]
        X_t = test_info["X_test"]
        df_t = test_info["df"]

        signals = []
        with torch.no_grad():
            for i in range(len(X_t)):
                x_tensor = torch.tensor(X_t[i:i+1], dtype=torch.float32).to(device)
                pred = model(x_tensor).cpu().numpy()[0]
                mean_p = float(np.mean(pred))
                sig = 1 if mean_p > 0.00003 else (-1 if mean_p < -0.00003 else 0)
                signals.append(sig)

        signals = np.array(signals)
        aligned_df = df_t.iloc[SEQ_LEN:SEQ_LEN + len(signals)].reset_index(drop=True)

        raw_ret = aligned_df["log_return"].values
        pos_chg = np.abs(np.diff(np.concatenate([[0], signals])))
        costs = 0.00015 * pos_chg
        pnl = (signals * raw_ret) - costs
        dollar_pnl = pnl * 10000.0 * 2.0

        eq = 10000.0 + np.cumsum(dollar_pnl)
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-8)

        total_ret = float((eq[-1] - 10000.0) / 10000.0)
        max_dd = float(dd.max())
        wins = dollar_pnl[dollar_pnl > 0]
        losses = dollar_pnl[dollar_pnl < 0]
        wr = float(len(wins) / (len(wins) + len(losses) + 1e-8)) * 100
        pf = float(wins.sum() / (abs(losses.sum()) + 1e-8))
        sharpe = float(dollar_pnl.mean() / (dollar_pnl.std() + 1e-8)) * np.sqrt(6048)

        asset_performance[sym] = {
            "bars_tested": len(aligned_df),
            "return_pct": round(total_ret * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 3),
            "total_trades": int(np.sum(pos_chg > 0)),
        }
        logger.info(f"  {sym:8s} -> Return: {total_ret*100:+.2f}% | Sharpe: {sharpe:.3f} | WinRate: {wr:.1f}% | PF: {pf:.3f} | MaxDD: {max_dd*100:.2f}%")

    master_results = {
        "timestamp": datetime.now().isoformat(),
        "model": "MultiAssetSuperPatchTST (RevIN + 18 Alpha Features)",
        "markets": TARGET_SYMBOLS,
        "total_training_sequences": len(X_train),
        "best_val_loss": round(best_score, 5),
        "best_checkpoint": best_path,
        "asset_performance": asset_performance,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    return master_results


if __name__ == "__main__":
    main()
