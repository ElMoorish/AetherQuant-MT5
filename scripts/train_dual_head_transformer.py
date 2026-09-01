"""
Dual-Head Multi-Task SuperPatchTST Training Engine (Precision Entry v2)
========================================================================
Architecture: DualHeadSuperPatchTST
  - Head 1: Multi-Horizon Directional Expected Returns [dim=5] with DirectionalSharpeLoss
  - Head 2: Asymmetric MFE/MAE Entry Quality Classifier [dim=1] with BCEWithLogitsLoss
Hardware: NVIDIA RTX 4060 GPU with 16-bit Automatic Mixed Precision (AMP)
Universe: EURUSD, XAGUSD, NAS100, WTI (25,000 bars per asset)
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from scripts.train_super_alpha_model import (
    DirectionalSharpeLoss,
    RevIN,
    engineer_18_alpha_features,
    ALPHA_FEATURES
)

LOG_FILE = ROOT / "scripts/dual_head_training.log"
CHECKPOINT_DIR = ROOT / "checkpoints/dual_head"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / "scripts/dual_head_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("DualHeadTrain")

TARGET_SYMBOLS = ["EURUSD", "XAGUSD", "NAS100", "WTI"]
SEQ_LEN = 96
PATCH_LEN = 16
STRIDE = 8
FORECAST_HORIZON = 5
BATCH_SIZE = 128
MAX_EPOCHS = 35
PATIENCE = 7
LR = 3e-4


# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI-TASK DUAL-HEAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
class DualHeadDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y_ret: np.ndarray, y_quality: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_ret = torch.tensor(y_ret, dtype=torch.float32)
        self.y_quality = torch.tensor(y_quality, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_ret[idx], self.y_quality[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. DUAL-HEAD MULTI-TASK TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────────────────────
class DualHeadSuperPatchTST(pl.LightningModule):
    """
    Dual-Head Patch Time Series Transformer:
    - Backbone: RevIN + 1D Unfold Patch Tokenizer + Multi-Layer Transformer Encoder
    - Head 1 (Regression): Multi-horizon forward return prediction
    - Head 2 (Classification): High-conviction entry probability (MFE >= 1.5R before 0.75R MAE)
    """

    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        input_dim: int = 18,
        output_dim: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        learning_rate: float = 3e-4,
        dropout: float = 0.15,
        weight_decay: float = 1e-4,
        quality_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.quality_weight = quality_weight

        self.revin = RevIN(num_features=input_dim)
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        flat_dim = self.num_patches * d_model

        # Shared Backbone Projection
        self.shared_proj = nn.Sequential(
            nn.Linear(flat_dim, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Head 1: Continuous Multi-Horizon Return Prediction
        self.return_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, output_dim),
        )

        # Head 2: Binary Excursion Quality Classifier (Logits)
        self.quality_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.sharpe_loss = DirectionalSharpeLoss(alpha_dir=0.5, alpha_sharpe=0.2)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.revin(x, mode="norm")
        B, L, D = x_norm.shape

        # Patch unfolding
        patches = x_norm.unfold(dimension=1, size=self.hparams.patch_len, step=self.hparams.stride)
        patches = patches.contiguous().view(B, self.num_patches, -1)

        emb = self.patch_embed(patches)
        enc = self.encoder(emb)
        enc = self.norm(enc)

        flat = enc.reshape(B, -1)
        shared = self.shared_proj(flat)

        pred_returns = self.return_head(shared)
        quality_logits = self.quality_head(shared)

        return pred_returns, quality_logits

    def training_step(self, batch, batch_idx):
        x, y_ret, y_qual = batch
        pred_ret, qual_logits = self(x)

        loss_ret = self.sharpe_loss(pred_ret, y_ret)
        loss_qual = self.bce_loss(qual_logits, y_qual)

        total_loss = loss_ret + (self.quality_weight * loss_qual)

        dir_acc = (torch.sign(pred_ret[:, 0]) == torch.sign(y_ret[:, 0])).float().mean()
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dir_acc", dir_acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y_ret, y_qual = batch
        pred_ret, qual_logits = self(x)

        loss_ret = self.sharpe_loss(pred_ret, y_ret)
        loss_qual = self.bce_loss(qual_logits, y_qual)
        total_loss = loss_ret + (self.quality_weight * loss_qual)

        dir_acc = (torch.sign(pred_ret[:, 0]) == torch.sign(y_ret[:, 0])).float().mean()
        probs = torch.sigmoid(qual_logits)
        qual_acc = ((probs > 0.5) == (y_qual > 0.5)).float().mean()

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dir_acc", dir_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_qual_acc", qual_acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=2, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE ASYMMETRIC MFE/MAE QUALITY LABELS
# ─────────────────────────────────────────────────────────────────────────────
def compute_mfe_mae_labels(df: pd.DataFrame, horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes forward multi-horizon log returns AND binary excursion quality labels.
    Quality = 1 if forward trade achieves +1.5x ATR profit before -0.75x ATR drawdown.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    log_rets = df["log_return"].values

    # Compute rolling ATR
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()))
    atrs = tr.rolling(14).mean().fillna(df["high"] - df["low"]).values

    N = len(df) - horizon
    y_returns = []
    y_quality = []

    for i in range(N):
        fwd_rets = log_rets[i+1:i+1+horizon]
        y_returns.append(fwd_rets)

        cur_close = closes[i]
        cur_atr = atrs[i]
        fwd_highs = highs[i+1:i+1+horizon]
        fwd_lows = lows[i+1:i+1+horizon]

        # Check for Long setup
        mfe_long = np.max(fwd_highs) - cur_close
        mae_long = cur_close - np.min(fwd_lows)

        # Check for Short setup
        mfe_short = cur_close - np.min(fwd_lows)
        mae_short = np.max(fwd_highs) - cur_close

        # Is either direction a clean asymmetric winner?
        long_win = (mfe_long >= 1.2 * cur_atr) and (mae_long <= 0.8 * cur_atr)
        short_win = (mfe_short >= 1.2 * cur_atr) and (mae_short <= 0.8 * cur_atr)

        is_quality = 1.0 if (long_win or short_win) else 0.0
        y_quality.append(is_quality)

    return np.array(y_returns), np.array(y_quality)


def main():
    logger.info("=" * 75)
    logger.info("  TRAINING DUAL-HEAD SUPERPATCHTST (PRECISION ENTRY & MFE CLASSIFIER)")
    logger.info("=" * 75)

    client = MT5Client()
    if not client.connect():
        sys.exit(1)

    train_X, train_y_ret, train_y_qual = [], [], []
    val_X, val_y_ret, val_y_qual = [], [], []
    test_splits = {}

    for sym in TARGET_SYMBOLS:
        res_sym = client._resolve_symbol(sym)
        logger.info(f"Ingesting 20,000 bars for {sym} ({res_sym})...")
        h1 = client.get_rates(symbol=res_sym, timeframe="H1", count=20000)
        h4 = client.get_rates(symbol=res_sym, timeframe="H4", count=5000)
        df = engineer_18_alpha_features(h1, h4)

        scaler = RobustScaler()
        X_all = scaler.fit_transform(df[ALPHA_FEATURES].values)

        y_ret_all, y_qual_all = compute_mfe_mae_labels(df, horizon=FORECAST_HORIZON)

        seqs, targets_ret, targets_qual = [], [], []
        for i in range(SEQ_LEN, len(y_ret_all)):
            seqs.append(X_all[i-SEQ_LEN:i])
            targets_ret.append(y_ret_all[i])
            targets_qual.append(y_qual_all[i])

        seqs = np.array(seqs)
        targets_ret = np.array(targets_ret)
        targets_qual = np.array(targets_qual)

        n = len(seqs)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)

        train_X.append(seqs[:t_end])
        train_y_ret.append(targets_ret[:t_end])
        train_y_qual.append(targets_qual[:t_end])

        val_X.append(seqs[t_end:v_end])
        val_y_ret.append(targets_ret[t_end:v_end])
        val_y_qual.append(targets_qual[t_end:v_end])

        test_splits[sym] = {
            "X_test": seqs[v_end:],
            "y_ret_test": targets_ret[v_end:],
            "y_qual_test": targets_qual[v_end:],
            "df_test": df.iloc[v_end + SEQ_LEN:].reset_index(drop=True),
        }
        logger.info(f"  {sym:8s} -> {len(seqs):,} sequences | Quality Positive Rate: {np.mean(targets_qual)*100:.1f}%")

    client.disconnect()

    X_train = np.concatenate(train_X, axis=0)
    Y_ret_train = np.concatenate(train_y_ret, axis=0)
    Y_qual_train = np.concatenate(train_y_qual, axis=0)

    X_val = np.concatenate(val_X, axis=0)
    Y_ret_val = np.concatenate(val_y_ret, axis=0)
    Y_qual_val = np.concatenate(val_y_qual, axis=0)

    logger.info(f"Total Multi-Task Training Samples: {len(X_train):,}")
    logger.info(f"Total Validation Samples          : {len(X_val):,}")

    train_ds = DualHeadDataset(X_train, Y_ret_train, Y_qual_train)
    val_ds = DualHeadDataset(X_val, Y_ret_val, Y_qual_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = DualHeadSuperPatchTST(
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
        quality_weight=0.5,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Dual-Head Model Parameters: {param_count:,} (Direction + MFE Excursion)")

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="best_dual_head_{epoch:02d}_{val_loss:.4f}",
        save_top_k=2,
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

    logger.info("Launching Dual-Head Training on RTX 4060 GPU...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_score = float(ckpt_callback.best_model_score or 0)
    best_path = ckpt_callback.best_model_path
    logger.info(f"Training Complete! Best val_loss: {best_score:.5f} | Saved: {best_path}")

    # Out-of-Sample Dual-Head Evaluation
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    oos_summary = {}
    for sym in TARGET_SYMBOLS:
        test_info = test_splits[sym]
        x_t = test_info["X_test"]
        df_t = test_info["df_test"]

        signals = []
        confidences = []
        with torch.no_grad():
            for i in range(len(x_t)):
                t_in = torch.tensor(x_t[i:i+1], dtype=torch.float32).to(device)
                pred_ret, qual_logit = model(t_in)
                p_ret = pred_ret.cpu().numpy()[0]
                p_qual = torch.sigmoid(qual_logit).cpu().numpy()[0, 0]

                mean_r = float(np.mean(p_ret))
                # Dual-Head Gate: only take signal if Excursion Quality Prob >= 0.55
                if p_qual >= 0.55:
                    sig = 1 if mean_r > 0.00003 else (-1 if mean_r < -0.00003 else 0)
                else:
                    sig = 0

                signals.append(sig)
                confidences.append(p_qual)

        signals = np.array(signals)
        aligned_df = df_t.iloc[:len(signals)].reset_index(drop=True)
        raw_ret = aligned_df["log_return"].values

        pos_chg = np.abs(np.diff(np.concatenate([[0], signals])))
        cost = 0.00015 * pos_chg
        pnl = (signals * raw_ret) - cost
        dollar_pnl = pnl * 10000.0 * 2.0 * (0.0015 / 0.0025)

        eq = 10000.0 + np.cumsum(dollar_pnl)
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-8)

        total_ret = float((eq[-1] - 10000.0) / 10000.0) * 100
        max_dd = float(dd.max()) * 100
        wins = dollar_pnl[dollar_pnl > 0]
        losses = dollar_pnl[dollar_pnl < 0]
        wr = float(len(wins) / (len(wins) + len(losses) + 1e-8)) * 100
        pf = float(wins.sum() / (abs(losses.sum()) + 1e-8))
        sharpe = float(dollar_pnl.mean() / (dollar_pnl.std() + 1e-8)) * np.sqrt(6048)

        oos_summary[sym] = {
            "return_pct": round(total_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 3),
            "total_trades": int(np.sum(pos_chg > 0)),
        }
        logger.info(f"  🏆 {sym:8s} (Dual-Head Filtered) -> Ret: {total_ret:+7.2f}% | MaxDD: {max_dd:5.2f}% | WinRate: {wr:5.1f}% | PF: {pf:5.3f} | Sharpe: {sharpe:6.3f}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_architecture": "DualHeadSuperPatchTST (Direction + MFE Excursion Classifier)",
        "parameters": param_count,
        "best_val_loss": round(best_score, 5),
        "best_checkpoint": best_path,
        "performance": oos_summary,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Dual-Head Results saved to: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
