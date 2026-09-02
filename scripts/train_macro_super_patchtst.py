"""
Macro-Aware SuperPatchTST Training Engine (23-Dimensional Alpha Universe)
========================================================================
Combines 19 Stationary Microstructure & Alpha Features with 4 Real-Time
Macroeconomic Calendar & Post-News Drift Indicators.
"""
import sys, os, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.economic_calendar import EconomicCalendarEngine
from scripts.train_super_alpha_model import (
    RevIN,
    DirectionalSharpeLoss,
    engineer_18_alpha_features,
    ALPHA_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MacroSuperPatchTST")

MACRO_FEATURES = [
    "hours_to_next_tier1",
    "hours_since_last_tier1",
    "pre_news_compression_score",
    "post_news_drift_momentum",
]

ALL_23_FEATURES = ALPHA_FEATURES + MACRO_FEATURES

def engineer_23_macro_alpha_features(h1_df: pd.DataFrame, h4_df: pd.DataFrame, calendar: EconomicCalendarEngine) -> pd.DataFrame:
    """Computes the 19 standard alpha features + 4 stationary macroeconomic calendar features."""
    df = engineer_18_alpha_features(h1_df, h4_df)
    times = pd.to_datetime(df["time"], utc=True)
    
    events_df = calendar.events_df.copy()
    ev_times = pd.to_datetime(events_df["datetime"], utc=True).sort_values().values

    n = len(df)
    log_rets = df["log_return"].values.astype(np.float32)
    t_vals = times.values

    # Vectorized binary search across all 661 macro events
    idx_next = np.searchsorted(ev_times, t_vals, side="right")
    idx_prev = np.searchsorted(ev_times, t_vals, side="right") - 1

    # Next upcoming events
    valid_next = idx_next < len(ev_times)
    dt_next_hours = np.full(n, 48.0, dtype=np.float32)
    if np.any(valid_next):
        dt_next_hours[valid_next] = (ev_times[idx_next[valid_next]] - t_vals[valid_next]) / np.timedelta64(1, "h")
    
    # Preceding events
    valid_prev = idx_prev >= 0
    dt_prev_hours = np.full(n, 48.0, dtype=np.float32)
    if np.any(valid_prev):
        dt_prev_hours[valid_prev] = (t_vals[valid_prev] - ev_times[idx_prev[valid_prev]]) / np.timedelta64(1, "h")

    df["hours_to_next_tier1"] = (np.clip(dt_next_hours, 0.0, 24.0) / 24.0).astype(np.float32)
    df["hours_since_last_tier1"] = (np.clip(dt_prev_hours, 0.0, 24.0) / 24.0).astype(np.float32)
    df["pre_news_compression_score"] = np.exp(-dt_next_hours / 6.0).astype(np.float32)
    df["post_news_drift_momentum"] = (np.exp(-dt_prev_hours / 12.0) * log_rets).astype(np.float32)

    return df



class MacroSuperPatchTST(pl.LightningModule):
    """
    23-Channel Transformer architecture with RevIN and Directional Sharpe Loss.
    """
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        input_dim: int = 23,
        output_dim: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        learning_rate: float = 3e-4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        self.revin = RevIN(num_features=input_dim)
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self.loss_fn = DirectionalSharpeLoss(alpha_dir=0.5, alpha_sharpe=0.2)

    def forward(self, x):
        # x: [B, seq_len, 23]
        x_norm = self.revin(x, mode="norm")
        B, L, C = x_norm.shape

        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            p = x_norm[:, i:i+self.patch_len, :]
            patches.append(p.reshape(B, -1))

        patches = torch.stack(patches, dim=1) # [B, num_patches, patch_len*C]
        h = self.patch_embed(patches) + self.pos_embed
        out = self.transformer(h)
        out_flat = out.reshape(B, -1)
        pred = self.head(out_flat)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]


class MultiAssetMacroDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 96):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len - 5

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        # Multi-horizon cumulative return targets: 1h, 2h, 3h, 4h, 5h
        y_targets = [torch.sum(self.y[idx + self.seq_len : idx + self.seq_len + k + 1]) for k in range(5)]
        return x_seq, torch.tensor(y_targets, dtype=torch.float32)


def main():
    logger.info("=" * 75)
    logger.info("  🚀 TRAINING MACRO-AWARE SUPERPATCHTST (23 ALPHA CHANNELS)")
    logger.info("  Universe: EURUSD, NAS100, WTI (3-Asset Clean Universe)")
    logger.info("=" * 75)

    client = MT5Client()
    if not client.connect():
        logger.error("MT5 connection failed")
        return

    symbols = ["EURUSD", "NAS100", "WTI"]
    feat_dfs = {}
    calendar = EconomicCalendarEngine()

    for s in symbols:
        res = client._resolve_symbol(s)
        logger.info(f"Ingesting 25,000 bars for {s} ({res})...")
        h1 = client.get_rates(symbol=res, timeframe="H1", count=25000)
        h4 = client.get_rates(symbol=res, timeframe="H4", count=6500)
        feat_dfs[s] = engineer_23_macro_alpha_features(h1, h4, calendar)

    client.disconnect()

    # Align multi-asset sequences
    min_len = min(len(feat_dfs[s]) for s in symbols)
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []

    for s in symbols:
        df = feat_dfs[s].iloc[-min_len:].reset_index(drop=True)
        scaler = RobustScaler()
        split_idx = int(len(df) * 0.85)
        
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        X_train = scaler.fit_transform(train_df[ALL_23_FEATURES].values)
        y_train = train_df["log_return"].values
        X_val = scaler.transform(val_df[ALL_23_FEATURES].values)
        y_val = val_df["log_return"].values

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)

    # Concat datasets
    train_datasets = [MultiAssetMacroDataset(all_X_train[i], all_y_train[i]) for i in range(len(symbols))]
    val_datasets = [MultiAssetMacroDataset(all_X_val[i], all_y_val[i]) for i in range(len(symbols))]

    from torch.utils.data import ConcatDataset
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=256, shuffle=False, num_workers=0)

    model = MacroSuperPatchTST(
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALL_23_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        learning_rate=3e-4,
    )

    ckpt_dir = ROOT / "checkpoints/macro_super_patchtst"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_macro_patchtst_epoch={epoch:02d}_val_loss={val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.info(f"✅ Macro-SuperPatchTST Training Complete! Best Checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
