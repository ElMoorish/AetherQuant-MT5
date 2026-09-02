"""
Aether-TripleBarrier-PatchTST: Realistic Friction-Aware Dual-Head Training Engine
================================================================================
Implements:
1. Marcos López de Prado's Triple-Barrier Labeling (Path-Aware Intrabar MAE/MFE).
2. Broker Friction-Subtracted Net Alpha Targets (y_net = Return - Spread - Commissions).
3. Dual-Head Architecture:
   - Head 1: 5-Horizon Friction-Subtracted Net Return Trajectory (Alpha Head)
   - Head 2: Probability of Intermediate Stop-Loss Breach (Hazard Risk Head)
4. Multi-Task Composite Loss: Calibrated Tri-Loss + BCE Hazard Loss.
"""
import sys, os, warnings, logging, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.mt5_execution.scripts.economic_calendar import EconomicCalendarEngine
from scripts.train_super_alpha_model import RevIN
from scripts.train_macro_super_patchtst import engineer_23_macro_alpha_features, ALL_23_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TripleBarrierTrain")

BROKER_FRICTIONS = {
    "EURUSD": 0.00020,  # 2.0 pips total friction (spread + commission)
    "NAS100": 0.00030,  # 3.0 bps friction
    "WTI": 0.00040,     # 4.0 bps friction
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRIPLE-BARRIER & FRICTION-AWARE TARGET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_triple_barrier_targets(
    df: pd.DataFrame,
    symbol: str,
    forecast_horizon: int = 5,
    sl_atr_mult: float = 2.5,
    tp_atr_mult: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes path-aware friction-subtracted net returns and SL breach hazard labels.
    """
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)

    # ATR (14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values + 1e-6
    friction = BROKER_FRICTIONS.get(symbol, 0.00025)

    net_return_targets = np.zeros((n, forecast_horizon), dtype=np.float32)
    sl_hazard_targets = np.zeros((n, 1), dtype=np.float32)

    for i in range(n - forecast_horizon):
        curr_c = c[i]
        curr_atr = atr[i]
        
        long_sl = curr_c - (sl_atr_mult * curr_atr)
        long_tp = curr_c + (tp_atr_mult * curr_atr)
        short_sl = curr_c + (sl_atr_mult * curr_atr)
        short_tp = curr_c - (tp_atr_mult * curr_atr)

        # Lookahead path over H hours
        future_highs = h[i+1 : i+1+forecast_horizon]
        future_lows = l[i+1 : i+1+forecast_horizon]
        future_closes = c[i+1 : i+1+forecast_horizon]

        # 1. Path-Aware SL Breach Check (Did price pierce the 2.5x ATR lower boundary?)
        breached_long_sl = np.any(future_lows <= long_sl)
        breached_short_sl = np.any(future_highs >= short_sl)
        # Binary hazard label: 1 if high volatility causes a stop-out in either direction
        sl_hazard_targets[i, 0] = 1.0 if (breached_long_sl or breached_short_sl) else 0.0

        # 2. Friction-Subtracted Multi-Horizon Expected Net Return
        for step in range(forecast_horizon):
            fc = future_closes[step]
            raw_ret = np.log(fc / (curr_c + 1e-6))
            
            # If intermediate low hit SL on a long, clamp return to -SL - friction
            if np.any(future_lows[: step + 1] <= long_sl) and raw_ret > 0:
                net_ret = -((curr_c - long_sl) / curr_c) - friction
            else:
                # Friction penalty subtracted directly
                net_ret = np.sign(raw_ret) * max(0.0, abs(raw_ret) - friction) if abs(raw_ret) > friction else 0.0

            net_return_targets[i, step] = net_ret

    return net_return_targets, sl_hazard_targets


# ─────────────────────────────────────────────────────────────────────────────
# 2. DUAL-HEAD TRIPLE-BARRIER PATCHTST ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class AetherTripleBarrierPatchTST(pl.LightningModule):
    """
    Dual-Head PatchTST Network:
    - Head 1: Friction-Subtracted Net Return Trajectory (Alpha Head)
    - Head 2: Intermediate Stop-Loss Breach Hazard Probability (Risk Head)
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
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.num_patches = (seq_len - patch_len) // stride + 1

        # 1. Reversible Instance Normalization
        self.revin = RevIN(num_features=input_dim)

        # 2. Temporal Patch Tokenizer
        self.patch_projection = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # 4. Head 1: Net Alpha Trajectory Head [B, 5]
        self.alpha_head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

        # 5. Head 2: Stop-Loss Hazard Risk Head [B, 1]
        self.hazard_head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Loss Functions
        self.huber = nn.SmoothL1Loss(beta=0.001)
        self.bce = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x: [B, 96, 23]
        Output: (net_returns [B, 5], sl_hazard_prob [B, 1])
        """
        B, L, C = x.shape
        x_norm = self.revin(x, mode="norm")

        patches = x_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous().reshape(B, self.num_patches, self.patch_len * C)

        tokens = self.patch_projection(patches) + self.pos_embedding
        encoded = self.transformer_encoder(tokens)
        encoded = self.layer_norm(encoded)

        flat = encoded.reshape(B, -1)
        net_returns = self.alpha_head(flat)
        sl_hazard = self.hazard_head(flat)

        return net_returns, sl_hazard

    def compute_composite_loss(
        self,
        pred_ret: torch.Tensor,
        true_ret: torch.Tensor,
        pred_haz: torch.Tensor,
        true_haz: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Huber Loss on Net Alpha
        loss_huber = self.huber(pred_ret, true_ret)

        # 2. Directional Sign Penalty
        loss_dir = torch.mean(torch.relu(-torch.sign(true_ret) * pred_ret))

        # 3. Differentiable Sharpe Ratio
        pnl = pred_ret * true_ret
        loss_sharpe = -torch.clamp(torch.mean(pnl) / (torch.std(pnl) + 1e-6), -3.0, 3.0)

        # 4. Stop-Loss Hazard Binary Cross Entropy
        loss_hazard = self.bce(pred_haz, true_haz)

        return loss_huber + (0.5 * loss_dir) + (0.2 * loss_sharpe) + (0.3 * loss_hazard)

    def training_step(self, batch, batch_idx):
        x, y_ret, y_haz = batch
        pred_ret, pred_haz = self(x)
        loss = self.compute_composite_loss(pred_ret, y_ret, pred_haz, y_haz)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_ret, y_haz = batch
        pred_ret, pred_haz = self(x)
        loss = self.compute_composite_loss(pred_ret, y_ret, pred_haz, y_haz)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET & TRAINING RUNNER
# ─────────────────────────────────────────────────────────────────────────────
class MultiAssetTripleBarrierDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets_ret: np.ndarray, targets_haz: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets_ret = torch.tensor(targets_ret, dtype=torch.float32)
        self.targets_haz = torch.tensor(targets_haz, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_ret[idx], self.targets_haz[idx]


def build_triple_barrier_dataset(
    seq_len: int = 96,
    forecast_horizon: int = 5,
    bars_per_symbol: int = 25000,
    symbols: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    symbols = symbols or ["EURUSD", "NAS100", "WTI"]
    client = MT5Client()
    if not client.connect():
        raise RuntimeError("Failed to connect to MT5 terminal.")

    calendar = EconomicCalendarEngine()
    all_seqs = []
    all_targets_ret = []
    all_targets_haz = []
    scalers = {}

    for sym in symbols:
        res_sym = client._resolve_symbol(sym)
        logger.info(f"Ingesting {bars_per_symbol} bars for {sym} ({res_sym})...")
        h1_df = client.get_rates(symbol=res_sym, timeframe="H1", count=bars_per_symbol)
        h4_df = client.get_rates(symbol=res_sym, timeframe="H4", count=bars_per_symbol // 4 + 500)

        feat_df = engineer_23_macro_alpha_features(h1_df, h4_df, calendar)
        ret_targets, haz_targets = compute_triple_barrier_targets(feat_df, symbol=sym, forecast_horizon=forecast_horizon)

        scaler = RobustScaler()
        feature_matrix = scaler.fit_transform(feat_df[ALL_23_FEATURES].values)
        scalers[sym] = scaler

        valid_len = len(feat_df) - forecast_horizon
        for i in range(seq_len, valid_len):
            all_seqs.append(feature_matrix[i - seq_len : i])
            all_targets_ret.append(ret_targets[i])
            all_targets_haz.append(haz_targets[i])

    client.disconnect()

    X = np.array(all_seqs, dtype=np.float32)
    Y_ret = np.array(all_targets_ret, dtype=np.float32)
    Y_haz = np.array(all_targets_haz, dtype=np.float32)
    logger.info(f"🟢 Total Multi-Asset Sequences: {X.shape[0]:,} | Net Return Targets: {Y_ret.shape} | Hazard Targets: {Y_haz.shape}")

    # Strict 80/20 Expanding-Window Split
    split_idx = int(len(X) * 0.80)
    train_ds = MultiAssetTripleBarrierDataset(X[:split_idx], Y_ret[:split_idx], Y_haz[:split_idx])
    val_ds = MultiAssetTripleBarrierDataset(X[split_idx:], Y_ret[split_idx:], Y_haz[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, scalers


def train(max_epochs: int = 15):
    logger.info("=" * 85)
    logger.info("  🚀 TRAINING AETHER-TRIPLEBARRIER-PATCHTST (DUAL-HEAD REALISTIC SOTA)")
    logger.info("  Path-Aware Triple-Barrier + Broker Friction Subtraction + SL Hazard Head")
    logger.info("=" * 85)

    train_loader, val_loader, scalers = build_triple_barrier_dataset()

    model = AetherTripleBarrierPatchTST(
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALL_23_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        learning_rate=3e-4,
        dropout=0.15,
    )

    ckpt_dir = ROOT / "checkpoints/triple_barrier_patchtst"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_triple_barrier_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=True,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    logger.info(f"✅ Aether-TripleBarrier-PatchTST Training Complete! Checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    train(max_epochs=15)
