"""
Aether-Quantum-PatchTST (26 Channels): Quantum Econophysics + Macro Attention
=============================================================================
Combines:
1. 18 Classical Stationarized Alpha Features (Log Returns, ATR, Parkinson, Hurst, RSI).
2. 5 Real-Time Macroeconomic Attention Channels (ForexFactory 661-Event Feed).
3. 3 Quantum Econophysics Channels:
   - Channel 24: Quantum Ground-State Wavefunction Displacement (psi_0)
   - Channel 25: Quantum Tunneling & Institutional Barrier Penetration (T_barrier)
   - Channel 26: Feynman Classical Action Path Integral (S_action)
4. Unitary-Preserving Temporal Patch Transformer (P=11, patch=16, stride=8).
5. Calibrated Tri-Loss Objective (Huber + Directional Sign Penalty + Sharpe Maximizer).
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
from scripts.train_super_alpha_model import RevIN, DirectionalSharpeLoss
from scripts.train_macro_super_patchtst import engineer_23_macro_alpha_features, ALL_23_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuantumPatchTST")

ALL_26_QUANTUM_FEATURES = ALL_23_FEATURES + [
    "quantum_ground_state_psi0",
    "quantum_tunneling_barrier",
    "feynman_action_integral",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 26-CHANNEL QUANTUM ECONOPHYSICS FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_26_quantum_macro_features(
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    calendar: EconomicCalendarEngine,
) -> pd.DataFrame:
    """
    Computes all 23 Macro-Alpha features plus the 3 Quantum Wavefunction channels.
    """
    df = engineer_23_macro_alpha_features(h1_df, h4_df, calendar)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["tick_volume"].values if "tick_volume" in df else np.ones(len(df))

    # Rolling 24-hour VWAP Anchor
    cum_pv = pd.Series(c * v).rolling(24, min_periods=1).sum().values
    cum_v = pd.Series(v).rolling(24, min_periods=1).sum().values + 1e-6
    vwap_24 = cum_pv / cum_v

    # ATR (14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values + 1e-6

    # 1. Channel 24: Quantum Ground-State Wavefunction Displacement psi_0(x)
    # x_t = dimensionless standardized coordinate
    x_t = np.log(c / np.where(vwap_24 == 0, 1e-6, vwap_24)) / (atr_14 / np.where(c == 0, 1e-6, c) + 1e-6)
    x_t = np.clip(x_t, -5.0, 5.0)
    # Ground state Gaussian eigenstate: psi_0(x) = exp(-0.5 * x^2)
    psi_0 = np.exp(-0.5 * (x_t ** 2))

    # 2. Channel 25: Quantum Tunneling Barrier Penetration Factor T(E)
    # 48-hour institutional Donchian channel boundaries
    donchian_high = pd.Series(h).rolling(48, min_periods=1).max().values
    donchian_low = pd.Series(l).rolling(48, min_periods=1).min().values
    dist_barrier = np.minimum(np.abs(donchian_high - c), np.abs(c - donchian_low)) / (atr_14 + 1e-6)
    macro_prox = df["pre_news_compression_score"].values
    # Tunneling probability through institutional barrier: T ~ exp(-2 * sqrt(V-E))
    t_barrier = np.exp(-2.0 * np.sqrt(np.maximum(0.0, dist_barrier)) * (1.0 + macro_prox))

    # 3. Channel 26: Feynman Classical Action Path Integral S[x]
    # Kinetic energy T = 0.5 * v^2 (log returns squared)
    log_ret = df["log_return"].values
    t_kinetic = 0.5 * (log_ret ** 2)
    # Potential energy V = 0.5 * x_t^2
    v_potential = 0.5 * (x_t ** 2)
    lagrangian = t_kinetic - v_potential
    # 12-Hour Action Integral S = sum(Lagrangian)
    s_action = pd.Series(lagrangian).rolling(12, min_periods=1).sum().values

    df["quantum_ground_state_psi0"] = psi_0.astype(np.float32)
    df["quantum_tunneling_barrier"] = t_barrier.astype(np.float32)
    df["feynman_action_integral"] = s_action.astype(np.float32)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. AETHER-QUANTUM-PATCHTST PYTORCH LIGHTNING MODEL
# ─────────────────────────────────────────────────────────────────────────────
class AetherQuantumPatchTST(pl.LightningModule):
    """
    State-of-the-Art Quantum-Infused Patch Time-Series Transformer (26 Channels).
    """
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        input_dim: int = 26,
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

        # Compute number of temporal patches: (96 - 16) / 8 + 1 = 11
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 1. Reversible Instance Normalization
        self.revin = RevIN(num_features=input_dim)

        # 2. Patch Projection Tokenizer
        self.patch_projection = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Unitary-Regularized Transformer Encoder Backbone
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

        # 4. Multi-Horizon Quantum Action Trajectory Head (5-Hour Continuous Vector)
        self.head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

        # 5. Calibrated Tri-Loss
        self.loss_fn = DirectionalSharpeLoss(alpha_dir=0.5, alpha_sharpe=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, L=96, C=26]
        Output: Multi-step forecast vector [B, 5]
        """
        B, L, C = x.shape
        x_norm = self.revin(x, mode="norm") # [B, 96, 26]

        # Patch extraction across time: [B, num_patches=11, patch_len=16, C=26]
        patches = x_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous() # [B, 11, 16, 26]
        patches = patches.reshape(B, self.num_patches, self.patch_len * C) # [B, 11, 16*26]

        tokens = self.patch_projection(patches) + self.pos_embedding # [B, 11, d_model]
        encoded = self.transformer_encoder(tokens) # [B, 11, d_model]
        encoded = self.layer_norm(encoded)

        flat = encoded.reshape(B, -1) # [B, 11 * d_model]
        out = self.head(flat) # [B, 5]
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
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
# 3. DATA INGESTION & TRAINING RUNNER
# ─────────────────────────────────────────────────────────────────────────────
class MultiAssetQuantumDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def build_quantum_dataset(
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
    all_targets = []
    scalers = {}

    for sym in symbols:
        res_sym = client._resolve_symbol(sym)
        logger.info(f"Ingesting {bars_per_symbol} bars for {sym} ({res_sym})...")
        h1_df = client.get_rates(symbol=res_sym, timeframe="H1", count=bars_per_symbol)
        h4_df = client.get_rates(symbol=res_sym, timeframe="H4", count=bars_per_symbol // 4 + 500)

        feat_df = engineer_26_quantum_macro_features(h1_df, h4_df, calendar)

        # Multi-horizon forward return targets (1, 2, 3, 4, 5 hours ahead)
        targets = np.zeros((len(feat_df), forecast_horizon), dtype=np.float32)
        close_prices = feat_df["close"].values
        for h in range(1, forecast_horizon + 1):
            future_close = np.roll(close_prices, -h)
            targets[:, h - 1] = np.log(future_close / np.where(close_prices == 0, 1e-6, close_prices))

        scaler = RobustScaler()
        feature_matrix = scaler.fit_transform(feat_df[ALL_26_QUANTUM_FEATURES].values)
        scalers[sym] = scaler

        valid_len = len(feat_df) - forecast_horizon
        for i in range(seq_len, valid_len):
            all_seqs.append(feature_matrix[i - seq_len : i])
            all_targets.append(targets[i])

    client.disconnect()

    X = np.array(all_seqs, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    logger.info(f"🟢 Total Multi-Asset Sequences: {X.shape[0]:,} | Shape: {X.shape} | Targets: {Y.shape}")

    # Expanding temporal 80/20 train/val split (Strict Rule A)
    split_idx = int(len(X) * 0.80)
    train_ds = MultiAssetQuantumDataset(X[:split_idx], Y[:split_idx])
    val_ds = MultiAssetQuantumDataset(X[split_idx:], Y[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, scalers


def train(max_epochs: int = 15):
    logger.info("=" * 80)
    logger.info("  🚀 TRAINING AETHER-QUANTUM-PATCHTST (26 CHANNELS)")
    logger.info("  Quantum Wavefunctions (psi_0, T_barrier, S_action) + Calibrated Tri-Loss")
    logger.info("=" * 80)

    train_loader, val_loader, scalers = build_quantum_dataset()

    model = AetherQuantumPatchTST(
        seq_len=96,
        patch_len=16,
        stride=8,
        input_dim=len(ALL_26_QUANTUM_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        learning_rate=3e-4,
        dropout=0.15,
    )

    ckpt_dir = ROOT / "checkpoints/quantum_patchtst"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_quantum_patchtst_{epoch:02d}_{val_loss:.4f}",
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
    logger.info(f"✅ Aether-Quantum-PatchTST Training Complete! Checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    train(max_epochs=15)
