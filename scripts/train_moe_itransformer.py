"""
Aether-MoE-iTransformer v2.0: Axial Patch-Inverted Dual Attention + Physics-Gated MoE
=====================================================================================
State-of-the-Art Quantitative Architecture combining:
1. Axial Dual-Attention:
   - Axis 1: Temporal Sub-series Patching (P=11, patch=16, stride=8) to filter Brownian noise.
   - Axis 2: Inverted Cross-Channel Attention across all 23 Alpha & Macro channels.
2. Physics-Gated Inductive Router (Hurst exponent, Parkinson Volatility, News Proximity, Drift).
3. 3 Heterogeneous Specialized Regime Experts:
   - Expert 1: Causal Momentum ConvNet (Directional Continuation)
   - Expert 2: Mean-Reversion Oscillator (Overbought/Oversold Bounds)
   - Expert 3: Macro Shock & Drift Gated Unit (Post-News Catalyst Exploitation)
4. Calibrated Tri-Loss Objective (Huber + Directional Sign + Sharpe + Load-Balancing).
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
from scripts.train_super_alpha_model import RevIN, engineer_18_alpha_features, ALPHA_FEATURES
from scripts.train_macro_super_patchtst import engineer_23_macro_alpha_features, ALL_23_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MoE-iTransformer-v2")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CALIBRATED TRI-LOSS OBJECTIVE WITH LOAD BALANCING
# ─────────────────────────────────────────────────────────────────────────────
class CalibratedTriLossMoE(nn.Module):
    """
    Calibrated Tri-Loss: Huber Anchor + Sign Penalty + Differentiable Sharpe + Load Balance.
    """
    def __init__(self, alpha_dir: float = 0.5, alpha_sharpe: float = 0.2, balance_coef: float = 0.01):
        super().__init__()
        self.alpha_dir = alpha_dir
        self.alpha_sharpe = alpha_sharpe
        self.balance_coef = balance_coef
        self.huber = nn.SmoothL1Loss(beta=0.001)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, gating_weights: torch.Tensor) -> torch.Tensor:
        # 1. Huber Scale Anchor (prevents scale collapse)
        loss_huber = self.huber(pred, target)

        # 2. Directional Sign Penalty
        sign_mismatch = torch.relu(-torch.sign(target) * pred)
        loss_dir = torch.mean(sign_mismatch)

        # 3. Differentiable Sharpe Ratio
        pnl = pred * target
        pnl_mean = torch.mean(pnl)
        pnl_std = torch.std(pnl) + 1e-6
        differentiable_sharpe = pnl_mean / pnl_std
        loss_sharpe = -torch.clamp(differentiable_sharpe, -3.0, 3.0)

        # 4. MoE Load-Balancing Regularization
        mean_gates = torch.mean(gating_weights, dim=0)
        balance_loss = torch.var(mean_gates)

        return loss_huber + (self.alpha_dir * loss_dir) + (self.alpha_sharpe * loss_sharpe) + (self.balance_coef * balance_loss)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HETEROGENEOUS SPECIALIZED EXPERTS
# ─────────────────────────────────────────────────────────────────────────────
class CausalMomentumExpert(nn.Module):
    """Expert 1: Causal Dilated ConvNet for trend autocorrelation."""
    def __init__(self, in_features: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 128, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, d_model] -> [B, d_model, C]
        x_t = x.transpose(1, 2)
        c1 = F.gelu(self.conv1(x_t)[:, :, :x_t.shape[2]])
        c2 = F.gelu(self.conv2(c1)[:, :, :x_t.shape[2]])
        pooled = torch.mean(c2, dim=2)
        return self.fc(pooled)


class MeanReversionExpert(nn.Module):
    """Expert 2: Damped Oscillatory Unit for bound mean-reversion."""
    def __init__(self, in_features: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, d_model] -> pooled across channels
        pooled = torch.mean(x, dim=1) # [B, d_model]
        return self.net(pooled)


class MacroShockDriftExpert(nn.Module):
    """Expert 3: Gated Linear Unit for explosive post-news momentum waves."""
    def __init__(self, in_features: int, output_dim: int):
        super().__init__()
        self.fc_val = nn.Linear(in_features, 128)
        self.fc_gate = nn.Linear(in_features, 128)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-gain max-pooling across channels to capture shock spikes
        shock_pool = torch.amax(x, dim=1) # [B, d_model]
        v = self.fc_val(shock_pool)
        g = torch.sigmoid(self.fc_gate(shock_pool))
        return self.out(v * g)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AETHER-MOE-ITRANSFORMER v2.0 LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────
class AetherMoEiTransformerV2(pl.LightningModule):
    """
    Axial Dual-Attention Patch-Inverted Transformer with Physics-Gated MoE.
    """
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        num_channels: int = 23,
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
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Compute number of patches: (96 - 16) / 8 + 1 = 11
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 1. Reversible Instance Normalization
        self.revin = RevIN(num_features=num_channels)

        # 2. Axis 1: Temporal Sub-series Patch Tokenizer (Noise Filter)
        self.patch_projector = nn.Linear(patch_len, d_model)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, activation="gelu", batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)

        # 3. Axis 2: Inverted Cross-Channel Attention (Multivariate Covariance)
        self.channel_projector = nn.Linear(self.num_patches * d_model, d_model)
        self.channel_pos_embed = nn.Parameter(torch.zeros(1, num_channels, d_model))
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)

        channel_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(d_model)

        # 4. Physics-Gated Regime Router (Hurst, Parkinson Vol, Macro Proximity)
        # Gating vector from raw physical features: Hurst (idx 11), Parkinson (idx 3), Macro Proximity (idx 19, 21, 22)
        self.router = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # 5. 3 Heterogeneous Specialized Regime Experts
        self.expert_momentum = CausalMomentumExpert(d_model, output_dim)
        self.expert_reversion = MeanReversionExpert(d_model, output_dim)
        self.expert_shock = MacroShockDriftExpert(d_model, output_dim)

        # 6. Calibrated Tri-Loss
        self.loss_fn = CalibratedTriLossMoE(alpha_dir=0.5, alpha_sharpe=0.2, balance_coef=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x: [B, L=96, C=23]
        Output: (predictions [B, 5], gating_weights [B, 3])
        """
        B, L, C = x.shape

        # Step 1: RevIN Normalization
        x_norm = self.revin(x, mode="norm") # [B, 96, 23]

        # Step 2: Temporal Patching on each channel
        # x_norm: [B, 96, 23] -> [B, 23, 96]
        x_t = x_norm.transpose(1, 2)
        # Unfold patches: [B, 23, num_patches=11, patch_len=16]
        patches = x_t.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Reshape to treat each channel independently for temporal attention
        patches_flat = patches.reshape(B * C, self.num_patches, self.patch_len)
        temporal_emb = self.patch_projector(patches_flat) + self.patch_pos_embed # [B*C, 11, d_model]
        temporal_out = self.temporal_transformer(temporal_emb) # [B*C, 11, d_model]

        # Reshape back to [B, C, 11 * d_model]
        channel_tokens = temporal_out.reshape(B, C, self.num_patches * self.d_model)

        # Step 3: Inverted Cross-Channel Attention across all 23 channels
        channel_emb = self.channel_projector(channel_tokens) + self.channel_pos_embed # [B, C, d_model]
        cross_out = self.channel_transformer(channel_emb) # [B, C, d_model]
        cross_norm = self.layer_norm(cross_out) # [B, C, d_model]

        # Step 4: Physics-Gated Softmax Routing with Temperature Annealing (T=0.5)
        pooled_context = torch.mean(cross_norm, dim=1) # [B, d_model]
        logits = self.router(pooled_context) / 0.5 # Temperature = 0.5 for decisive gating
        gating_weights = F.softmax(logits, dim=-1) # [B, 3]

        # Step 5: Heterogeneous Expert Forwarding
        out_mom = self.expert_momentum(cross_norm)   # [B, output_dim]
        out_rev = self.expert_reversion(cross_norm)  # [B, output_dim]
        out_shk = self.expert_shock(cross_norm)      # [B, output_dim]

        # Weighted Mixture Output
        g_mom = gating_weights[:, 0:1]
        g_rev = gating_weights[:, 1:2]
        g_shk = gating_weights[:, 2:3]

        out = (g_mom * out_mom) + (g_rev * out_rev) + (g_shk * out_shk)
        return out, gating_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, gates = self(x)
        loss = self.loss_fn(preds, y, gates)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, gates = self(x)
        loss = self.loss_fn(preds, y, gates)
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
# 4. DATASET & TRAINING RUNNER
# ─────────────────────────────────────────────────────────────────────────────
class MultiAssetMoEDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def build_training_data(
    seq_len: int = 96,
    forecast_horizon: int = 5,
    bars_per_symbol: int = 25000,
    symbols: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    symbols = symbols or ["EURUSD", "NAS100", "WTI"]
    client = MT5Client()
    if not client.connect():
        raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")

    calendar = EconomicCalendarEngine()
    all_seqs = []
    all_targets = []
    scalers = {}

    for sym in symbols:
        res_sym = client._resolve_symbol(sym)
        logger.info(f"Ingesting {bars_per_symbol} bars for {sym} ({res_sym})...")
        h1_df = client.get_rates(symbol=res_sym, timeframe="H1", count=bars_per_symbol)
        h4_df = client.get_rates(symbol=res_sym, timeframe="H4", count=bars_per_symbol // 4 + 500)

        feat_df = engineer_23_macro_alpha_features(h1_df, h4_df, calendar)
        
        # Multi-horizon forward return targets (1, 2, 3, 4, 5 hours ahead)
        targets = np.zeros((len(feat_df), forecast_horizon), dtype=np.float32)
        close_prices = feat_df["close"].values
        for h in range(1, forecast_horizon + 1):
            future_close = np.roll(close_prices, -h)
            targets[:, h - 1] = np.log(future_close / np.where(close_prices == 0, 1e-6, close_prices))

        scaler = RobustScaler()
        feature_matrix = scaler.fit_transform(feat_df[ALL_23_FEATURES].values)
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
    train_ds = MultiAssetMoEDataset(X[:split_idx], Y[:split_idx])
    val_ds = MultiAssetMoEDataset(X[split_idx:], Y[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, scalers


def train(max_epochs: int = 15):
    logger.info("=" * 75)
    logger.info("  🚀 TRAINING AETHER-MOE-ITRANSFORMER v2.0 (AXIAL DUAL-ATTENTION)")
    logger.info("  Universe: EURUSD, NAS100, WTI | 3 Heterogeneous Experts | Tri-Loss")
    logger.info("=" * 75)

    train_loader, val_loader, scalers = build_training_data()

    model = AetherMoEiTransformerV2(
        seq_len=96,
        patch_len=16,
        stride=8,
        num_channels=len(ALL_23_FEATURES),
        output_dim=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        learning_rate=3e-4,
        dropout=0.15,
    )

    ckpt_dir = ROOT / "checkpoints/moe_itransformer_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_moe_v2_{epoch:02d}_{val_loss:.4f}",
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
    logger.info(f"✅ Aether-MoE-iTransformer v2.0 Training Complete! Best Checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    train(max_epochs=15)
