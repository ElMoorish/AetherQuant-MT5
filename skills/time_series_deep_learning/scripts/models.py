"""
PyTorch Lightning Sequence Models for Quantitative Forecasting.
Includes Temporal Transformer and PatchTST architectures with multi-horizon heads
and directional penalization losses.
"""
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning.pytorch as pl
    except ImportError:
        pl = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order awareness."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DirectionalLoss(nn.Module):
    """
    Combined MSE loss and Directional Sign Penalty.
    Penalizes wrong-way price direction forecasts.
    Formula:
        L = MSE(y_hat, y) + lambda_dir * mean( max(0, -sign(y_hat) * sign(y)) * |y_hat - y| )
    """

    def __init__(self, lambda_dir: float = 0.5):
        super().__init__()
        self.lambda_dir = lambda_dir

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(y_hat, y)
        sign_mismatch = (torch.sign(y_hat) != torch.sign(y)).float()
        directional_penalty = torch.mean(sign_mismatch * torch.abs(y_hat - y))
        return mse + self.lambda_dir * directional_penalty


BaseLightningModule = pl.LightningModule if pl is not None else nn.Module

class TemporalTransformerForecaster(BaseLightningModule):
    """
    Temporal Transformer LightningModule for multi-horizon financial forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_dir: float = 0.5
    ):
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        self.criterion = DirectionalLoss(lambda_dir=lambda_dir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D_in]
        x_emb = self.input_projection(x)
        x_pe = self.pos_encoder(x_emb)
        encoded = self.transformer_encoder(x_pe)
        pooled = encoded.mean(dim=1)
        out = self.head(pooled)  # [B, output_dim]
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if hasattr(self, "log"):
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        val_mse = F.mse_loss(y_hat, y)
        dir_acc = (torch.sign(y_hat) == torch.sign(y)).float().mean()

        if hasattr(self, "log"):
            self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
            self.log("val_mse", val_mse, on_epoch=True)
            self.log("val_dir_acc", dir_acc, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        if hasattr(self, "log"):
            self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }


class PatchTSTLightning(BaseLightningModule):
    """
    Patch Time Series Transformer (PatchTST) LightningModule.
    Segments sequence into patches of length P with stride S for robust representations.
    """

    def __init__(
        self,
        seq_len: int = 60,
        patch_len: int = 12,
        stride: int = 6,
        input_dim: int = 3,
        output_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()
        self.patch_len = patch_len
        self.stride = stride
        self.learning_rate = learning_rate

        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        self.criterion = DirectionalLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x.shape
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]
            patch_flat = patch.reshape(batch_size, -1)
            patches.append(patch_flat)

        patches_tensor = torch.stack(patches, dim=1)
        emb = self.patch_embedding(patches_tensor)
        emb_pe = self.pos_encoder(emb)
        encoded = self.transformer(emb_pe)
        out = self.head(encoded)
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if hasattr(self, "log"):
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dir_acc = (torch.sign(y_hat) == torch.sign(y)).float().mean()
        if hasattr(self, "log"):
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_dir_acc", dir_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
