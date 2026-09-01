"""
PyTorch Lightning Training Pipeline Orchestrator.
Configures EarlyStopping, ModelCheckpoint, gradient clipping, and executes model fitting.
"""
import os
import time
import warnings
import logging
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Suppress cosmetic Swig C-extension deprecation warnings from MetaTrader5 bindings
warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*", category=DeprecationWarning)

# Unlock Tensor Core throughput on Ampere/Ada GPUs (RTX 30xx / 40xx series)
try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = logging.getLogger(__name__)


def _resolve_accelerator() -> str:
    """Auto-detect GPU availability; fall back to CPU gracefully."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — using accelerator='gpu'.")
            return "gpu"
    except Exception:
        pass
    logger.info("No CUDA GPU detected — falling back to accelerator='cpu'.")
    return "cpu"


def run_training_pipeline(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    max_epochs: int = 25,
    checkpoint_dir: str = "./checkpoints",
    monitor_metric: str = "val_loss",
    patience: int = 5,
    gradient_clip_val: float = 1.0,
    accelerator: str = "auto"
) -> pl.Trainer:
    """
    Executes an end-to-end PyTorch Lightning training workflow.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Setup Callbacks
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode="min",
        verbose=True
    )

    # Windows-safe checkpoint strategy:
    # save_top_k=-1 keeps ALL checkpoints (never deletes any).
    # This avoids PermissionError [WinError 32] where Windows locks .ckpt files
    # still held open by the GPU process during checkpoint rotation.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_{epoch:02d}_{val_loss:.4f}",
        save_top_k=-1,          # keep all — Windows cannot safely delete GPU-open files
        save_last=True,         # always save latest epoch too
        monitor=monitor_metric,
        mode="min",
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 2. Configure Trainer — resolve GPU automatically if caller passed "auto"
    resolved_accelerator = _resolve_accelerator() if accelerator == "auto" else accelerator
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=resolved_accelerator,
        devices=1,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        precision="16-mixed" if resolved_accelerator == "gpu" else 32,
    )

    # 3. Execute Fit
    logger.info("Starting PyTorch Lightning training loop...")
    trainer.fit(model, datamodule=datamodule)

    logger.info(f"Training complete. Best model checkpoint saved to: {checkpoint_callback.best_model_path}")
    return trainer
