"""
Unit test suite for Time Series Deep Learning modules.
"""
import warnings
import pytest
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# Suppress cosmetic Swig C-extension deprecation warnings
warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*", category=DeprecationWarning)

# Enable Tensor Core optimisation on Ampere/Ada GPUs (RTX 30xx/40xx)
torch.set_float32_matmul_precision("high")

from skills.time_series_deep_learning.scripts.data_module import TimeSeriesDataModule, SlidingWindowDataset
from skills.time_series_deep_learning.scripts.models import TemporalTransformerForecaster, PatchTSTLightning
from skills.time_series_deep_learning.scripts.train_pipeline import run_training_pipeline


@pytest.fixture
def synthetic_financial_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    ret = np.random.normal(0, 0.01, n)
    vol = np.abs(np.random.normal(0.01, 0.005, n))
    mom = np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({
        "time": dates,
        "log_return": ret,
        "volatility": vol,
        "momentum": mom,
    })
    return df


def test_sliding_window_dataset():
    features = np.random.randn(100, 3)
    targets = np.random.randn(100)
    dataset = SlidingWindowDataset(features, targets, seq_len=20, forecast_horizon=5)

    assert len(dataset) == 100 - 20 - 5 + 1
    x, y = dataset[0]
    assert x.shape == (20, 3)
    assert y.shape == (5,)


def test_time_series_datamodule_setup(synthetic_financial_data):
    dm = TimeSeriesDataModule(
        df=synthetic_financial_data,
        feature_cols=["log_return", "volatility", "momentum"],
        target_col="log_return",
        seq_len=30,
        forecast_horizon=3,
        batch_size=16
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    batch_x, batch_y = next(iter(train_loader))

    assert batch_x.shape == (16, 30, 3)
    assert batch_y.shape == (16, 3)


def test_temporal_transformer_forward():
    model = TemporalTransformerForecaster(
        input_dim=3,
        output_dim=5,
        d_model=32,
        nhead=2,
        num_layers=2
    )
    x = torch.randn(8, 30, 3)  # Batch=8, Seq=30, Dim=3
    out = model(x)
    assert out.shape == (8, 5)


def test_patchtst_forward():
    model = PatchTSTLightning(
        seq_len=30,
        patch_len=6,
        stride=3,
        input_dim=3,
        output_dim=4,
        d_model=32,
        nhead=2,
        num_layers=2
    )
    x = torch.randn(4, 30, 3)
    out = model(x)
    assert out.shape == (4, 4)


def test_trainer_fast_dev_run(synthetic_financial_data):
    dm = TimeSeriesDataModule(
        df=synthetic_financial_data,
        feature_cols=["log_return", "volatility", "momentum"],
        target_col="log_return",
        seq_len=20,
        forecast_horizon=2,
        batch_size=16,
        num_workers=0,  # keep 0 in tests to avoid multiprocessing overhead
    )
    model = TemporalTransformerForecaster(
        input_dim=3,
        output_dim=2,
        d_model=16,
        nhead=2,
        num_layers=1
    )
    # Use run_training_pipeline so GPU resolution + warning suppression are inherited
    trainer = run_training_pipeline(
        model=model,
        datamodule=dm,
        max_epochs=1,
        checkpoint_dir="./checkpoints/test_run",
        monitor_metric="val_loss",
        patience=1,
        gradient_clip_val=1.0,
        accelerator="auto",
    )
    assert trainer is not None
