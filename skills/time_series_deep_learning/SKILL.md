---
name: time-series-deep-learning
description: >-
  PyTorch Lightning sequence modeling, Temporal Transformers, PatchTST,
  multi-horizon forecasting, and strict non-leakage temporal cross-validation.
---

# Time Series Deep Learning Skill

This skill defines the architectural standards for sequence modeling, temporal transformers, and PyTorch Lightning training pipelines tailored for quantitative finance.

---

## 1. Architectural Standards & Core Components

1. **`TimeSeriesDataModule` (`scripts/data_module.py`)**:
   - Implements strict **Rule A (Data Safety)**: No random train/test splitting.
   - Partitions time-series sequentially into expanding-window or `TimeSeriesSplit` folds.
   - Fits standardizers (`StandardScaler`, `RobustScaler`) strictly on the training partition and applies them to validation/test partitions.
   - Builds sliding-window sequence tensors: $[B, L_{in}, D_{features}] \to [B, L_{out}, D_{targets}]$.

2. **`TemporalTransformerForecaster` & `PatchTSTLightning` (`scripts/models.py`)**:
   - `LightningModule` architectures with Multi-Head Attention, learned temporal embeddings, and multi-horizon linear projection heads.
   - Supports directional loss penalizing wrong-way forecast sign errors in addition to MSE/Quantile loss:
     $$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda_{\text{dir}} \cdot \mathbb{I}(\text{sign}(\hat{y}) \neq \text{sign}(y)) \cdot |\hat{y} - y|$$
   - Built-in Cosine Annealing learning rate scheduling and gradient clipping.

3. **`TrainingPipeline` (`scripts/train_pipeline.py`)**:
   - High-throughput Lightning `Trainer` orchestration with `EarlyStopping`, `ModelCheckpoint`, and TensorBoard metrics logging.

---

## 2. Standard Usage Workflow

```python
import pandas as pd
from skills.time_series_deep_learning.scripts.data_module import TimeSeriesDataModule
from skills.time_series_deep_learning.scripts.models import TemporalTransformerForecaster
from skills.time_series_deep_learning.scripts.train_pipeline import run_training_pipeline

# 1. Prepare stationary dataframe
# df must contain stationary features (e.g. log_return, rsi_norm, atr_norm)
data_module = TimeSeriesDataModule(
    df=df,
    feature_cols=["log_return", "volatility", "momentum"],
    target_col="log_return",
    seq_len=60,
    forecast_horizon=5,
    batch_size=64
)

# 2. Instantiate Model
model = TemporalTransformerForecaster(
    input_dim=3,
    output_dim=5,
    d_model=64,
    nhead=4,
    num_layers=3,
    learning_rate=1e-3
)

# 3. Train with EarlyStopping & Checkpointing
trainer = run_training_pipeline(model, data_module, max_epochs=20)
```

---

## 3. Module Directory Structure

```text
skills/time_series_deep_learning/
├── SKILL.md
├── scripts/
│   ├── __init__.py
│   ├── data_module.py      # Zero-leakage temporal PyTorch LightningDataModule
│   ├── models.py           # Temporal Transformer & PatchTST LightningModules
│   └── train_pipeline.py   # Lightning Trainer with callbacks & logging
└── tests/
    ├── __init__.py
    └── test_deep_learning.py # Pytest suite for tensor flows and training steps
```
