"""
Zero-Leakage Time Series DataModule for PyTorch Lightning.
Enforces Rule A: Strictly temporal splitting (never random k-fold),
and fits scalers strictly on training folds before transforming validation/test sets.
"""
import warnings
from typing import List, Optional, Tuple

# Suppress cosmetic Swig C-extension deprecation warnings
warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning.pytorch as pl
    except ImportError:
        pl = None


class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset for sequence forecasting.
    Converts 2D feature matrix [N, D] into 3D sequence tensors [M, seq_len, D]
    and target sequences [M, forecast_horizon].
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 60,
        forecast_horizon: int = 5
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.num_samples = len(features) - seq_len - forecast_horizon + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough data points ({len(features)}) for seq_len={seq_len} "
                f"and forecast_horizon={forecast_horizon}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.forecast_horizon]
        return x, y


BaseDataModule = pl.LightningDataModule if pl is not None else object

class TimeSeriesDataModule(BaseDataModule):
    """
    PyTorch Lightning DataModule handling chronological temporal splits,
    zero-lookahead standard scaling, and DataLoader construction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int = 60,
        forecast_horizon: int = 5,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,        # Safe default for Windows; set 0 to disable multiprocessing
        use_robust_scaler: bool = False
    ):
        if pl is not None:
            super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_robust_scaler = use_robust_scaler

        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.target_scaler = RobustScaler() if use_robust_scaler else StandardScaler()

        self.train_dataset: Optional[SlidingWindowDataset] = None
        self.val_dataset: Optional[SlidingWindowDataset] = None
        self.test_dataset: Optional[SlidingWindowDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits data chronologically and fits scalers strictly on training data.
        """
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]

        # 1. Fit scalers strictly on train_df
        train_features = self.scaler.fit_transform(train_df[self.feature_cols].values)
        val_features = self.scaler.transform(val_df[self.feature_cols].values)
        test_features = self.scaler.transform(test_df[self.feature_cols].values)

        train_target = train_df[[self.target_col]].values
        val_target = val_df[[self.target_col]].values
        test_target = test_df[[self.target_col]].values

        # Optional: target standard scaling
        train_target = self.target_scaler.fit_transform(train_target).flatten()
        val_target = self.target_scaler.transform(val_target).flatten()
        test_target = self.target_scaler.transform(test_target).flatten()

        # 2. Construct datasets
        self.train_dataset = SlidingWindowDataset(
            train_features, train_target, self.seq_len, self.forecast_horizon
        )
        self.val_dataset = SlidingWindowDataset(
            val_features, val_target, self.seq_len, self.forecast_horizon
        )
        self.test_dataset = SlidingWindowDataset(
            test_features, test_target, self.seq_len, self.forecast_horizon
        )

    @staticmethod
    def _pin_memory() -> bool:
        """Enable pin_memory only when a CUDA GPU is available for faster host→GPU transfers."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
            persistent_workers=self.num_workers > 0,
        )
