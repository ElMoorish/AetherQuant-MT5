"""
MT5 Live Data Training Pipeline
=================================
Pulls real OHLCV bars from MetaTrader 5, engineers stationary features,
trains the Temporal Transformer on GPU (RTX 4060), evaluates with SHAP,
and fits the survival model on simulated trade durations derived from
actual volatility patterns.

Windows Note: stdout/stderr are reconfigured to UTF-8 at startup to prevent
UnicodeEncodeError from PyTorch Lightning's emoji log messages (cp1252 codec
cannot encode 💡 U+1F4A1 without this fix).
"""
import sys
import io
# Force UTF-8 on Windows stdout/stderr before any imports emit emoji characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

Rule A  — TimeSeriesSplit / chronological splits only, scalers fit on train only
Rule B  — 0.25% risk per trade (applied in orchestrator)
Rule D  — SHAP feature attribution + stationarity audit
"""
import warnings
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ── Suppress cosmetic warnings ────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

# ── Tensor Core unlock (RTX 4060 Ada) ────────────────────────────────────────
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scripts/live_training.log", mode="w"),
    ],
)
logger = logging.getLogger("MT5LiveTraining")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.mt5_execution.scripts.mt5_client import MT5Client
from skills.time_series_deep_learning.scripts.data_module import TimeSeriesDataModule
from skills.time_series_deep_learning.scripts.models import (
    TemporalTransformerForecaster,
    PatchTSTLightning,
)
from skills.time_series_deep_learning.scripts.train_pipeline import run_training_pipeline
from skills.survival_ml_interpretability.scripts.survival_pipeline import TradeSurvivalPipeline
from skills.survival_ml_interpretability.scripts.shap_diagnostics import SHAPDiagnosticsAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS    = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME  = "H1"
COUNT      = 5000
SEQ_LEN    = 60
HORIZON    = 5
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE   = 12
D_MODEL    = 128
NHEAD      = 8
NUM_LAYERS = 4
LR         = 3e-4

FEATURE_COLS = ["log_return", "volatility_14", "momentum_10",
                "high_low_ratio", "rsi_14", "atr_norm"]
TARGET_COL   = "log_return"

CHECKPOINT_ROOT = Path("checkpoints/mt5_live")
RESULTS_PATH    = Path("scripts/live_training_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (Rule A — stationary features only)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["volatility_14"] = df["log_return"].rolling(14).std()
    df["momentum_10"]   = df["log_return"].rolling(10).sum()
    df["high_low_ratio"]= (df["high"] - df["low"]) / df["close"]

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi_14"] = (100 - 100 / (1 + rs) - 50) / 50  # centred & bounded [-1, 1]

    tr1  = df["high"] - df["low"]
    tr2  = (df["high"] - df["close"].shift(1)).abs()
    tr3  = (df["low"]  - df["close"].shift(1)).abs()
    atr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df["atr_norm"] = atr / df["close"]

    df = df.iloc[30:].reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_survival_analysis(df: pd.DataFrame) -> float:
    logger.info("[SURVIVAL] Fitting CoxPH trade duration model...")
    n = len(df)
    np.random.seed(42)
    vol_z     = (df["volatility_14"] - df["volatility_14"].mean()) / (df["volatility_14"].std() + 1e-8)
    durations = np.clip(np.random.exponential(scale=20.0, size=n) / (1 + vol_z.values), 1, 200)
    events    = (durations < 25).astype(bool)
    pipeline  = TradeSurvivalPipeline(model_type="coxph", alpha=0.05)
    pipeline.fit(df[FEATURE_COLS], durations, events)
    c_index   = pipeline.evaluate_c_index(df[FEATURE_COLS], durations, events)
    logger.info(f"[SURVIVAL] C-Index: {c_index:.4f} (>0.55 is informative)")
    return c_index


# ─────────────────────────────────────────────────────────────────────────────
# SHAP AUDIT  (Rule D)
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_audit(df: pd.DataFrame) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    logger.info("[SHAP] Fitting GBM proxy for SHAP attribution...")
    X     = df[FEATURE_COLS].iloc[:-50].values
    y     = df[TARGET_COL].shift(-1).iloc[:-50].fillna(0).values
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    diag  = SHAPDiagnosticsAnalyzer(model=model, feature_names=FEATURE_COLS, explainer_type="tree")
    shap_vals     = diag.compute_shap_values(df[FEATURE_COLS].iloc[-50:])
    passed, report = diag.validate_stationarity(
        shap_values=shap_vals,
        stationary_features=FEATURE_COLS,
        max_non_stationary_mass=0.25,
    )
    logger.info(
        f"[SHAP] Audit {'PASSED' if passed else 'FAILED'} | "
        f"Stationary mass: {report['stationary_mass_ratio']*100:.1f}% | "
        f"Top features: {report['top_features']}"
    )
    return {"passed": passed, **report}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = {}

    logger.info("=" * 70)
    logger.info("  MT5 LIVE DATA TRAINING PIPELINE  —  EA AI Research Agent")
    logger.info("=" * 70)

    client    = MT5Client()
    connected = client.connect()
    logger.info(f"MT5 Connected: {connected}")
    if not connected:
        logger.error("MT5 terminal not reachable. Aborting.")
        sys.exit(1)

    acc = client.get_account_info()
    logger.info(
        f"Account #{acc.get('login')} | "
        f"{acc.get('currency','USD')} {acc.get('balance',0):,.2f} balance | "
        f"Equity {acc.get('equity',0):,.2f} | Leverage 1:{acc.get('leverage',0)}"
    )

    # ── Pull & engineer features for all symbols ───────────────────────────
    all_dfs = {}
    for sym in SYMBOLS:
        logger.info(f"[DATA] Fetching {COUNT} {TIMEFRAME} bars for {sym}...")
        raw  = client.get_rates(symbol=sym, timeframe=TIMEFRAME, count=COUNT)
        feat = engineer_features(raw)
        all_dfs[sym] = feat
        logger.info(f"[DATA] {sym}: {len(feat)} bars | "
                    f"{raw['time'].iloc[0]} → {raw['time'].iloc[-1]}")
        results[f"{sym}_bars"] = len(feat)

    df_primary = all_dfs["EURUSD"]

    # ── Survival Analysis ──────────────────────────────────────────────────
    c_index = run_survival_analysis(df_primary)
    results["survival_c_index"] = round(c_index, 4)

    # ── SHAP Audit ─────────────────────────────────────────────────────────
    shap_report = run_shap_audit(df_primary)
    results["shap_passed"]          = shap_report["passed"]
    results["shap_stationary_mass"] = round(shap_report["stationary_mass_ratio"], 4)
    results["shap_top_features"]    = shap_report["top_features"]

    # ── Train Temporal Transformer — EURUSD H1 ─────────────────────────────
    logger.info("[DL] Training TemporalTransformer on real EURUSD H1 data...")
    dm = TimeSeriesDataModule(
        df=df_primary,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        forecast_horizon=HORIZON,
        batch_size=BATCH_SIZE,
        train_ratio=0.70,
        val_ratio=0.15,
        num_workers=4,
        use_robust_scaler=True,
    )
    model = TemporalTransformerForecaster(
        input_dim=len(FEATURE_COLS),
        output_dim=HORIZON,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        learning_rate=LR,
        dropout=0.15,
        weight_decay=1e-4,
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[DL] Params: {params:,} | d_model={D_MODEL} nhead={NHEAD} layers={NUM_LAYERS}")

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    trainer = run_training_pipeline(
        model=model,
        datamodule=dm,
        max_epochs=MAX_EPOCHS,
        checkpoint_dir=str(CHECKPOINT_ROOT / "eurusd_h1"),
        monitor_metric="val_loss",
        patience=PATIENCE,
        gradient_clip_val=1.0,
        accelerator="auto",
    )
    best_ckpt  = trainer.checkpoint_callback.best_model_path
    best_score = float(trainer.checkpoint_callback.best_model_score or 0)
    results["dl_best_val_loss"] = round(best_score, 5)
    results["dl_best_epoch"]    = trainer.current_epoch
    results["dl_checkpoint"]    = best_ckpt
    results["dl_param_count"]   = params
    logger.info(f"[DL] Transformer best val_loss: {best_score:.5f} @ epoch {trainer.current_epoch}")

    # ── Train PatchTST — GBPUSD H1 ────────────────────────────────────────
    logger.info("[PatchTST] Training PatchTST on GBPUSD H1...")
    dm_gbp = TimeSeriesDataModule(
        df=all_dfs["GBPUSD"],
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        forecast_horizon=HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=4,
        use_robust_scaler=True,
    )
    patchtst = PatchTSTLightning(
        seq_len=SEQ_LEN,
        patch_len=12,
        stride=6,
        input_dim=len(FEATURE_COLS),
        output_dim=HORIZON,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        learning_rate=LR,
        dropout=0.15,
    )
    trainer_gbp = run_training_pipeline(
        model=patchtst,
        datamodule=dm_gbp,
        max_epochs=MAX_EPOCHS,
        checkpoint_dir=str(CHECKPOINT_ROOT / "gbpusd_h1_patchtst"),
        monitor_metric="val_loss",
        patience=PATIENCE,
        gradient_clip_val=1.0,
        accelerator="auto",
    )
    gbp_best = float(trainer_gbp.checkpoint_callback.best_model_score or 0)
    results["patchtst_gbpusd_val_loss"] = round(gbp_best, 5)
    logger.info(f"[PatchTST] GBPUSD best val_loss: {gbp_best:.5f}")

    # ── Disconnect & Save ─────────────────────────────────────────────────
    client.disconnect()
    results["timestamp"]        = datetime.now().isoformat()
    results["mt5_connected"]    = connected
    results["account_balance"]  = acc.get("balance", 0)
    results["account_currency"] = acc.get("currency", "USD")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LIVE TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Account        : #{acc.get('login')} | "
          f"{acc.get('currency','USD')} {acc.get('balance',0):,.2f}")
    for sym in SYMBOLS:
        print(f"  {sym} Bars     : {results.get(f'{sym}_bars', 0):,}")
    print(f"  Survival C-Idx : {results['survival_c_index']:.4f}")
    print(f"  SHAP Audit     : {'PASSED' if results['shap_passed'] else 'FAILED'} "
          f"({results['shap_stationary_mass']*100:.1f}% stationary mass)")
    print(f"  Top Features   : {results['shap_top_features']}")
    print(f"  Transformer    : val_loss={results['dl_best_val_loss']:.5f} "
          f"@ epoch {results['dl_best_epoch']}")
    print(f"  PatchTST       : val_loss={results['patchtst_gbpusd_val_loss']:.5f}")
    print(f"  Best ckpt      : {Path(results['dl_checkpoint']).name}")
    print(f"  Results saved  : {RESULTS_PATH}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
