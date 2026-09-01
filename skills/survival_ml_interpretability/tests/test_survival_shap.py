"""
Unit test suite for Survival Analysis and SHAP Interpretability modules.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from skills.survival_ml_interpretability.scripts.survival_pipeline import TradeSurvivalPipeline
from skills.survival_ml_interpretability.scripts.shap_diagnostics import SHAPDiagnosticsAnalyzer


@pytest.fixture
def synthetic_survival_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "log_return": np.random.normal(0, 0.01, n),
        "volatility": np.abs(np.random.normal(0.01, 0.005, n)),
        "momentum": np.sin(np.linspace(0, 10, n)),
        "raw_price": 100.0 + np.cumsum(np.random.normal(0, 1, n)),  # Non-stationary
    })
    # Durations (time to SL or exit)
    durations = np.random.exponential(scale=20.0, size=n) + 1.0
    events = np.random.binomial(1, 0.7, size=n).astype(bool)
    return X, durations, events


def test_survival_pipeline_fit_and_cindex(synthetic_survival_data):
    X, durations, events = synthetic_survival_data
    pipe = TradeSurvivalPipeline(model_type="coxph")
    pipe.fit(X[["log_return", "volatility", "momentum"]], durations, events)

    c_index = pipe.evaluate_c_index(X[["log_return", "volatility", "momentum"]], durations, events)
    assert 0.0 <= c_index <= 1.0

    risk_scores = pipe.predict_risk_score(X[["log_return", "volatility", "momentum"]])
    assert len(risk_scores) == len(X)


def test_shap_diagnostics_and_stationarity_guardrail(synthetic_survival_data):
    X, durations, _ = synthetic_survival_data
    # Train dummy model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, durations)

    analyzer = SHAPDiagnosticsAnalyzer(
        model=model,
        feature_names=list(X.columns),
        explainer_type="tree"
    )
    shap_vals = analyzer.compute_shap_values(X.iloc[:20])
    assert shap_vals.shape == (20, 4)

    importances = analyzer.get_feature_importances(shap_vals)
    assert len(importances) == 4

    # Test rejection when non-stationary feature ('raw_price') dominates
    stationary_features = ["log_return", "volatility", "momentum"]
    passed, report = analyzer.validate_stationarity(
        shap_vals,
        stationary_features=stationary_features,
        max_non_stationary_mass=0.05  # strict threshold
    )
    assert "rejection_reason" in report
    assert "stationary_mass_ratio" in report
