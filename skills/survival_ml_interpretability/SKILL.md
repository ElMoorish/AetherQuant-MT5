---
name: survival-ml-interpretability
description: >-
  scikit-survival (v0.28+) trade duration & SL hazard modeling, 
  SHAP feature attribution diagnostics, and automated price-leakage / non-stationarity rejection filters.
---

# Survival Analysis & SHAP Model Interpretability Skill

This skill defines the quantitative standards for modeling trade lifetime risk using survival analysis and enforcing rigorous feature transparency via SHAP attributions.

---

## 1. Survival Analysis in Algorithmic Trading

Traditional classification models treat trade outcomes as static binary labels (Win/Loss), ignoring the time dimension. Survival analysis models the **time-to-event** $T$, where the event can be:
1. **Stop-Loss Breach**: Modeling hazard rate $\lambda(t)$ of hitting SL at time $t$.
2. **Trade Duration / Stale Exit**: Modeling probability of trade survival $S(t) = P(T > t)$ before reaching profitability.

### Supported Estimators (`scripts/survival_pipeline.py`)
- **`CoxPHSurvivalAnalysis`**: Semi-parametric proportional hazards model estimating feature log-hazard ratios:
  $$\lambda(t | x) = \lambda_0(t) \exp(\beta^T x)$$
- **`RandomSurvivalForest` & `GradientBoostingSurvivalAnalysis`**: Non-linear ensemble estimators modeling complex interactions and non-proportional hazards.
- **Evaluation Metric**: Harrell's Concordance Index ($C\text{-index} \in [0.5, 1.0]$) evaluated on out-of-time test folds.

---

## 2. SHAP Diagnostics & Stationarity Guardrail (Rule D Compliance)

- **Mandatory Feature Attribution**: Every ML/DL model must produce SHAP summary values before live deployment.
- **Automated Price-Leakage Rejection**: If non-stationary features (e.g. raw price level, moving average price without normalization) represent $> 20\%$ of total attribution mass, the pipeline automatically **rejects** the model:
  $$\text{Stationarity Ratio} = \frac{\sum_{i \in \text{Stationary}} |\phi_i|}{\sum_{j \in \text{All}} |\phi_j|} \ge 0.80$$

---

## 3. Standard Usage Workflow

```python
from skills.survival_ml_interpretability.scripts.survival_pipeline import TradeSurvivalPipeline
from skills.survival_ml_interpretability.scripts.shap_diagnostics import SHAPDiagnosticsAnalyzer

# 1. Fit Survival Hazard Model
survival_pipe = TradeSurvivalPipeline(model_type="coxph")
survival_pipe.fit(X_train, durations_train, events_train)
c_index = survival_pipe.evaluate(X_test, durations_test, events_test)
survival_curves = survival_pipe.predict_survival_function(X_test)

# 2. Run SHAP Interpretability & Stationarity Audit
analyzer = SHAPDiagnosticsAnalyzer(model=ml_model, feature_names=feature_cols)
shap_vals = analyzer.compute_shap_values(X_sample)
is_valid, report = analyzer.validate_stationarity(shap_vals, stationary_features=["log_return", "volatility", "rsi_norm"])
```

---

## 4. Module Directory Structure

```text
skills/survival_ml_interpretability/
├── SKILL.md
├── scripts/
│   ├── __init__.py
│   ├── survival_pipeline.py  # scikit-survival CoxPH & RSF estimators
│   └── shap_diagnostics.py   # SHAP attribution & stationarity validation
└── tests/
    ├── __init__.py
    └── test_survival_shap.py # Pytest test suite for survival and SHAP
```
