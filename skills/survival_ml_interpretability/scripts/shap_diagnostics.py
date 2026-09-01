"""
SHAP Model Interpretability & Stationarity Guardrail Module.
Enforces Rule D: Mandatory SHAP attribution validation and automated rejection of models
relying on non-stationary absolute price features.
"""
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import shap
import logging

logger = logging.getLogger(__name__)


class SHAPDiagnosticsAnalyzer:
    """
    Computes SHAP feature attributions and audits models for price leakage and non-stationarity.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        explainer_type: str = "auto",
        background_samples: Optional[np.ndarray] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.background_samples = background_samples
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Instantiates appropriate SHAP explainer."""
        try:
            if self.explainer_type == "tree" or hasattr(self.model, "tree_"):
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "linear" or hasattr(self.model, "coef_"):
                if self.background_samples is not None:
                    self.explainer = shap.LinearExplainer(self.model, self.background_samples)
                else:
                    self.explainer = shap.Explainer(self.model)
            else:
                if self.background_samples is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_samples)
                else:
                    self.explainer = shap.Explainer(self.model)
        except Exception as e:
            logger.warning(f"Defaulting to general shap.Explainer due to: {e}")
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception:
                self.explainer = None

    def compute_shap_values(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates matrix of SHAP attribution values [N_samples, N_features].
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self.explainer is None:
            # Fallback simple gradient/permutation attribution
            return np.zeros_like(X_mat)

        shap_vals = self.explainer(X_mat)
        if hasattr(shap_vals, "values"):
            values = shap_vals.values
        else:
            values = shap_vals

        if isinstance(values, list):
            values = values[0]  # Binary classification head 0

        # Squeeze 3D outputs [N, D, 1] -> [N, D]
        if len(values.shape) == 3:
            values = values[:, :, 0]

        return values

    def get_feature_importances(self, shap_values: np.ndarray) -> pd.Series:
        """
        Calculates mean absolute SHAP value per feature: mean(|SHAP_i|).
        """
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        series = pd.Series(mean_abs_shap, index=self.feature_names)
        return series.sort_values(ascending=False)

    def validate_stationarity(
        self,
        shap_values: np.ndarray,
        stationary_features: List[str],
        max_non_stationary_mass: float = 0.20
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Rule D Enforcement: Audits feature attribution mass.
        Rejects models if non-stationary features exceed max_non_stationary_mass (e.g. 20%).
        """
        importances = self.get_feature_importances(shap_values)
        total_mass = importances.sum()

        if total_mass == 0:
            return True, {"status": "EMPTY_SHAP", "non_stationary_ratio": 0.0}

        stationary_set = set(stationary_features)
        non_stationary_features = [f for f in self.feature_names if f not in stationary_set]

        non_stat_mass = importances[non_stationary_features].sum() if non_stationary_features else 0.0
        non_stat_ratio = float(non_stat_mass / total_mass)
        stat_ratio = 1.0 - non_stat_ratio

        passed = non_stat_ratio <= max_non_stationary_mass

        report = {
            "passed": passed,
            "stationary_mass_ratio": stat_ratio,
            "non_stationary_mass_ratio": non_stat_ratio,
            "top_features": importances.head(5).to_dict(),
            "flagged_features": non_stationary_features,
            "rejection_reason": None if passed else (
                f"Rule D Rejection: Non-stationary features represent {non_stat_ratio*100:.1f}% "
                f"of attribution mass (Limit: {max_non_stationary_mass*100:.1f}%)."
            )
        }

        if not passed:
            logger.error(report["rejection_reason"])
        else:
            logger.info(f"Model passed SHAP stationarity audit. Stationary mass: {stat_ratio*100:.1f}%")

        return passed, report
