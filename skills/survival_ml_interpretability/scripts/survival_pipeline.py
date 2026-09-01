"""
Survival Analysis Pipeline for Trade Lifetime and Stop-Loss Hazard Modeling.
Leverages scikit-survival (v0.28+) to model dynamic trade exit timing and hazard rates.
"""
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

try:
    import sksurv
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    sksurv = None
    CoxPHSurvivalAnalysis = None
    RandomSurvivalForest = None
    GradientBoostingSurvivalAnalysis = None
    concordance_index_censored = None
    SKSURV_AVAILABLE = False


class TradeSurvivalPipeline:
    """
    Fits and evaluates survival analysis models on trade duration / time-to-stop-loss data.
    """

    def __init__(self, model_type: str = "coxph", alpha: float = 0.1, n_estimators: int = 100):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        if not SKSURV_AVAILABLE:
            self.model = None
            return

        if self.model_type == "coxph":
            self.model = CoxPHSurvivalAnalysis(alpha=self.alpha)
        elif self.model_type == "rsf":
            self.model = RandomSurvivalForest(
                n_estimators=self.n_estimators, min_samples_split=10, min_samples_leaf=5, random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingSurvivalAnalysis(
                n_estimators=self.n_estimators, learning_rate=0.05, random_state=42
            )
        else:
            raise ValueError(f"Unsupported survival model type: {self.model_type}")

    @staticmethod
    def to_survival_target(events: np.ndarray, durations: np.ndarray) -> np.ndarray:
        """
        Formats boolean event indicators and continuous durations into a structured array.
        """
        events_bool = np.asarray(events, dtype=bool)
        durations_float = np.asarray(durations, dtype=float)
        y = np.empty(
            len(events_bool),
            dtype=[("event", bool), ("time", float)]
        )
        y["event"] = events_bool
        y["time"] = durations_float
        return y

    def fit(self, X: Union[pd.DataFrame, np.ndarray], durations: np.ndarray, events: np.ndarray) -> "TradeSurvivalPipeline":
        """
        Fits the survival estimator on feature matrix X, durations, and event indicators.
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_struct = self.to_survival_target(events, durations)

        if SKSURV_AVAILABLE and self.model is not None:
            self.model.fit(X_mat, y_struct)
        else:
            # Fallback linear ridge regression proxy for risk score when sksurv is not compiled
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.alpha)
            self.model.fit(X_mat, -durations)

        return self

    def predict_risk_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts relative risk scores (higher score = higher hazard / shorter survival).
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if SKSURV_AVAILABLE and hasattr(self.model, "predict"):
            return self.model.predict(X_mat)
        elif hasattr(self.model, "predict"):
            return self.model.predict(X_mat)
        return np.zeros(len(X_mat))

    def evaluate_c_index(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        durations: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Computes Harrell's concordance index (C-index) on validation/test data.
        C-index = 0.5 (random guess), 1.0 (perfect ranking).
        """
        risk_scores = self.predict_risk_score(X)
        events_bool = np.asarray(events, dtype=bool)
        durations_float = np.asarray(durations, dtype=float)

        if SKSURV_AVAILABLE and concordance_index_censored is not None:
            c_index, _, _, _, _ = concordance_index_censored(events_bool, durations_float, risk_scores)
            return float(c_index)
        else:
            # Manual simple concordance index fallback
            valid_pairs = 0
            concordant_pairs = 0
            n = len(durations_float)
            for i in range(n):
                for j in range(i + 1, n):
                    if events_bool[i] and durations_float[i] < durations_float[j]:
                        valid_pairs += 1
                        if risk_scores[i] > risk_scores[j]:
                            concordant_pairs += 1
                        elif risk_scores[i] == risk_scores[j]:
                            concordant_pairs += 0.5
            return concordant_pairs / valid_pairs if valid_pairs > 0 else 0.5

    def predict_survival_function(self, X: Union[pd.DataFrame, np.ndarray]) -> List[Any]:
        """
        Predicts step survival functions S(t) = P(T > t) for each sample in X.
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if SKSURV_AVAILABLE and hasattr(self.model, "predict_survival_function"):
            return self.model.predict_survival_function(X_mat)
        return []
