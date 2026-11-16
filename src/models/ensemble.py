"""
Soft-Voting Ensemble Classifier.
Combines Random Forest, XGBoost, and Multilayer Perceptron through soft probability voting.
Author: Archee Jaiswal
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler


class DrowsinessEnsemble(BaseEstimator, ClassifierMixin):
    """
    Multi-Class Drowsiness Ensemble combining RF, XGBoost, and MLP via soft voting.
    Classes: 0 (Alert), 1 (Low Vigilant), 2 (Drowsy).
    """

    def __init__(
        self,
        rf_model: Any = None,
        xgb_model: Any = None,
        mlp_model: Any = None,
        weights: Optional[List[float]] = None,
        scale_inputs: bool = True,
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.mlp_model = mlp_model
        self.weights = weights if weights is not None else [1.0, 1.2, 0.9]
        self.scale_inputs = scale_inputs

        self.scaler = StandardScaler() if scale_inputs else None
        self.ensemble_: Optional[VotingClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DrowsinessEnsemble":
        """Fits the underlying estimators and builds the soft-voting ensemble."""
        if self.scale_inputs and self.scaler is not None:
            X_proc = self.scaler.fit_transform(X)
        else:
            X_proc = np.asarray(X)

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        estimators = [
            ("rf", self.rf_model),
            ("xgb", self.xgb_model),
            ("mlp", self.mlp_model),
        ]

        self.ensemble_ = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=self.weights,
            n_jobs=None,
        )
        self.ensemble_.fit(X_proc, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (0, 1, or 2)."""
        if self.ensemble_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_proc = self.scaler.transform(X) if self.scale_inputs else np.asarray(X)
        return self.ensemble_.predict(X_proc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class probabilities."""
        if self.ensemble_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_proc = self.scaler.transform(X) if self.scale_inputs else np.asarray(X)
        return self.ensemble_.predict_proba(X_proc)

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[int, np.ndarray, str]:
        """
        Convenience method for real-time inference on a single 1D feature array.
        Returns: (predicted_class_id, probabilities_array, class_name)
        """
        from src.utils.config import CLASSES

        x_2d = np.array(feature_vector).reshape(1, -1)
        probs = self.predict_proba(x_2d)[0]
        pred_class = int(np.argmax(probs))
        class_name = CLASSES.get(pred_class, "Unknown")
        return pred_class, probs, class_name

