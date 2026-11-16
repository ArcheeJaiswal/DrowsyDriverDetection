"""
Individual Classifiers and Hyperparameter Optimization.
Implements Random Forest, XGBoost, and Multilayer Perceptron with tuning grids.
Author: Archee Jaiswal
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns baseline un-tuned instances of the three target classifiers.
    """
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.002,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        ))
    ])

    return {
        "Random Forest": rf,
        "XGBoost": xgb_clf,
        "MLP": mlp_pipeline,
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Defines parameter search grids for tuning each individual model.
    """
    rf_grid = {
        "n_estimators": [100, 150, 250, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    xgb_grid = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.08, 0.15],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "gamma": [0, 0.1, 0.2],
    }

    mlp_grid = {
        "mlp__hidden_layer_sizes": [
            (64, 32),
            (128, 64),
            (128, 64, 32),
            (256, 128),
        ],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [0.001, 0.002, 0.005],
        "mlp__activation": ["relu", "tanh"],
    }

    return {
        "Random Forest": rf_grid,
        "XGBoost": xgb_grid,
        "MLP": mlp_grid,
    }


def tune_classifier(
    name: str,
    base_model: Any,
    param_grid: Dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    n_iter: int = 15,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Performs RandomizedSearchCV on a single model and returns:
    (best_estimator, best_params, best_score)
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)

