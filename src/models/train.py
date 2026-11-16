"""
Model Training and Serialization Pipeline.
Trains individual classifiers (RF, XGBoost, MLP), tunes hyperparameters,
constructs the soft-voting ensemble, and persists models to disk.
Author: Archee Jaiswal
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.classifiers import (
    get_default_models,
    get_hyperparameter_grids,
    tune_classifier,
)
from src.models.ensemble import DrowsinessEnsemble
from src.utils.config import (
    ALL_FEATURE_NAMES,
    SAVED_MODELS_DIR,
)


def train_and_save_pipeline(
    df: pd.DataFrame,
    tune_hyperparameters: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Path = SAVED_MODELS_DIR,
) -> Tuple[DrowsinessEnsemble, Dict[str, float]]:
    """
    Full training pipeline:
    1. Splits dataset into train/test (stratified)
    2. Trains or tunes RF, XGBoost, MLP
    3. Assembles SoftVotingEnsemble
    4. Evaluates test set performance
    5. Saves models and metadata to output_dir
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = get_default_models(random_state=random_state)
    trained_models = {}

    if tune_hyperparameters:
        print("[Model Tuning] Optimizing hyperparameters for individual models...")
        grids = get_hyperparameter_grids()
        for name, model in models.items():
            print(f"  -> Tuning {name}...")
            best_est, best_params, best_score = tune_classifier(
                name, model, grids[name], X_train, y_train, random_state=random_state
            )
            trained_models[name] = best_est
            print(f"     Best {name} CV F1: {best_score:.4f}")
    else:
        print("[Model Training] Fitting baseline classifiers...")
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model

    # Build Soft-Voting Ensemble
    print("[Ensemble Construction] Constructing Soft-Voting Ensemble...")
    # Clean inner MLP if pipeline was used
    rf_est = trained_models["Random Forest"]
    xgb_est = trained_models["XGBoost"]
    mlp_est = trained_models["MLP"]

    ensemble = DrowsinessEnsemble(
        rf_model=rf_est,
        xgb_model=xgb_est,
        mlp_model=mlp_est,
        weights=[1.0, 1.2, 0.9],
        scale_inputs=True,
    )
    ensemble.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = ensemble.predict(X_test)
    test_acc = float(np.mean(y_pred == y_test))

    # Evaluate individual models
    scores = {"Ensemble": test_acc}
    for name, model in trained_models.items():
        pred_i = model.predict(X_test)
        scores[name] = float(np.mean(pred_i == y_test))

    print(f"\n[Training Results]")
    for m_name, acc in scores.items():
        print(f"  {m_name:15s} Test Accuracy: {acc*100:.2f}%")

    # Serialize artifacts
    ensemble_path = output_dir / "drowsiness_ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)

    metadata = {
        "feature_names": feature_cols,
        "test_scores": scores,
        "test_size": test_size,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "random_state": random_state,
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Model Artifacts] Successfully saved models to {output_dir}")
    return ensemble, scores

