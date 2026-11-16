"""
Validation Protocols: Seen-Subject and Unseen-Subject Cross Validation.
Compares individual baselines (RF, XGB, MLP) against the Soft-Voting Ensemble.
Author: Archee Jaiswal
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_metrics, compute_video_level_accuracy, plot_confusion_matrix
from src.models.classifiers import get_default_models
from src.models.ensemble import DrowsinessEnsemble
from src.utils.config import ALL_FEATURE_NAMES, RESULTS_DIR


class ValidationEvaluator:
    """
    Executes Seen-Subject (participant-dependent) and
    Unseen-Subject (participant-independent / GroupKFold) validation protocols.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_seen_subject_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        save_plots: bool = True,
    ) -> pd.DataFrame:
        """
        Protocol 1: Seen-Subject (Participant-Dependent) Validation.
        Uses StratifiedKFold cross-validation across all segments.
        """
        print(f"\n[Evaluation] Starting Seen-Subject {n_splits}-Fold Cross Validation...")
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
        X = df[feature_cols].values
        y = df["label"].values.astype(int)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results = []
        oof_y_true = []
        oof_y_pred_ens = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit individual models
            models = get_default_models(random_state=self.random_state)
            fitted_models = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_i = model.predict(X_val)
                m_metrics = compute_metrics(y_val, y_pred_i)
                m_metrics["model"] = name
                m_metrics["fold"] = fold
                m_metrics["protocol"] = "Seen-Subject"
                results.append(m_metrics)
                fitted_models[name] = model

            # Fit Ensemble
            ens = DrowsinessEnsemble(
                rf_model=fitted_models["Random Forest"],
                xgb_model=fitted_models["XGBoost"],
                mlp_model=fitted_models["MLP"],
                scale_inputs=True,
            )
            ens.fit(X_train, y_train)
            y_pred_ens = ens.predict(X_val)
            ens_metrics = compute_metrics(y_val, y_pred_ens)
            ens_metrics["model"] = "Soft-Voting Ensemble"
            ens_metrics["fold"] = fold
            ens_metrics["protocol"] = "Seen-Subject"
            results.append(ens_metrics)

            oof_y_true.extend(y_val.tolist())
            oof_y_pred_ens.extend(y_pred_ens.tolist())

        res_df = pd.DataFrame(results)
        summary = res_df.groupby("model")[["accuracy", "f1_macro", "precision_macro", "recall_macro"]].mean().reset_index()
        summary.to_csv(RESULTS_DIR / "seen_subject_summary.csv", index=False)

        if save_plots and oof_y_true:
            plot_confusion_matrix(
                y_true=np.array(oof_y_true),
                y_pred=np.array(oof_y_pred_ens),
                title="Seen-Subject CV Confusion Matrix (Ensemble)",
                save_filename="confusion_matrix_seen.png",
            )

        return res_df

    def run_unseen_subject_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        save_plots: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Protocol 2: Unseen-Subject (Participant-Independent) Validation.
        Uses GroupKFold grouped by subject_id. No subject in validation is ever in training.
        Also evaluates Video-Level Accuracy.
        """
        print(f"\n[Evaluation] Starting Unseen-Subject {n_splits}-Fold Cross Validation...")
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
        X = df[feature_cols].values
        y = df["label"].values.astype(int)
        groups = df["subject_id"].values

        n_splits = min(n_splits, df["subject_id"].nunique())
        gkf = GroupKFold(n_splits=n_splits)
        segment_results = []
        video_results = []
        oof_y_true = []
        oof_y_pred_ens = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            models = get_default_models(random_state=self.random_state)
            fitted_models = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_i = model.predict(X_val)
                y_prob_i = model.predict_proba(X_val)

                m_metrics = compute_metrics(y_val, y_pred_i)
                m_metrics["model"] = name
                m_metrics["fold"] = fold
                m_metrics["protocol"] = "Unseen-Subject"
                segment_results.append(m_metrics)
                fitted_models[name] = model

                # Video-level
                if "video_id" in df_val.columns:
                    vid_metrics = compute_video_level_accuracy(df_val, y_prob_i)
                    vid_metrics["model"] = name
                    vid_metrics["fold"] = fold
                    vid_metrics["level"] = "Video-Level"
                    video_results.append(vid_metrics)

            # Fit Ensemble
            ens = DrowsinessEnsemble(
                rf_model=fitted_models["Random Forest"],
                xgb_model=fitted_models["XGBoost"],
                mlp_model=fitted_models["MLP"],
                scale_inputs=True,
            )
            ens.fit(X_train, y_train)
            y_pred_ens = ens.predict(X_val)
            y_prob_ens = ens.predict_proba(X_val)

            ens_metrics = compute_metrics(y_val, y_pred_ens)
            ens_metrics["model"] = "Soft-Voting Ensemble"
            ens_metrics["fold"] = fold
            ens_metrics["protocol"] = "Unseen-Subject"
            segment_results.append(ens_metrics)

            oof_y_true.extend(y_val.tolist())
            oof_y_pred_ens.extend(y_pred_ens.tolist())

            if "video_id" in df_val.columns:
                vid_metrics = compute_video_level_accuracy(df_val, y_prob_ens)
                vid_metrics["model"] = "Soft-Voting Ensemble"
                vid_metrics["fold"] = fold
                vid_metrics["level"] = "Video-Level"
                video_results.append(vid_metrics)

        seg_df = pd.DataFrame(segment_results)
        vid_df = pd.DataFrame(video_results) if video_results else pd.DataFrame()

        summary_seg = seg_df.groupby("model")[["accuracy", "f1_macro", "precision_macro", "recall_macro"]].mean().reset_index()
        summary_seg.to_csv(RESULTS_DIR / "unseen_subject_segment_summary.csv", index=False)

        if not vid_df.empty:
            summary_vid = vid_df.groupby("model")[["accuracy", "f1_macro"]].mean().reset_index()
            summary_vid.to_csv(RESULTS_DIR / "unseen_subject_video_summary.csv", index=False)

        if save_plots and oof_y_true:
            plot_confusion_matrix(
                y_true=np.array(oof_y_true),
                y_pred=np.array(oof_y_pred_ens),
                title="Unseen-Subject CV Confusion Matrix (Ensemble)",
                save_filename="confusion_matrix_unseen.png",
            )

        return seg_df, vid_df

