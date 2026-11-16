"""
Feature Group Ablation Study.
Assesses individual and combined contributions of Eye, Mouth, Head Pose, and Eyebrow features.
Author: Archee Jaiswal
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import compute_metrics
from src.models.classifiers import get_default_models
from src.models.ensemble import DrowsinessEnsemble
from src.utils.config import FEATURE_GROUPS, RESULTS_DIR


class FeatureAblationStudy:
    """
    Evaluates classification performance across different feature group subsets
    under participant-independent (unseen-subject) validation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def define_configurations(self) -> Dict[str, List[str]]:
        """Defines the feature subsets to evaluate."""
        eye = FEATURE_GROUPS["eye"]
        mouth = FEATURE_GROUPS["mouth"]
        pose = FEATURE_GROUPS["head_pose"]
        brow = FEATURE_GROUPS["eyebrow"]

        all_features = eye + mouth + pose + brow

        configs = {
            "Full Hybrid (All)": all_features,
            "Eye Only": eye,
            "Mouth Only": mouth,
            "Head Pose Only": pose,
            "Eyebrow Only": brow,
            "Eye + Mouth (Classic)": eye + mouth,
            "No Head Pose": eye + mouth + brow,
            "No Eyebrow": eye + mouth + pose,
        }
        return configs

    def run_study(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Runs GroupKFold evaluation for each feature configuration.
        """
        print(f"\n[Ablation Study] Evaluating feature groups using {n_splits}-fold GroupKFold...")
        configs = self.define_configurations()
        y = df["label"].values.astype(int)
        groups = df["subject_id"].values
        n_splits = min(n_splits, df["subject_id"].nunique())
        gkf = GroupKFold(n_splits=n_splits)

        records = []

        for config_name, feature_list in configs.items():
            valid_cols = [c for c in feature_list if c in df.columns]
            if not valid_cols:
                continue

            X_subset = df[valid_cols].values
            acc_list = []
            f1_list = []

            for train_idx, val_idx in gkf.split(X_subset, y, groups):
                X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                models = get_default_models(random_state=self.random_state)
                fitted = {}
                for name, model in models.items():
                    model.fit(X_tr, y_tr)
                    fitted[name] = model

                ens = DrowsinessEnsemble(
                    rf_model=fitted["Random Forest"],
                    xgb_model=fitted["XGBoost"],
                    mlp_model=fitted["MLP"],
                    scale_inputs=True,
                )
                ens.fit(X_tr, y_tr)

                y_pred = ens.predict(X_val)
                metrics = compute_metrics(y_val, y_pred)
                acc_list.append(metrics["accuracy"])
                f1_list.append(metrics["f1_macro"])

            records.append({
                "Configuration": config_name,
                "Num Features": len(valid_cols),
                "Mean Accuracy": float(np.mean(acc_list)),
                "Std Accuracy": float(np.std(acc_list)),
                "Mean F1 (Macro)": float(np.mean(f1_list)),
                "Std F1 (Macro)": float(np.std(f1_list)),
            })
            print(f"  Configuration: {config_name:24s} | Accuracy: {np.mean(acc_list)*100:5.2f}% | F1: {np.mean(f1_list):.4f}")

        results_df = pd.DataFrame(records).sort_values(by="Mean Accuracy", ascending=False).reset_index(drop=True)
        results_df.to_csv(RESULTS_DIR / "ablation_study_results.csv", index=False)

        if save_plot:
            self._plot_results(results_df)

        return results_df

    def _plot_results(self, results_df: pd.DataFrame):
        """Generates a publication-ready comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("mako", len(results_df))

        # Sort so highest accuracy is at the top
        sorted_df = results_df.sort_values(by="Mean Accuracy", ascending=True).reset_index(drop=True)

        means = sorted_df["Mean Accuracy"] * 100
        stds = sorted_df["Std Accuracy"] * 100

        bars = ax.barh(
            sorted_df["Configuration"],
            means,
            xerr=stds,
            capsize=4,
            color=palette,
            edgecolor="black",
            alpha=0.88,
        )

        ax.set_xlabel("Unseen-Subject Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title("Feature Group Ablation Study (Ensemble Model)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            err = stds[i] if i < len(stds) else 0.0
            # Place text cleanly past error bar with a crisp background badge
            text_x = min(width + err + 2.0, 95.0)
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.9, linewidth=0.5),
            )

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "ablation_study.png", dpi=300)
        plt.close(fig)

