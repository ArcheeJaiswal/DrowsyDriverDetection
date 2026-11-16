"""
Evaluation Metrics Computation.
Computes segment-level and video-level metrics (accuracy, precision, recall, F1, confusion matrix).
Author: Archee Jaiswal
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import CLASS_NAMES, RESULTS_DIR


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, float]:
    """
    Computes standard multi-class classification metrics:
    accuracy, precision (macro & weighted), recall (macro & weighted), F1 (macro & weighted).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_video_level_accuracy(
    df_segments: pd.DataFrame,
    y_pred_proba: np.ndarray,
    video_id_col: str = "video_id",
    label_col: str = "label",
) -> Dict[str, float]:
    """
    Computes video-level accuracy by aggregating segment predictions for each video.
    Aggregation method: soft probability averaging across all clips of that video.
    """
    df_temp = df_segments.copy().reset_index(drop=True)
    n_classes = y_pred_proba.shape[1]

    for c in range(n_classes):
        df_temp[f"prob_{c}"] = y_pred_proba[:, c]

    prob_cols = [f"prob_{c}" for c in range(n_classes)]
    grouped = df_temp.groupby(video_id_col)

    video_true = []
    video_pred = []

    for vid, group in grouped:
        true_label = int(group[label_col].iloc[0])
        mean_probs = group[prob_cols].mean().values
        pred_label = int(np.argmax(mean_probs))

        video_true.append(true_label)
        video_pred.append(pred_label)

    return compute_metrics(video_true, video_pred)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_filename: Optional[str] = "confusion_matrix.png",
    class_names: List[str] = CLASS_NAMES,
) -> plt.Figure:
    """
    Plots and saves a normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_filename:
        save_path = RESULTS_DIR / save_filename
        fig.savefig(save_path, dpi=300)

    return fig

