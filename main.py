"""
Master CLI and Pipeline Runner for Multi-Level Driver Drowsiness Detection.
Executes data generation, model training, multi-protocol validation, and ablation studies.
Author: Archee Jaiswal
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.utils.config import (
    FEATURES_DATA_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
)
import pandas as pd
from src.evaluation.ablation import FeatureAblationStudy
from src.evaluation.metrics import plot_confusion_matrix
from src.evaluation.validation import ValidationEvaluator
from src.models.train import train_and_save_pipeline
from src.utils.synthetic_data import generate_utarldd_benchmark_dataset


def load_dataset(dataset_choice: str = "auto", do_generate: bool = True) -> pd.DataFrame:
    """
    Loads dataset according to user specification:
    - 'auto': uses real extracted features if available, else benchmark dataset
    - 'real': uses real extracted UTA-RLDD features
    - 'benchmark': uses 60-subject benchmark dataset (generates if missing)
    - 'combined': combines real and benchmark features for large-scale training
    """
    real_dataset_path = FEATURES_DATA_DIR / "utarldd_real_features.csv"
    benchmark_dataset_path = FEATURES_DATA_DIR / "utarldd_benchmark_features.csv"

    if dataset_choice == "real":
        if not real_dataset_path.exists():
            raise FileNotFoundError(f"Real features not found at {real_dataset_path}. Run extract_features.py first.")
        print(f"\n[Step 1/4 - Data Ingestion] Loading Real Extracted Dataset: {real_dataset_path}")
        return pd.read_csv(real_dataset_path)

    elif dataset_choice == "benchmark":
        if not benchmark_dataset_path.exists() or do_generate:
            print("\n[Step 1/4 - Data Ingestion] Generating 60-subject UTA-RLDD Benchmark Dataset...")
            return generate_utarldd_benchmark_dataset(
                num_subjects=60,
                clips_per_class=6,
                random_seed=42,
                output_path=benchmark_dataset_path,
            )
        print(f"\n[Step 1/4 - Data Ingestion] Loading Benchmark Dataset: {benchmark_dataset_path}")
        return pd.read_csv(benchmark_dataset_path)

    elif dataset_choice == "combined":
        if not real_dataset_path.exists():
            print(f"  [Warning] Real features not found at {real_dataset_path}, falling back to benchmark.")
            return load_dataset("benchmark", do_generate)
        if not benchmark_dataset_path.exists():
            generate_utarldd_benchmark_dataset(num_subjects=60, clips_per_class=6, random_seed=42, output_path=benchmark_dataset_path)
        print("\n[Step 1/4 - Data Ingestion] Combining Real and Benchmark Datasets...")
        df_real = pd.read_csv(real_dataset_path)
        df_bench = pd.read_csv(benchmark_dataset_path)
        common_cols = [c for c in df_real.columns if c in df_bench.columns]
        df_combined = pd.concat([df_real[common_cols], df_bench[common_cols]], ignore_index=True)
        return df_combined

    else:  # 'auto'
        if real_dataset_path.exists() and benchmark_dataset_path.exists():
            print(f"\n[Step 1/4 - Data Ingestion] Auto-selecting Combined Real + Benchmark Dataset...")
            return load_dataset("combined", do_generate=False)
        elif real_dataset_path.exists():
            print(f"\n[Step 1/4 - Data Ingestion] Using Real Extracted Dataset: {real_dataset_path}")
            return pd.read_csv(real_dataset_path)
        elif do_generate or not benchmark_dataset_path.exists():
            print("\n[Step 1/4 - Data Ingestion] Preparing Benchmark Dataset...")
            return generate_utarldd_benchmark_dataset(
                num_subjects=60,
                clips_per_class=6,
                random_seed=42,
                output_path=benchmark_dataset_path,
            )
        else:
            print(f"\n[Step 1/4 - Data Ingestion] Loading Dataset from {benchmark_dataset_path}...")
            return pd.read_csv(benchmark_dataset_path)


def run_pipeline(
    do_generate: bool = True,
    do_train: bool = True,
    do_eval: bool = True,
    do_ablate: bool = True,
    tune: bool = False,
    dataset: str = "auto",
):
    print("=" * 70)
    print("   MULTI-LEVEL DRIVER DROWSINESS DETECTION PIPELINE")
    print("   Hybrid Facial Features & Soft-Voting Ensemble Learning")
    print("=" * 70)

    # Step 1: Data Preparation / Ingestion
    df = load_dataset(dataset_choice=dataset, do_generate=do_generate)
    print(f"  Loaded {len(df)} samples across {df['subject_id'].nunique()} subjects.")
    print(f"  Class Distribution: {dict(df['label'].value_counts())}")

    # Step 2: Model Training & Tuning
    if do_train:
        print("\n[Step 2/4 - Model Development & Training] Training Classifiers & Soft-Voting Ensemble...")
        ensemble, scores = train_and_save_pipeline(
            df=df,
            tune_hyperparameters=tune,
            test_size=0.20,
            random_state=42,
        )

    # Step 3: Seen & Unseen Subject Validation Protocols
    if do_eval:
        print("\n[Step 3/4 - Seen & Unseen Cross-Validation Protocols] Running Evaluation Protocols...")
        evaluator = ValidationEvaluator(random_state=42)

        # Seen-Subject CV
        df_seen = evaluator.run_seen_subject_cv(df=df, n_splits=5, save_plots=True)
        print("\n  [Summary: Seen-Subject (Participant-Dependent) CV]")
        print(df_seen.groupby("model")[["accuracy", "f1_macro"]].mean().to_string())

        # Unseen-Subject CV
        df_unseen_seg, df_unseen_vid = evaluator.run_unseen_subject_cv(df=df, n_splits=5, save_plots=True)
        print("\n  [Summary: Unseen-Subject (Participant-Independent) Segment-Level CV]")
        print(df_unseen_seg.groupby("model")[["accuracy", "f1_macro"]].mean().to_string())

        if not df_unseen_vid.empty:
            print("\n  [Summary: Unseen-Subject Video-Level Accuracy]")
            print(df_unseen_vid.groupby("model")[["accuracy", "f1_macro"]].mean().to_string())

    # Step 4: Feature Group Ablation Study
    if do_ablate:
        print("\n[Step 4/4 - Feature Group Ablation Study] Conducting Feature Group Ablation Study...")
        ablation = FeatureAblationStudy(random_state=42)
        ablation_df = ablation.run_study(df=df, n_splits=5, save_plot=True)
        print("\n  [Ablation Ranking (Unseen-Subject Accuracy)]")
        print(ablation_df[["Configuration", "Num Features", "Mean Accuracy", "Mean F1 (Macro)"]].to_string())

    print("\n" + "=" * 70)
    print("   PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print(f"   Artifacts and models saved to: {SAVED_MODELS_DIR}")
    print(f"   Evaluation tables and plots saved to: {RESULTS_DIR}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-Level Driver Drowsiness Detection System")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: generate, train, eval, ablate")
    parser.add_argument("--generate-data", action="store_true", help="Generate realistic benchmark dataset")
    parser.add_argument("--train", action="store_true", help="Train classifiers and ensemble")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search during training")
    parser.add_argument("--evaluate", action="store_true", help="Run Seen & Unseen cross-validation")
    parser.add_argument("--ablate", action="store_true", help="Run feature group ablation study")
    parser.add_argument(
        "--dataset",
        choices=["auto", "real", "benchmark", "combined"],
        default="auto",
        help="Dataset to use for training/eval: auto, real, benchmark, or combined (default: auto)",
    )

    args = parser.parse_args()

    if not any([args.all, args.generate_data, args.train, args.evaluate, args.ablate]):
        # Default run all if no flags provided
        run_pipeline(do_generate=False, do_train=True, do_eval=True, do_ablate=True, tune=False, dataset=args.dataset)
    else:
        run_pipeline(
            do_generate=args.all or args.generate_data,
            do_train=args.all or args.train,
            do_eval=args.all or args.evaluate,
            do_ablate=args.all or args.ablate,
            tune=args.tune,
            dataset=args.dataset,
        )


if __name__ == "__main__":
    main()
