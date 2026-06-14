"""
compare_models.py
Loads data/features.csv, trains Random Forest and SVM, prints a detailed
comparison table, and returns the best model by accuracy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data" / "features.csv"


# ─── Helpers ──────────────────────────────────────────────────────────────── #

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["Label"]).values
    y  = df["Label"].values
    feature_names = list(df.drop(columns=["Label"]).columns)
    return X, y, feature_names


def build_pipelines() -> dict:
    """Return dict of {model_name: sklearn Pipeline}."""
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])

    return {"Random Forest": rf, "SVM (RBF)": svm}


def evaluate_model_cv(pipeline, X, y, cv=5):
    """
    Perform Stratified K-Fold cross-validation and return mean ± std
    of Accuracy, Precision, Recall, F1.
    """
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_obj  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = cross_validate(
        pipeline, X, y,
        cv=cv_obj,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    metrics = {}
    for metric in scoring:
        key = f"test_{metric}"
        metrics[metric] = {
            "mean": results[key].mean(),
            "std":  results[key].std(),
        }
    return metrics


def print_comparison_table(comparison: dict):
    """Pretty-print comparison results."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    col_w   = 22

    header = f"{'Metric':<15}" + "".join(f"{name:^{col_w}}" for name in comparison)
    sep    = "─" * len(header)

    print("\n" + "=" * len(header))
    print("  MODEL COMPARISON (5-Fold Stratified Cross-Validation)")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in metrics:
        row = f"{m.capitalize():<15}"
        for name in comparison:
            val = comparison[name][m]
            row += f"{val['mean']:.4f} ± {val['std']:.4f}".center(col_w)
        print(row)

    print("=" * len(header))


def compare_models(csv_path: Path = DATA_CSV):
    """
    Full comparison pipeline.  Returns (best_name, best_pipeline, X, y,
    feature_names, all_results).
    """
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found at {csv_path}")
        print("[INFO]  Run:  python src/merge_features.py   first.")
        sys.exit(1)

    X, y, feature_names = load_data(csv_path)
    print(f"[INFO] Dataset loaded: {X.shape[0]} samples, "
          f"{X.shape[1]} features  |  "
          f"Healthy={int((y==0).sum())}  Parkinson={int((y==1).sum())}")

    pipelines   = build_pipelines()
    comparison  = {}

    print("\n[INFO] Running cross-validation … (this may take a moment)")
    for name, pipe in pipelines.items():
        print(f"  ▶  {name}")
        comparison[name] = evaluate_model_cv(pipe, X, y, cv=5)

    print_comparison_table(comparison)

    # ── Pick best model by mean accuracy ────────────────────────────────── #
    best_name = max(comparison, key=lambda n: comparison[n]["accuracy"]["mean"])
    print(f"\n[RESULT] Best model → {best_name}  "
          f"(accuracy = {comparison[best_name]['accuracy']['mean']:.4f})\n")

    # Re-fit best pipeline on full data
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X, y)

    return best_name, best_pipeline, X, y, feature_names, comparison


# ─── Entry-point ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    best_name, best_pipeline, X, y, feature_names, comparison = compare_models()
