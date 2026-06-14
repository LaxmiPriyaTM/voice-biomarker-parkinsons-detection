"""
train_model.py
Orchestrates the full training pipeline:
  1. Generate / load dataset (data/features.csv)
  2. Compare Random Forest vs SVM
  3. Save the best model to models/best_model.pkl
  4. Save feature names and model metadata

Run from the project root:
    python src/train_model.py
"""

import sys
import json
import joblib
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_CSV   = ROOT_DIR / "data" / "features.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH  = MODELS_DIR / "model_meta.json"

# Allow running from src/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ensure_dataset():
    """Create features.csv if it does not exist."""
    if not DATA_CSV.exists():
        print("[INFO] features.csv not found — generating dataset …")
        from merge_features import generate_synthetic_dataset
        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset(n_healthy=150, n_parkinson=150)
        df.to_csv(DATA_CSV, index=False)
        print(f"[INFO] Synthetic dataset saved → {DATA_CSV}")
    else:
        print(f"[INFO] Using existing dataset → {DATA_CSV}")


def train():
    ensure_dataset()

    from compare_models import compare_models

    best_name, best_pipeline, X, y, feature_names, comparison = compare_models(DATA_CSV)

    # ── Save model ───────────────────────────────────────────────────────── #
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    # ── Save metadata (feature names, model name, metrics) ───────────────── #
    meta = {
        "model_name":    best_name,
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "n_samples":     int(len(y)),
        "metrics": {
            name: {
                metric: {
                    "mean": round(float(vals["mean"]), 4),
                    "std":  round(float(vals["std"]),  4),
                }
                for metric, vals in model_metrics.items()
            }
            for name, model_metrics in comparison.items()
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved → {META_PATH}")

    # ── Feature importances (Random Forest only) ─────────────────────────── #
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        ranked      = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\n[INFO] Top-10 Feature Importances:")
        for rank, (feat, imp) in enumerate(ranked[:10], 1):
            bar = "█" * int(imp * 50)
            print(f"  {rank:>2}. {feat:<12}  {imp:.4f}  {bar}")

    print("\n[SUCCESS] Training complete!")
    print(f"          Best model : {best_name}")
    print(f"          Saved to   : {MODEL_PATH}")


# ─── Entry-point ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    train()
