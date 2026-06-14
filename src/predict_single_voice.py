"""
predict_single_voice.py
Loads a trained model and predicts Parkinson's likelihood from a single .wav
file. Also surfaces the top contributing features (Explainable AI).

Usage
-----
    python src/predict_single_voice.py path/to/voice.wav
"""

import sys
import json
import joblib
import numpy as np
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
META_PATH  = ROOT_DIR / "models" / "model_meta.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_extraction import extract_mfcc_features
from clinical_features  import extract_clinical_features


# ─── Feature extraction ────────────────────────────────────────────────────── #

def extract_features_for_prediction(audio_path: str) -> np.ndarray:
    mfcc   = extract_mfcc_features(audio_path)
    clin   = extract_clinical_features(audio_path)
    clin_v = np.array([clin["Jitter"], clin["Shimmer"], clin["HNR"]])
    return np.concatenate([mfcc, clin_v]).reshape(1, -1)


# ─── Risk level mapping ────────────────────────────────────────────────────── #

def probability_to_risk(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Moderate"
    else:
        return "High"


# ─── XAI: top contributing features ────────────────────────────────────────── #

def get_top_features(pipeline, feature_names: list, top_n: int = 3) -> list:
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_n]


# ─── NEW: Clinical Explanation Function ───────────────────────────────────── #

def get_clinical_explanation(top_features, risk_level):
    explanation_map = {
        "Jitter": "increased pitch instability",
        "Shimmer": "variability in loudness",
        "HNR": "reduced voice clarity (more noise)",
    }

    # Extract only feature names (ignore importance values)
    feature_names = [feat for feat, _ in top_features]

    mapped = [explanation_map.get(f, f) for f in feature_names]

    if risk_level == "Low":
        return "Voice characteristics appear stable with no significant abnormalities."

    elif risk_level == "Moderate":
        return f"The voice shows mild irregularities including {', '.join(mapped)}."

    else:
        return f"The voice shows significant abnormalities including {', '.join(mapped)}, which are commonly associated with Parkinson’s disease."


# ─── Main prediction function ──────────────────────────────────────────────── #

def predict(audio_path: str) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run: python src/train_model.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    feature_names = None

    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names")

    if feature_names is None:
        feature_names = [f"MFCC_{i+1}" for i in range(13)] + ["Jitter", "Shimmer", "HNR"]

    # Extract features
    X = extract_features_for_prediction(audio_path)
    raw_features = dict(zip(feature_names, X[0]))

    # Prediction
    label   = int(pipeline.predict(X)[0])
    proba   = pipeline.predict_proba(X)[0]
    pk_prob = float(proba[1])

    prediction = "Parkinson's" if label == 1 else "Healthy"
    risk_level = probability_to_risk(pk_prob)

    # XAI
    top_features = get_top_features(pipeline, feature_names, top_n=3)

    # ─── NEW: Clinical Explanation ─── #
    clinical_text = get_clinical_explanation(top_features, risk_level)

    return {
        "prediction":   prediction,
        "risk_level":   risk_level,
        "probability":  pk_prob,
        "top_features": top_features,
        "raw_features": raw_features,
        "clinical_text": clinical_text,   # ← NEW FIELD
    }


# ─── Pretty printer ───────────────────────────────────────────────────────── #

def print_result(result: dict):
    width = 52
    pk_prob = result["probability"]
    bar_len = int(pk_prob * 40)
    bar     = "█" * bar_len + "░" * (40 - bar_len)

    print("\n" + "═" * width)
    print("   VOICE BIOMARKER ANALYSIS — PARKINSON'S DETECTION")
    print("═" * width)
    print(f"  Prediction  :  {result['prediction']}")
    print(f"  Risk Level  :  {result['risk_level']}")
    print(f"  Probability :  {pk_prob:.1%}  [{bar}]")

    print("─" * width)
    print("  Top 3 Contributing Features (XAI):")
    for i, (feat, imp) in enumerate(result["top_features"], 1):
        imp_bar = "▮" * int(imp * 50)
        print(f"    {i}. {feat:<14}  importance = {imp:.4f}  {imp_bar}")

    # ─── NEW: Clinical Explanation Output ─── #
    print("─" * width)
    print("  Clinical Interpretation:")
    print(f"    {result['clinical_text']}")

    print("═" * width + "\n")


# ─── Entry-point ──────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_single_voice.py <path/to/voice.wav>")
        sys.exit(1)

    audio = sys.argv[1]
    print(f"[INFO] Analysing: {audio}")
    result = predict(audio)
    print_result(result)