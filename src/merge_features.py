"""
merge_features.py
Walks two directories (healthy/ and parkinson/ inside data/raw/),
extracts MFCC + clinical features from every .wav file, and saves
a combined CSV to data/features.csv.

Expected folder layout
----------------------
data/
  raw/
    healthy/      <- .wav files labelled 0
    parkinson/    <- .wav files labelled 1

Output
------
data/features.csv  columns: MFCC_1 … MFCC_13, Jitter, Shimmer, HNR, Label
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_extraction import extract_mfcc_features, get_mfcc_feature_names
from clinical_features import extract_clinical_features, get_clinical_feature_names

# ─── Paths ────────────────────────────────────────────────────────────────── #
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
OUT_CSV  = ROOT_DIR / "data" / "features.csv"

LABEL_MAP = {
    "healthy":   0,
    "parkinson": 1,
}


def extract_all_features(audio_path: str) -> np.ndarray:
    """
    Combine MFCC (13) + clinical (3) into a single feature vector of length 16.
    """
    mfcc   = extract_mfcc_features(audio_path)       # shape (13,)
    clin   = extract_clinical_features(audio_path)   # dict
    clin_v = np.array([clin["Jitter"], clin["Shimmer"], clin["HNR"]])
    return np.concatenate([mfcc, clin_v])


def build_dataset() -> pd.DataFrame:
    """
    Iterate over healthy/ and parkinson/ subdirectories, extract features,
    and return a labelled DataFrame.
    """
    col_names = get_mfcc_feature_names() + get_clinical_feature_names()
    rows = []

    for class_name, label in LABEL_MAP.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"[WARN] Directory not found, skipping: {class_dir}")
            continue

        wav_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.WAV"))
        print(f"[INFO] {class_name}: found {len(wav_files)} WAV files")

        for wav_path in wav_files:
            try:
                features = extract_all_features(str(wav_path))
                row      = dict(zip(col_names, features))
                row["Label"] = label
                rows.append(row)
                print(f"  ✓  {wav_path.name}")
            except Exception as exc:
                print(f"  ✗  {wav_path.name}: {exc}")

    if not rows:
        raise RuntimeError(
            "No features extracted.  Make sure data/raw/healthy/ and "
            "data/raw/parkinson/ contain .wav files."
        )

    df = pd.DataFrame(rows, columns=col_names + ["Label"])
    return df


def generate_synthetic_dataset(n_healthy: int = 100, n_parkinson: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic dataset for demonstration / testing when real audio
    data is not available.  Feature distributions are inspired by published
    Parkinson's voice studies (e.g., Little et al., 2008).
    """
    rng = np.random.default_rng(42)
    col_names = get_mfcc_feature_names() + get_clinical_feature_names()
    rows = []

    def make_row(label: int) -> dict:
        # MFCCs — healthy slightly higher mean energy
        if label == 0:  # healthy
            mfcc   = rng.normal(loc=[-200, 100, -10, 10, -5, 8, -3, 5, -2, 4, -1, 2, -1],
                                scale=[30]*13)
            jitter  = rng.uniform(0.001, 0.004)   # < 1% healthy range
            shimmer = rng.uniform(0.02,  0.06)    # low shimmer
            hnr     = rng.uniform(18,    25)       # high HNR → clean voice
        else:           # parkinson
            mfcc   = rng.normal(loc=[-220, 80, -15, 5, -8, 4, -5, 2, -4, 1, -2, 0, -2],
                                scale=[35]*13)
            jitter  = rng.uniform(0.004, 0.020)   # elevated jitter
            shimmer = rng.uniform(0.06,  0.20)    # elevated shimmer
            hnr     = rng.uniform(5,     18)       # lower HNR → noisy voice

        row = dict(zip(col_names[:13], mfcc))
        row["Jitter"]  = jitter
        row["Shimmer"] = shimmer
        row["HNR"]     = hnr
        row["Label"]   = label
        return row

    for _ in range(n_healthy):
        rows.append(make_row(0))
    for _ in range(n_parkinson):
        rows.append(make_row(1))

    df = pd.DataFrame(rows, columns=col_names + ["Label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─── Entry-point ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Check whether real audio data exists
    has_real_data = (
        (RAW_DIR / "healthy").exists() or
        (RAW_DIR / "parkinson").exists()
    )

    if has_real_data:
        print("[INFO] Real audio data found — extracting features …")
        df = build_dataset()
    else:
        print("[INFO] No real audio data found in data/raw/.")
        print("[INFO] Generating SYNTHETIC dataset for demonstration …")
        df = generate_synthetic_dataset(n_healthy=150, n_parkinson=150)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n[INFO] Dataset saved → {OUT_CSV}")
    print(f"[INFO] Shape: {df.shape}")
    print(df.head())
    print("\nLabel distribution:")
    print(df["Label"].value_counts().rename({0: "Healthy", 1: "Parkinson"}))
