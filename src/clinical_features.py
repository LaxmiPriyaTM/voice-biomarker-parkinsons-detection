"""
clinical_features.py
Extracts clinical voice biomarkers (Jitter, Shimmer, HNR) using Praat via
the praat-parselmouth library.  These features are well-established
Parkinson's disease indicators in the literature.
"""

import parselmouth
from parselmouth.praat import call
import numpy as np


def extract_clinical_features(audio_path: str) -> dict:
    """
    Extract Jitter, Shimmer, and HNR from a .wav file.

    Parameters
    ----------
    audio_path : str
        Path to the .wav file (sustained vowel recommended).

    Returns
    -------
    dict
        {'Jitter': float, 'Shimmer': float, 'HNR': float}
        Returns zeros on failure.
    """
    defaults = {"Jitter": 0.0, "Shimmer": 0.0, "HNR": 0.0}

    try:
        sound = parselmouth.Sound(audio_path)

        # ------------------------------------------------------------------ #
        # Jitter (local) — cycle-to-cycle F0 variation, key PD biomarker     #
        # ------------------------------------------------------------------ #
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        # ------------------------------------------------------------------ #
        # Shimmer (local, dB) — amplitude variation between cycles           #
        # ------------------------------------------------------------------ #
        shimmer = call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        # ------------------------------------------------------------------ #
        # Harmonics-to-Noise Ratio — voiced/unvoiced energy balance          #
        # ------------------------------------------------------------------ #
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Guard against NaN / Praat returning --undefined--
        jitter  = float(jitter)  if (jitter  is not None and not np.isnan(jitter))  else 0.0
        shimmer = float(shimmer) if (shimmer is not None and not np.isnan(shimmer)) else 0.0
        hnr     = float(hnr)     if (hnr     is not None and not np.isnan(hnr))     else 0.0

        return {"Jitter": jitter, "Shimmer": shimmer, "HNR": hnr}

    except Exception as exc:
        print(f"[ERROR] Clinical feature extraction failed for {audio_path}: {exc}")
        return defaults


def get_clinical_feature_names() -> list:
    """Return ordered list of clinical feature column names."""
    return ["Jitter", "Shimmer", "HNR"]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clinical_features.py <path/to/audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    feats = extract_clinical_features(path)
    print("Clinical Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")
