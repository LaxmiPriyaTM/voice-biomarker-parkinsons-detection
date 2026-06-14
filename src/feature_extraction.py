"""
feature_extraction.py
Extracts 13 MFCC coefficients from a .wav audio file using librosa.
"""

import librosa
import numpy as np


def extract_mfcc_features(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Load a WAV file and extract mean MFCC coefficients.

    Parameters
    ----------
    audio_path : str
        Path to the .wav file.
    n_mfcc : int
        Number of MFCC coefficients to extract (default: 13).

    Returns
    -------
    np.ndarray
        1-D array of shape (n_mfcc,) containing mean MFCCs.
    """
    try:
        # Load audio, resample to 22050 Hz, keep mono
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # Compute MFCCs (shape: n_mfcc × frames)
        mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Average over time axis → one value per coefficient
        mfcc_means = np.mean(mfcc_matrix, axis=1)

        return mfcc_means

    except Exception as exc:
        print(f"[ERROR] MFCC extraction failed for {audio_path}: {exc}")
        return np.zeros(n_mfcc)


def get_mfcc_feature_names(n_mfcc: int = 13) -> list:
    """Return column names for MFCC features."""
    return [f"MFCC_{i + 1}" for i in range(n_mfcc)]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <path/to/audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    features = extract_mfcc_features(path)
    names = get_mfcc_feature_names()

    print("MFCC Features:")
    for name, val in zip(names, features):
        print(f"  {name}: {val:.4f}")
