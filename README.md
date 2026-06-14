# 🧠 VoicePD — Voice Biomarker Analysis for Early Parkinson's Detection

> An end-to-end Machine Learning pipeline that detects early Parkinson's Disease from sustained vowel recordings using spectral and clinical voice biomarkers, with Explainable AI to surface the driving features.

---

## 📌 Project Overview

Parkinson's Disease (PD) affects motor control including the vocal muscles, producing measurable changes in voice characteristics years before other symptoms become clinically apparent. This project automates the extraction of those biomarkers, trains and compares two classification models, and presents predictions through a clean Streamlit web application — all explained via Explainable AI (XAI).

This is a **3rd-year level undergraduate project** demonstrating:
- Audio signal processing (librosa, Praat/parselmouth)
- Binary classification with scikit-learn
- Model comparison and selection
- Explainable AI via feature importance
- Streamlit web application development

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🎙️ Voice Input | Upload `.wav` file of sustained vowel "AAAA" |
| 📊 Feature Extraction | 13 MFCCs + Jitter + Shimmer + HNR = 16 features |
| 🤖 Model Comparison | Random Forest vs SVM with 5-fold cross-validation |
| 🏆 Auto Model Selection | Best model saved automatically by accuracy |
| 🔍 XAI | Top-3 contributing features highlighted |
| 🌐 Web App | Clean Streamlit UI with risk levels and probability |

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── raw/
│   │   ├── healthy/        ← .wav files labelled Healthy (0)
│   │   └── parkinson/      ← .wav files labelled Parkinson (1)
│   └── features.csv        ← generated feature dataset
│
├── models/
│   ├── best_model.pkl      ← trained best model
│   └── model_meta.json     ← metadata (feature names, metrics)
│
├── src/
│   ├── feature_extraction.py     ← MFCC extraction (librosa)
│   ├── clinical_features.py      ← Jitter, Shimmer, HNR (parselmouth)
│   ├── merge_features.py         ← builds features.csv from audio files
│   ├── compare_models.py         ← RF vs SVM cross-validation comparison
│   ├── train_model.py            ← orchestrates training, saves model
│   └── predict_single_voice.py   ← single-file prediction + XAI
│
├── app.py                  ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🔬 Feature Engineering

### A. Spectral Features — MFCCs (Mel-Frequency Cepstral Coefficients)

MFCCs model the short-term power spectrum of sound in a way that approximates the human auditory system.  13 coefficients are extracted per audio frame and their mean across all frames is used.  They capture vocal tract shape, which is affected by PD-related rigidity.

### B. Clinical Voice Biomarkers

| Feature | Description | Healthy Range | PD Range |
|---------|-------------|--------------|----------|
| **Jitter** | Cycle-to-cycle fundamental frequency (F0) variation | < 1% | > 1–2% |
| **Shimmer** | Cycle-to-cycle amplitude variation | < 3% | > 6% |
| **HNR** | Harmonics-to-Noise Ratio — energy in voiced vs noisy components | > 20 dB | < 15 dB |

These are computed via the **Praat** acoustic analysis program through the `praat-parselmouth` Python binding.

---

## 🤖 Model Comparison

Two classifiers are evaluated using **5-fold Stratified Cross-Validation**:

| Metric | Random Forest | SVM (RBF) |
|--------|--------------|-----------|
| Accuracy | ~0.91 | ~0.89 |
| Precision | ~0.92 | ~0.90 |
| Recall | ~0.90 | ~0.88 |
| F1-Score | ~0.91 | ~0.89 |

> ℹ️ Exact values depend on your dataset.  Results above are illustrative.

The model with the highest mean accuracy is automatically saved as `models/best_model.pkl`.

---

## 🔍 Explainable AI (XAI)

For **Random Forest**, the built-in `feature_importances_` attribute provides a Gini-impurity-based importance score for each of the 16 features.  The top 3 most influential features are surfaced in the web app and CLI output.

Example output:
```
Top 3 Contributing Features:
  1. HNR          importance = 0.1842  ████████████████
  2. Shimmer       importance = 0.1531  █████████████
  3. MFCC_1        importance = 0.1205  ██████████
```

For **SVM**, the absolute values of the decision-function coefficients are used as a proxy importance measure.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add Real Audio Data

Place `.wav` voice recordings in:
```
data/raw/healthy/     ← healthy speakers
data/raw/parkinson/   ← PD speakers
```

If no data is present, the pipeline auto-generates a **synthetic demonstration dataset**.

### 3. Train the Model

```bash
python src/train_model.py
```

This will:
- Generate or load the feature dataset
- Compare Random Forest vs SVM
- Save the best model to `models/best_model.pkl`

### 4. Run the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. CLI Prediction (Optional)

```bash
python src/predict_single_voice.py path/to/voice.wav
```

---

## 🖥️ Example Output

```
════════════════════════════════════════════════════
   VOICE BIOMARKER ANALYSIS — PARKINSON'S DETECTION
════════════════════════════════════════════════════
  Prediction  :  Parkinson's
  Risk Level  :  High
  Probability :  78.4%  [████████████████████████████████░░░░░░░░]
────────────────────────────────────────────────────
  Top 3 Contributing Features (XAI):
    1. HNR            importance = 0.1842  ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮
    2. Shimmer         importance = 0.1531  ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮
    3. MFCC_1          importance = 0.1205  ▮▮▮▮▮▮▮▮▮▮▮▮
════════════════════════════════════════════════════
```

---

## 📊 Dataset Notes

- If you have a real dataset (e.g., **UCI Parkinson's Dataset** by Max Little), place the `.wav` files into `data/raw/healthy/` and `data/raw/parkinson/`.
- The pipeline automatically uses real data if available, or generates synthetic data with realistic distributions for demonstration.
- The UCI dataset can be found at: https://archive.ics.uci.edu/ml/datasets/parkinsons

---

## 🔭 Future Scope

| Enhancement | Description |
|-------------|-------------|
| 🌐 SHAP Integration | Replace mean feature importance with SHAP values for per-sample explanations |
| 📱 Mobile Recording | In-browser audio recording via WebRTC instead of file upload |
| 🔁 Transfer Learning | Pre-trained audio models (wav2vec, HuBERT) as feature extractors |
| 📈 Longitudinal Tracking | Store predictions over time to track disease progression |
| 🧪 More Biomarkers | Add RPDE, DFA, PPE features from literature |
| 🏥 EHR Integration | Link with electronic health records for clinical deployment |
| 🩺 Multi-disease | Extend to detect ALS, essential tremor, or dysarthria |

---

## 📚 References

1. Little, M.A. et al. (2008). *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* IEEE Transactions on Biomedical Engineering.
2. Tsanas, A. et al. (2012). *Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease.* IEEE TBME.
3. Boersma, P. (2001). *Praat, a system for doing phonetics by computer.* Glot International.

---

## ⚠️ Disclaimer

This project is for **academic and research purposes only**.  It is not a certified medical device and must not be used for clinical diagnosis.  Always consult a qualified neurologist.

---

## 🧑‍💻 Built With

`Python` · `scikit-learn` · `librosa` · `praat-parselmouth` · `Streamlit` · `joblib` · `pandas` · `numpy`
