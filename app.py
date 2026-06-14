"""
app.py
Streamlit Web Application — Voice Biomarker Analysis for Early Parkinson's Detection
Run with:   streamlit run app.py
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# ─── Page Config ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title  = "VoicePD — Parkinson's Voice Analyzer",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── Header gradient card ── */
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.header-card h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #f8fafc;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.header-card p  { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── Result cards ── */
.result-card {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-healthy {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
}
.card-parkinson {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
}
.card-neutral {
    background: #1e293b;
    border: 1px solid #334155;
}

.card-label   { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.25rem; }
.card-value   { font-size: 1.75rem; font-weight: 700; color: #f8fafc; font-family: 'Space Mono', monospace; }
.card-sub     { font-size: 0.85rem; color: #cbd5e1; margin-top: 0.15rem; }

/* ── Risk badge ── */
.risk-low      { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
.risk-moderate { background: #713f12; color: #fbbf24; border: 1px solid #f59e0b; }
.risk-high     { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }
.risk-badge    { display: inline-block; border-radius: 999px; padding: 0.25rem 0.85rem;
                 font-weight: 600; font-size: 0.9rem; letter-spacing: 0.04em; }

/* ── Progress bar override ── */
.stProgress > div > div > div { border-radius: 8px; }

/* ── Feature bars ── */
.feat-row { display: flex; align-items: center; margin-bottom: 0.75rem; gap: 0.75rem; }
.feat-name { min-width: 110px; font-size: 0.85rem; color: #cbd5e1; font-family: 'Space Mono', monospace; }
.feat-bar-bg { flex: 1; background: #1e293b; border-radius: 999px; height: 10px; overflow: hidden; }
.feat-bar-fill { height: 100%; border-radius: 999px;
                 background: linear-gradient(90deg, #6366f1, #a78bfa); }
.feat-score { min-width: 52px; text-align: right; font-size: 0.8rem;
              color: #a78bfa; font-family: 'Space Mono', monospace; }

/* ── Upload zone ── */
.upload-hint { font-size: 0.85rem; color: #64748b; text-align: center; margin-top: 0.5rem; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #6366f1; margin-bottom: 0.75rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.8rem; color: #64748b;
    text-align: center; margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────── #
st.markdown("""
<div class="header-card">
    <h1>🧠 VoicePD Analyzer</h1>
    <p>Voice Biomarker Analysis for Early Parkinson's Disease Detection · Powered by Explainable AI</p>
</div>
""", unsafe_allow_html=True)


# ─── Lazy imports (avoid crashing on cold start) ──────────────────────────── #
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    import joblib
    model_path = ROOT_DIR / "models" / "best_model.pkl"
    if not model_path.exists():
        return None, None
    pipeline = joblib.load(model_path)
    meta_path = ROOT_DIR / "models" / "model_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return pipeline, meta


# ─── Prediction helper ─────────────────────────────────────────────────────── #

def run_prediction(wav_path: str, pipeline, feature_names: list) -> dict:
    from predict_single_voice import (
        extract_features_for_prediction,
        probability_to_risk,
        get_top_features,
    )

    X            = extract_features_for_prediction(wav_path)
    label        = int(pipeline.predict(X)[0])
    proba        = pipeline.predict_proba(X)[0]
    pk_prob      = float(proba[1])
    prediction   = "Parkinson's" if label == 1 else "Healthy"
    risk_level   = probability_to_risk(pk_prob)
    top_features = get_top_features(pipeline, feature_names, top_n=3)
    raw_features = dict(zip(feature_names, X[0]))

    return {
        "prediction":   prediction,
        "risk_level":   risk_level,
        "probability":  pk_prob,
        "top_features": top_features,
        "raw_features": raw_features,
    }


# ─── Sidebar: model info ───────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
This tool analyses sustained vowel recordings ("AAAA") to detect
early Parkinson's disease biomarkers using:

- **13 MFCC** spectral features  
- **Jitter** (pitch variation)  
- **Shimmer** (amplitude variation)  
- **HNR** (harmonics-to-noise ratio)

Two models are compared — **Random Forest** and **SVM** — and the
best is automatically selected.

Explainable AI highlights the top 3 features driving each prediction.
    """)
    st.divider()
    st.caption("⚠️ For research use only. Not a medical device.")


# ─── Main layout ──────────────────────────────────────────────────────────── #
col_upload, col_results = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="section-label">01 · Upload Voice Recording</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a .wav file",
        type=["wav"],
        help="Record yourself sustaining the vowel 'AAAA' for 3–5 seconds, then upload.",
        label_visibility="collapsed",
    )

    st.markdown(
        '<p class="upload-hint">🎙️ Record a sustained vowel "AAAA" (3–5 seconds) · .wav format</p>',
        unsafe_allow_html=True,
    )

    if uploaded:
        st.audio(uploaded, format="audio/wav")

    st.divider()
    st.markdown('<div class="section-label">02 · System Status</div>',
                unsafe_allow_html=True)

    pipeline, meta = load_model()

    if pipeline is None:
        st.error("⚠️ No trained model found.  Run `python src/train_model.py` first.")
        model_ready = False
    else:
        model_name = meta.get("model_name", "Unknown")
        n_features = meta.get("n_features", 16)
        n_samples  = meta.get("n_samples",  "—")

        st.success("✅ Model loaded")
        st.markdown(f"""
| Parameter     | Value |
|---------------|-------|
| Algorithm     | {model_name} |
| Input features | {n_features} |
| Training samples | {n_samples} |
        """)
        model_ready = True

    # ── Analyse button ─────────────────────────────────────────────────── #
    st.markdown("")
    analyse_clicked = st.button(
        "🔬  Analyse Voice",
        type="primary",
        disabled=(not model_ready or uploaded is None),
        use_container_width=True,
    )


# ─── Results column ────────────────────────────────────────────────────────── #
with col_results:
    st.markdown('<div class="section-label">03 · Analysis Results</div>',
                unsafe_allow_html=True)

    if not uploaded:
        st.info("Upload a .wav file and click **Analyse Voice** to see results.")

    elif analyse_clicked or st.session_state.get("last_result"):

        if analyse_clicked:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            feature_names = meta.get(
                "feature_names",
                [f"MFCC_{i+1}" for i in range(13)] + ["Jitter", "Shimmer", "HNR"],
            )

            with st.spinner("Extracting features and running prediction …"):
                try:
                    result = run_prediction(tmp_path, pipeline, feature_names)
                    st.session_state["last_result"] = result
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()
        else:
            result = st.session_state["last_result"]

        # ── Prediction card ─────────────────────────────────────────────── #
        is_pk   = result["prediction"] == "Parkinson's"
        card_cls = "card-parkinson" if is_pk else "card-healthy"
        icon     = "🔴" if is_pk else "🟢"

        risk_css_map = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}
        risk_cls = risk_css_map[result["risk_level"]]

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
<div class="result-card {card_cls}">
    <div class="card-label">Prediction</div>
    <div class="card-value">{icon} {result['prediction']}</div>
</div>
""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
<div class="result-card card-neutral">
    <div class="card-label">Risk Level</div>
    <div class="card-value">
        <span class="risk-badge {risk_cls}">{result['risk_level']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        with r3:
            pct = result["probability"] * 100
            st.markdown(f"""
<div class="result-card card-neutral">
    <div class="card-label">PD Probability</div>
    <div class="card-value">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        # ── Probability bar ─────────────────────────────────────────────── #
        prob = result["probability"]
        color = "#ef4444" if prob > 0.65 else ("#f59e0b" if prob > 0.35 else "#22c55e")
        st.progress(float(prob))

        # ── XAI: top features ────────────────────────────────────────────── #
        st.markdown("---")
        st.markdown('<div class="section-label">04 · Explainable AI — Top Contributing Features</div>',
                    unsafe_allow_html=True)

        top_feats = result["top_features"]
        max_imp   = max(imp for _, imp in top_feats) if top_feats else 1.0

        feat_html = ""
        for feat_name, imp in top_feats:
            fill_pct = (imp / max_imp) * 100
            feat_html += f"""
<div class="feat-row">
    <span class="feat-name">{feat_name}</span>
    <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:{fill_pct:.1f}%"></div>
    </div>
    <span class="feat-score">{imp:.4f}</span>
</div>"""

        st.markdown(f'<div style="margin-top:0.5rem">{feat_html}</div>',
                    unsafe_allow_html=True)

        # ── Raw feature values ───────────────────────────────────────────── #
        with st.expander("🔎 View all extracted features"):
            raw = result["raw_features"]
            import pandas as pd
            df_feat = pd.DataFrame(
                [(k, f"{v:.4f}") for k, v in raw.items()],
                columns=["Feature", "Value"],
            )
            st.dataframe(df_feat, use_container_width=True, hide_index=True)

        # ── Clinical feature spotlight ───────────────────────────────────── #
        st.markdown("---")
        st.markdown('<div class="section-label">05 · Clinical Biomarker Snapshot</div>',
                    unsafe_allow_html=True)

        raw = result["raw_features"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Jitter",  f"{raw.get('Jitter',  0):.5f}",  help="Cycle-to-cycle F0 variation. PD > 1%")
        with c2:
            st.metric("Shimmer", f"{raw.get('Shimmer', 0):.5f}",  help="Amplitude variation. PD > 6%")
        with c3:
            st.metric("HNR",     f"{raw.get('HNR',     0):.2f} dB", help="Harmonics-to-noise ratio. PD typically < 15 dB")

    else:
        st.info("Upload a .wav file and click **Analyse Voice** to see results.")

# ── Model Comparison ───────────────────────────────────── #
st.markdown("---")
st.markdown(
    '<div class="section-label">07 · Model Comparison</div>',
    unsafe_allow_html=True
)

rf_acc = meta.get("rf_accuracy", None)
svm_acc = meta.get("svm_accuracy", None)
best_model = meta.get("model_name", "Unknown")

if rf_acc and svm_acc:
    st.write(f"Random Forest Accuracy: {rf_acc:.3f}")
    st.write(f"SVM Accuracy: {svm_acc:.3f}")

    if best_model == "SVM":
        st.success(f"Best Model: {best_model}")
    else:
        st.info(f"Best Model: {best_model}")
else:
    st.warning("Model comparison data not available")

# ── Model Comparison Graph ───────────────────────────── #
st.markdown("---")
st.markdown(
    '<div class="section-label">08 · Model Performance Comparison</div>',
    unsafe_allow_html=True
)

rf_acc = meta.get("rf_accuracy", None)
svm_acc = meta.get("svm_accuracy", None)

if rf_acc and svm_acc:
    import pandas as pd

    df = pd.DataFrame({
        "Model": ["Random Forest", "SVM"],
        "Accuracy": [rf_acc, svm_acc]
    })

    st.bar_chart(df.set_index("Model"))
else:
    st.warning("Model comparison data not available")

    # ── Confusion Matrix Heatmap ───────────────────────────── #
st.markdown("---")
st.markdown(
    '<div class="section-label">09 · Confusion Matrix (Heatmap)</div>',
    unsafe_allow_html=True
)

cm = meta.get("confusion_matrix", None)

if cm:
    import matplotlib.pyplot as plt
    import numpy as np

    tn = cm["tn"]
    fp = cm["fp"]
    fn = cm["fn"]
    tp = cm["tp"]

    matrix = np.array([[tn, fp],
                       [fn, tp]])

    fig, ax = plt.subplots()

    im = ax.imshow(matrix)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Parkinson"])
    ax.set_yticklabels(["Healthy", "Parkinson"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Add numbers inside cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center", color="white", fontsize=12)

    st.pyplot(fig)

    # Summary
    st.success(f"✔ True Positives: {tp} | ✔ True Negatives: {tn}")
    st.warning(f"⚠ False Positives: {fp} | ⚠ False Negatives: {fn}")

else:
    st.warning("Confusion matrix data not available")


# ─── Disclaimer ────────────────────────────────────────────────────────────── #
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool is intended for academic and research purposes only.
It is <strong>not a certified medical device</strong> and should not be used for clinical diagnosis.
Always consult a qualified neurologist for medical advice.
</div>
""", unsafe_allow_html=True)

