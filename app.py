# app.py
import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="Cancer Risk Predictor", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------- مسارات الملفات (نسبية عشان تشتغل في أي مكان) ----------------
# Issue #8 final layout: the app runs from the four production artifacts written
# by `python export_production.py` -- model.pkl (a Pipeline that embeds the
# fitted scaler, so no unscaled input can reach the forest), label_encoder.pkl,
# feature_names.pkl, and metadata.json. The deprecated root-level .pkl files are
# never read here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.join(BASE_DIR, "artifacts", "production")
MODEL_PATH      = os.path.join(PROD_DIR, "model.pkl")
LABEL_PATH      = os.path.join(PROD_DIR, "label_encoder.pkl")
FEATURES_PATH   = os.path.join(PROD_DIR, "feature_names.pkl")
METADATA_PATH   = os.path.join(PROD_DIR, "metadata.json")
PROD_FILES = (MODEL_PATH, LABEL_PATH, FEATURES_PATH, METADATA_PATH)

@st.cache_resource
def load_artifacts():
    missing = [p for p in PROD_FILES if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing production artifacts:\n\n```\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n```\n\n"
            "Build them first (no retraining needed if `train.py` already ran):\n\n"
            "```\npython export_production.py\n```"
        )
        st.stop()
    model = joblib.load(MODEL_PATH)          # Pipeline [StandardScaler -> RandomForest]
    le = joblib.load(LABEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    with open(METADATA_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    return model, le, feature_names, meta

model, le, FEATURE_NAMES, META = load_artifacts()

RISK_COLORS = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}

# ---------------- الألوان والتنسيق (CSS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', 'Segoe UI', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #081426 0%, #0f2c4a 50%, #14406b 100%);
}
h1, h2, h3, p, label, .stMarkdown,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #EAF6FF; }

.hero {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    border-radius: 20px; padding: 24px 30px; margin: 6px 0 20px;
    box-shadow: 0 12px 34px rgba(0,0,0,.35);
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; color: #fff; }
.hero p  { margin: 6px 0 0; color: #dfeaff; }

.card {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 16px; padding: 18px 22px; margin-bottom: 16px;
    backdrop-filter: blur(8px);
    box-shadow: 0 6px 18px rgba(0,0,0,.18);
}
.card h3 { margin: 0 0 12px; color: #FFD166; font-weight: 700; }

.badge {
    display: inline-block; padding: 10px 26px; border-radius: 999px;
    font-weight: 800; font-size: 1.3rem; color: #fff;
    box-shadow: 0 8px 22px rgba(0,0,0,.3);
}
.badge-Low    { background: linear-gradient(90deg,#2ecc71,#1abc9c); }
.badge-Medium { background: linear-gradient(90deg,#f39c12,#e67e22); }
.badge-High   { background: linear-gradient(90deg,#e74c3c,#c0392b); }

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: #fff; font-weight: 700; border: none; border-radius: 12px;
    padding: .6rem 1.5rem; transition: .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    color: #fff; box-shadow: 0 6px 18px rgba(6,182,212,.45); transform: translateY(-1px);
}

.stSlider, [data-testid="stSlider"] { accent-color: #06b6d4; }
input[type=range]::-webkit-slider-thumb { background: #06b6d4; }
input[type=range]::-moz-range-thumb     { background: #06b6d4; }

[data-testid="stRadio"] label { color: #EAF6FF; }
[data-testid="stRadio"] input  { accent-color: #06b6d4; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    border-radius: 14px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- الهيدر ----------------
st.markdown("""
<div class="hero">
  <div>
    <h1>🧬 Cancer Risk Predictor</h1>
    <p>Predict <b>Risk Level</b> (Low / Medium / High) from patient lifestyle &amp; genetic factors.</p>
  </div>
  <span class="badge-hero" style="font-size:.9rem;background:rgba(255,255,255,.18);
        border-radius:999px;padding:8px 16px;color:#fff;">3-class ML model</span>
</div>
""", unsafe_allow_html=True)

# ---------------- أدوات مساعدة ----------------
def preprocess_input(df):
    """
    Coerce a raw input frame into the exact matrix the model was trained on.

    Two things matter here and both were broken before Issue #8:

    1. Column set AND order must equal `FEATURE_NAMES`. The model's thresholds
       are positional, so a reordered CSV silently produces wrong predictions.
    2. Standardization happens INSIDE model.pkl: the shipped artifact is a
       Pipeline whose first step is the scaler fitted on the training split.
       Inputs are deliberately returned UNSCALED, so there is no code path that
       can reach the forest with raw values. (Older versions scaled here and
       relied on a loose scaler; that is exactly the failure mode Issue #2
       documented.)
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in input — filling {len(missing)} missing columns with zeros: {missing}")
        for c in missing:
            df[c] = 0
    extra = [c for c in df.columns if c not in FEATURE_NAMES]
    if extra:
        st.info(f"Ignoring {len(extra)} column(s) not used by this model: {extra}")
    df = df[FEATURE_NAMES].copy()
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)

def render_probabilities(probs):
    prob_df = pd.DataFrame({"class": list(le.classes_), "probability": probs}) \
        .sort_values("probability", ascending=False).reset_index(drop=True)
    for _, row in prob_df.iterrows():
        cls = row["class"]
        color = RISK_COLORS.get(cls, "#888")
        pct = float(row["probability"])
        st.markdown(
            f'<div style="margin:4px 0;font-weight:600;">{cls} &nbsp;'
            f'<span style="color:{color};">({pct*100:.1f}%)</span></div>',
            unsafe_allow_html=True)
        st.progress(pct)
    st.caption("Percentages are calibrated probabilities (out-of-fold "
               "recalibration, Issue #11). The predicted class comes from the "
               "uncalibrated risk model, so all published metrics remain valid.")

# ---------------- شريط اختيار الوضع ----------------
mode = st.radio("Prediction mode", ["📁 Batch upload (CSV)", "🧑 Manual input"], horizontal=True)

# ======================= وضع رفع CSV =======================
if "Batch" in mode:
    st.markdown('<div class="card"><h3>📁 Batch prediction</h3>'
                '<p style="margin:0;">Upload a CSV containing the same feature columns used in training. '
                'UTF-8 and Excel (Latin-1) encodings are both supported.</p></div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            input_df = pd.read_csv(uploaded_file, encoding="latin-1")

        X = preprocess_input(input_df)    # ordered + cleaned; scaling happens inside model.pkl
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)

        # Echo back the user's own columns, not the scaled internal matrix.
        result = input_df.copy()
        result["Predicted_Risk_Level"] = preds
        for i, cls in enumerate(le.classes_):
            result[f"prob_{cls}"] = probs[:, i]

        counts = result["Predicted_Risk_Level"].value_counts()
        cols = st.columns(1 + len(le.classes_))
        cols[0].metric("Total patients", len(result))
        for i, cls in enumerate(le.classes_):
            cols[i + 1].metric(f"Predicted {cls}", int(counts.get(cls, 0)))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True)
        st.download_button("⬇️ Download results (CSV)",
                           result.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= وضع الإدخال اليدوي =======================
else:
    st.markdown('<div class="card"><h3>🧑 Manual input — enter patient features</h3></div>',
                unsafe_allow_html=True)

    # Issue #5 proved Overall_Risk_Score is a 100% accurate threshold function of
    # Risk_Level, so asking the user for it made the form trivially self-answering.
    # It is no longer a model input, so it is no longer asked for.
    RULES = {
        "Age":                {"widget": "slider", "min": 20, "max": 90, "step": 1,   "default": 50},
        "Gender":             {"widget": "select", "options": [("Female", 0), ("Male", 1)]},
        "BMI":                {"widget": "number", "min": 15.0, "max": 50.0, "step": 0.1, "default": 25.0},
        "Family_History":     {"widget": "yesno"},
        "BRCA_Mutation":      {"widget": "yesno"},
        "H_Pylori_Infection": {"widget": "yesno"},
        "_default_0_10":      {"widget": "slider", "min": 0, "max": 10, "step": 1, "default": 5},
    }

    GROUPS = [
        ("👤 Demographics",
         ["Age", "Gender", "BMI"]),
        ("🏃 Lifestyle & Environment",
         ["Smoking", "Alcohol_Use", "Obesity", "Diet_Red_Meat", "Diet_Salted_Processed",
          "Fruit_Veg_Intake", "Physical_Activity", "Physical_Activity_Level",
          "Air_Pollution", "Occupational_Hazards", "Calcium_Intake"]),
        ("🧬 Genetic / Medical flags",
         ["Family_History", "BRCA_Mutation", "H_Pylori_Infection"]),
    ]

    def build_widget(feat):
        rule = RULES.get(feat, RULES["_default_0_10"])
        kind = rule.get("widget")
        if kind == "select":
            labels = [o[0] for o in rule["options"]]
            idx = st.radio(feat, labels, horizontal=True)
            return dict(rule["options"])[idx]
        if kind == "yesno":
            val = st.radio(feat, ["No", "Yes"], horizontal=True)
            return 1 if val == "Yes" else 0
        if kind == "number":
            return st.number_input(feat, min_value=rule["min"], max_value=rule["max"],
                                   step=rule["step"], value=rule["default"], format="%.1f")
        # slider (int or float)
        return st.slider(feat, min_value=rule["min"], max_value=rule["max"],
                         step=rule["step"], value=rule["default"])

    input_data = {}
    known = set()

    for title, feats in GROUPS:
        present = [f for f in feats if f in FEATURE_NAMES]
        if not present:
            continue
        known.update(feats)
        st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, feat in enumerate(present):
            with cols[idx % 3]:
                input_data[feat] = build_widget(feat)
        st.markdown("</div>", unsafe_allow_html=True)

    others = [f for f in FEATURE_NAMES if f not in known]
    if others:
        st.markdown('<div class="card"><h3>⚙️ Other features</h3>', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, feat in enumerate(others):
            with cols[idx % 3]:
                input_data[feat] = build_widget(feat)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🧪 Predict risk level", use_container_width=True):
        X_single = pd.DataFrame([input_data])
        X_proc = preprocess_input(X_single)
        pred_enc = model.predict(X_proc)[0]
        probs = model.predict_proba(X_proc)[0]
        pred = le.inverse_transform([pred_enc])[0]

        # البطاقة الملونة بالناتج
        st.markdown(f'<div style="text-align:center;padding:10px 0;">'
                    f'<span class="badge badge-{pred}">{pred} risk</span></div>',
                    unsafe_allow_html=True)

        # احتمالات كل فئة
        st.markdown('<div class="card"><h3>📈 Class probabilities</h3>', unsafe_allow_html=True)
        render_probabilities(probs)
        st.markdown("</div>", unsafe_allow_html=True)

        advice = {
            "High":   "⚠️ High risk detected — clinical follow-up is strongly recommended.",
            "Medium": "🟠 Moderate risk — recommend regular monitoring and lifestyle review.",
            "Low":    "🟢 Low risk — keep up a healthy lifestyle and routine screening.",
        }
        st.info(advice.get(pred, ""))

# ---------------- Model provenance (Issue #8) ----------------
with st.sidebar:
    st.markdown("### 📋 Model card")
    m = META.get("metrics", {})
    st.markdown(
        f"""
| | |
|---|---|
| **Version** | `{META.get('model_version', '?')}` |
| **Estimator** | Random Forest, {META.get('config', {}).get('n_estimators', '?')} trees |
| **Class balancing** | `{META.get('resample', '?')}` |
| **Features** | {len(FEATURE_NAMES)} (leak-free) |
| **Trained** | {META.get('trained_at', '?')[:19].replace('T', ' ')} UTC |

**Held-out test (400 rows)**
| Metric | Value |
|---|---|
| Accuracy | {m.get('accuracy', float('nan')):.3f} |
| Macro F1 | {m.get('f1_macro', float('nan')):.3f} |
| Macro recall | {m.get('recall_macro', float('nan')):.3f} |

<details>
<summary>Why not a higher number?</summary>

An earlier version of this model scored **F1 0.9975** by including
`Overall_Risk_Score` — a column that turned out to be a direct encoding of the
answer. It also never received standardized inputs, which is why this app used
to return *Medium* for every single patient.

Both defects are fixed. The scores above are from a model that has only ever
seen the 17 real risk factors. See `docs/TRAINING_AND_LEAKAGE.md`.
</details>
"""
    , unsafe_allow_html=True)

# ---------------- فوتر ----------------
st.markdown("---")
st.caption(
    f"Model v{META.get('model_version', '?')} — artifacts rebuilt with "
    "`python export_production.py`. "
    "Metrics are from a held-out split on synthetic data. "
    "For research/education only — not a medical diagnosis."
)