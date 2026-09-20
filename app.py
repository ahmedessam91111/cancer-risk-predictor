# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="Cancer Risk Predictor", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------- مسارات الملفات (نسبية عشان تشتغل في أي مكان) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_xgb_new.pkl")
LE_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, le, feature_names

model, le, FEATURE_NAMES = load_artifacts()

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
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in input — filling {len(missing)} missing columns with zeros: {missing}")
        for c in missing:
            df[c] = 0
    df = df[FEATURE_NAMES].copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

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

        X = preprocess_input(input_df)
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)

        result = X.copy()
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

    RULES = {
        "Age":                {"widget": "slider", "min": 20, "max": 90, "step": 1,   "default": 50},
        "Gender":             {"widget": "select", "options": [("Female", 0), ("Male", 1)]},
        "BMI":                {"widget": "number", "min": 15.0, "max": 50.0, "step": 0.1, "default": 25.0},
        "Overall_Risk_Score": {"widget": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
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
        ("📊 Engineered score",
         ["Overall_Risk_Score"]),
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

# ---------------- فوتر ----------------
st.markdown("---")
st.caption("Model: saved classifier (see training notebook). For research/education only — not a medical diagnosis.")