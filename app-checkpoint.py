# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Risk Predictor", layout="centered")

# --- Load artifacts once on start ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(r"C:\Users\Ahmed\Downloads\model_xgb_new.pkl")
    le = joblib.load('label_encoder.pkl')         # LabelEncoder fitted on y_train
    feature_names = joblib.load('feature_names.pkl')
    return model, le, feature_names

model, le, FEATURE_NAMES = load_artifacts()

st.title("Cancer Risk Level Predictor")
st.markdown("Predict `Risk_Level` (Low / Medium / High) from patient features.")

# --- Upload CSV or manual input ---
option = st.radio("Prediction mode", ("Upload CSV (batch)", "Manual input (single)"))

def preprocess_input(df):
    """
    Ensure DF has columns in FEATURE_NAMES order and numeric dtype.
    Fills missing columns with 0 and reorders.
    """
    # If uploaded CSV contains extra columns, keep only the feature columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in input — filling {len(missing)} missing columns with zeros: {missing}")
        for c in missing:
            df[c] = 0
    # Keep only required columns and in same order
    df = df[FEATURE_NAMES].copy()
    # Convert to numeric (coerce errors to NaN then fill)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

if option == "Upload CSV (batch)":
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type=['csv'])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        X = preprocess_input(input_df)
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)
        result = X.copy()
        result['Predicted_Risk_Level'] = preds
        # attach probabilities for each class
        for i, cls in enumerate(le.classes_):
            result[f'prob_{cls}'] = probs[:, i]
        st.success("Predictions ready")
        st.dataframe(result)
        st.download_button("Download results (CSV)", result.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

else:
    st.sidebar.header("Patient features (manual)")
    input_data = {}
    # create numeric inputs for each feature (you may want to group or change ranges)
    for feat in FEATURE_NAMES:
        # heuristic default: use 0 as default; change for Age/BMI etc if you prefer
        val = st.sidebar.number_input(feat, value=float(0.0))
        input_data[feat] = val

    if st.sidebar.button("Predict"):
        X_single = pd.DataFrame([input_data])
        X_proc = preprocess_input(X_single)
        pred_enc = model.predict(X_proc)[0]
        probs = model.predict_proba(X_proc)[0]
        pred = le.inverse_transform([pred_enc])[0]

        st.write("### Prediction")
        st.write(f"**Predicted Risk_Level:** {pred}")
        st.write("**Class probabilities:**")
        prob_df = pd.DataFrame({
            'class': list(le.classes_),
            'probability': probs
        }).sort_values('probability', ascending=False).reset_index(drop=True)
        st.table(prob_df)

        # optional: highlight 'High' probability and threshold advice
        high_prob = prob_df.loc[prob_df['class']=='High','probability'].values[0]
        st.info(f"Probability of High risk: {high_prob:.2f}")
        if high_prob >= 0.5:
            st.warning("High risk probability >= 0.5 — consider clinical follow-up.")
        else:
            st.success("High risk probability below 0.5")

# Footer
st.markdown("---")
st.caption("Model: class-weighted XGBoost (tuned via Optuna). Ensure uploaded CSV has the same feature columns used in training.")
