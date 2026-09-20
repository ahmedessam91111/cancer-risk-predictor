# 🧬 Cancer Risk Predictor

A 3-class ML web app (Streamlit) that predicts **Cancer Risk Level** — *Low / Medium / High* — from patient lifestyle and genetic factors.

## 📦 Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app (main entry) |
| `model_xgb_new.pkl` | Trained XGBoost classifier |
| `label_encoder.pkl` | LabelEncoder for risk classes |
| `feature_names.pkl` | Feature column names used in training |
| `Cancer_Risk_Prediction_(ML).ipynb` | Training notebook |
| `requirements.txt` | Python dependencies |

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🎯 Features

- 📁 **Batch CSV upload** — upload a CSV of patient records, get predictions + probabilities, download results
- 🧑 **Manual input** — enter patient features via widgets and get an instant risk prediction
- Models risk level with class probabilities

> ⚠️ For research/education only — **not** a medical diagnosis tool.