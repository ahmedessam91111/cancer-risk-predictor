# Model Input Contract (Issue #14)

**Status:** Verified. The assertions below are derived from the deployed artifacts, not the notebook or filenames.

**Script:** `verify_model_contract.py` (read-only). It exits non-zero only on hard mismatches (FAIL); expected warnings (e.g., `forest.feature_names_in_ is None` because the forest was fit on a numpy array) do not cause a FAIL.

---

## 1. What the saved model expects

| Property | Value | Source (artifact-first) |
|---|---|---|
| **Model artifact** | `artifacts/production/model.pkl` (sklearn `Pipeline`): `StandardScaler` → `CalibratedModel` (hybrid, Issue #11). `predict()` = raw forest argmax (metrics unchanged); `predict_proba()` = OOF-calibrated. | production export (export_production.py) |
| **Estimator type** | `CalibratedModel` wrapping `RandomForestClassifier` | `verify_model_contract.py` (final estimator) |
| **Number of features (n_features_in_)** | **17** | `pipeline.n_features_in_ == 17`, `CalibratedModel.n_features_in_ == 17`, `scaler.n_features_in_ == 17` (all read directly from the artifacts) |
| **Records column names?** | **Pipeline/scaler do; the forest does not.** | `pipeline.feature_names_in_` is **present**; `scaler.feature_names_in_` is **present** (list of 17 names, in canonical order). `forest.feature_names_in_` is **None** because `train.py` fit the forest on a numpy array (StandardScaler output). `CalibratedModel.feature_names_in_` delegates to the forest (→ None). The name record therefore lives in the pipeline/scaler and in `feature_names.pkl`. |

---

## 2. Canonical feature order (exact input order)

The input order is the **contract**. Any reordering silently changes what each position means unless corrected.

| # | Feature | Notes |
|---|---|---|
| 1 | Age | continuous (int) |
| 2 | Gender | binary 0/1 (app labels 0=Female, 1=Male — label semantics not stored in the model, only numeric values) |
| 3 | Smoking | ordinal 0–10 |
| 4 | Alcohol_Use | ordinal 0–10 |
| 5 | Obesity | ordinal 0–10 |
| 6 | Family_History | binary 0/1 |
| 7 | Diet_Red_Meat | ordinal 0–10 |
| 8 | Diet_Salted_Processed | ordinal 0–10 |
| 9 | Fruit_Veg_Intake | ordinal 0–10 |
| 10 | Physical_Activity | ordinal 0–10 |
| 11 | Air_Pollution | ordinal 0–10 |
| 12 | Occupational_Hazards | ordinal 0–10 |
| 13 | BRCA_Mutation | binary 0/1 |
| 14 | H_Pylori_Infection | binary 0/1 |
| 15 | Calcium_Intake | ordinal 0–10 |
| 16 | BMI | continuous (float) |
| 17 | Physical_Activity_Level | ordinal 0–10 |

**Where this list/order comes from:** `train.py` derives it via `derive_features(df)` as `df.columns` after removing `["Patient_ID","Cancer_Type","Risk_Level","Overall_Risk_Score"]`. That gives the CSV's **physical column order** after those drops; that exact list is stored in `artifacts/production/feature_names.pkl` (by `export_production.py` copying `bundle["feature_names"]`, which comes from `train.py`). The scaler's `feature_names_in_` equals this list in the same order. **There is no reordering in training.**

---

## 3. Does the saved model record column names?

| Artifact | `feature_names_in_` | Meaning |
|---|---|---|
| `Pipeline` (model.pkl) | **present, 17 names** | Records the canonical names it was fitted with (in order). |
| `StandardScaler` (inside Pipeline) | **present, 17 names** | Fitted on a labelled DataFrame, so sklearn preserved column names. This is the strongest order guarantee inside the pipeline. |
| `CalibratedModel(forest)` | **delegates → forest; forest has None** | The underlying forest was fit on numpy arrays (no column labels), so it cannot record them. |
| `RandomForestClassifier` | **None** | Expected given the training code used `scaler.fit_transform(X_train)` (numpy). |

**Conclusion:** the **pipeline + scaler** know the canonical order by name. The forest does not. The external contract is `feature_names.pkl` (a separate, explicit file) that all recorded witnesses (`feature_names.pkl`, `bundle.feature_names`, `scaler.feature_names_in_`) agree on **byte-identical, in order**. `metadata.json` does **not** duplicate the list (by design); the checksum of `feature_names.pkl` is recorded in `metadata.files["feature_names.pkl"].sha256` and verified by `verify_model_contract.py`.

---

## 4. If a CSV comes in with columns reordered

The app **defends the order explicitly** (Issue #13/#14 check).

`app.py` → `preprocess_input()`:
1. `missing`: warns and fills with 0
2. `extra`: info only, ignores
3. **`df = df[FEATURE_NAMES].copy()` — reorders to the canonical list**
4. Coerce to numeric, fillna(0)
5. Feed the ordered DataFrame to `model.predict(X)` (the pipeline sees the columns in canonical order)

End-to-end test (`verify_model_contract.py`, assertions 12–13):
- Take a DataFrame with the **same 17 columns, permuted** (e.g., last 4 moved to front). Feed directly to `model.pkl` (pipeline) **without** the app's reorder: predictions **differ on ~9%** of rows (a visible flip). 
- Feed the same permuted upload through the app's `preprocess_input()` (which does `df[FEATURE_NAMES]`): predictions become **bit-identical** to the canonical order.
- Also: when the pipeline records `feature_names_in_`, sklearn can raise on mismatched/duplicated column sets if given a DataFrame with duplicate names in a wrong layout; in this case `verify_model_contract.py` (11) documents whether the pipeline defends by name. The practical protection is the app's explicit **positional reorder** to the canonical list.

**Plain statement:** the column order **is verified by cross-checking the three witnesses** (`feature_names.pkl`, `bundle.feature_names`, `scaler.feature_names_in_`) and is **enforced at runtime by the app** via `df[FEATURE_NAMES]` (which restores canonical order). The model itself does not rely solely on implicit positional assumption because the serving pipeline uses the canonical feature list at the preprocess step; order-correctness is therefore **verified (witnessed) + enforced (app reorder)**, not merely assumed.

---

## 5. Value ranges (expected)

Derived from the **full validated dataset** (`cancer-risk-factors.csv`, 2,000 rows, SHA-256 `01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5`, see `docs/DATASET_PROVENANCE.md`).

| Feature | Dtype | Min | Max | Scale | How established |
|---|---|---|---|---|---|
| Age | int64 | 25 | 90 | continuous | dataset statistics (checksum-gated, `verify_dataset.py`) |
| Gender | int64 | 0 | 1 | binary 0/1 | same |
| Smoking | int64 | 0 | 10 | ordinal 0–10 | same |
| Alcohol_Use | int64 | 0 | 10 | ordinal 0–10 | same |
| Obesity | int64 | 0 | 10 | ordinal 0–10 | same |
| Family_History | int64 | 0 | 1 | binary 0/1 | same |
| Diet_Red_Meat | int64 | 0 | 10 | ordinal 0–10 | same |
| Diet_Salted_Processed | int64 | 0 | 10 | ordinal 0–10 | same |
| Fruit_Veg_Intake | int64 | 0 | 10 | ordinal 0–10 | same |
| Physical_Activity | int64 | 0 | 10 | ordinal 0–10 | same |
| Air_Pollution | int64 | 0 | 10 | ordinal 0–10 | same |
| Occupational_Hazards | int64 | 0 | 10 | ordinal 0–10 | same |
| BRCA_Mutation | int64 | 0 | 1 | binary 0/1 | same |
| H_Pylori_Infection | int64 | 0 | 1 | binary 0/1 | same |
| Calcium_Intake | int64 | 0 | 10 | ordinal 0–10 | same |
| BMI | float64 | 15 | 41.4 | continuous | same |
| Physical_Activity_Level | int64 | 0 | 10 | ordinal 0–10 | same |

> Binary/ordinal conventions match the dataset (0/1 for flags; 0–10 ordinal intensity/frequency). App labels Gender 0=Female/1=Male in the UI; no numeric semantics are stored in the model. The pipeline treats all inputs as numeric in the above ranges (no one-hot). Out-of-range values are not rejected (numeric coercion + fillna(0) by the app's preprocess step) — behavior is defined by the preprocess_input path in `app.py`.