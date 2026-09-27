# Dataset Provenance — Cancer Risk Prediction

**Scope:** Issue #1 — make the training dataset reproducible.
**Status:** investigation complete. Nothing was retrained, tuned, or modified.
**Verified by:** `python verify_dataset.py` (read-only; writes nothing)

---

## 1. Dataset identity

| Property | Value |
|---|---|
| Filename | `cancer-risk-factors.csv` |
| Location | repository root |
| Size | 141,851 bytes |
| **SHA-256** | `01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5` |
| Version / date | **none recorded** — no version file, no download script, no dataset card |
| Tracked in git | **No** |

### Source

The dataset is a public "cancer risk factors" synthetic dataset. It is **not
vendored or downloaded by any code in this repo** — it was uploaded by hand to
a Google Colab session and read from Colab's local filesystem:

```python
# Cancer_Risk_Prediction_(ML).ipynb  cell 2
df = pd.read_csv('/content/cancer-risk-factors.csv')

# Multiclass_Classfication-checkpoint.ipynb  cell 2
df = pd.read_csv("/content/cancer-risk-factors.csv")
```

`/content/` is a Colab-only path. Neither notebook runs outside Colab without
editing, and no cell records a download URL, a Kaggle dataset slug, a version
number, or a retrieval date. The original upstream source therefore **cannot be
confirmed from the repository** — the file is the only surviving copy.

Filesystem timestamps (`2026-09-13`) date the file's arrival, not its upstream
publication.

---

## 2. Shape and schema

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 21 |
| Model features | **18** |
| Non-feature columns | 3 — `Patient_ID`, `Cancer_Type`, `Risk_Level` |
| Target | `Risk_Level` (3 classes) |

### Target distribution

| Class | Count | Share |
|---|---:|---:|
| Medium | 1,574 | 78.7% |
| Low | 324 | 16.2% |
| High | 102 | **5.1%** |

Severely imbalanced. Raw accuracy is not a meaningful metric for this dataset
— a constant "always Medium" classifier scores 0.787. This matters for Issue #3.

### Data quality

| Check | Result |
|---|---|
| Missing values | **0** cells across all 21 columns |
| Fully duplicated rows | **0** |
| Duplicate `Patient_ID` | **0** (2,000 unique of 2,000) |
| Duplicate feature rows | **0** |
| Text / categorical columns | 3 — `Patient_ID`, `Cancer_Type`, `Risk_Level` |
| `Cancer_Type` values | `Breast`, `Colon`, `Lung`, `Prostate`, `Skin` (5) |

The dataset is completely clean — no imputation, encoding, or dedup logic is
needed anywhere in the pipeline.

### Feature types and ranges

| # | Feature | dtype | min | max | mean | std | distinct |
|--:|---|---|--:|--:|--:|--:|--:|
| 0 | Age | int | 25 | 90 | 63.25 | 10.46 | 61 |
| 1 | Gender | int | 0 | 1 | 0.49 | 0.50 | 2 |
| 2 | Smoking | int | 0 | 10 | 5.16 | 3.33 | 11 |
| 3 | Alcohol_Use | int | 0 | 10 | 5.04 | 3.26 | 11 |
| 4 | Obesity | int | 0 | 10 | 5.97 | 3.06 | 11 |
| 5 | Family_History | int | 0 | 1 | 0.19 | 0.40 | 2 |
| 6 | Diet_Red_Meat | int | 0 | 10 | 5.19 | 3.15 | 11 |
| 7 | Diet_Salted_Processed | int | 0 | 10 | 4.56 | 3.09 | 11 |
| 8 | Fruit_Veg_Intake | int | 0 | 10 | 4.93 | 3.05 | 11 |
| 9 | Physical_Activity | int | 0 | 10 | 4.02 | 2.98 | 11 |
| 10 | Air_Pollution | int | 0 | 10 | 5.32 | 3.21 | 11 |
| 11 | Occupational_Hazards | int | 0 | 10 | 4.98 | 3.21 | 11 |
| 12 | BRCA_Mutation | int | 0 | 1 | 0.03 | 0.18 | 2 |
| 13 | H_Pylori_Infection | int | 0 | 1 | 0.20 | 0.40 | 2 |
| 14 | Calcium_Intake | int | 0 | 10 | 3.94 | 3.05 | 11 |
| 15 | Overall_Risk_Score | float | 0.029 | 0.852 | 0.454 | 0.123 | 2000 |
| 16 | BMI | float | 15.0 | 41.4 | 26.18 | 3.95 | 208 |
| 17 | Physical_Activity_Level | int | 0 | 10 | 4.94 | 3.17 | 11 |

Two encoding conventions coexist and are **not documented anywhere**:

- **Binary** — `Gender`, `Family_History`, `BRCA_Mutation`, `H_Pylori_Infection` use 0/1.
  `app.py` renders `Gender` as `Female=0, Male=1`; the CSV's actual mapping is
  **unverified** (no documentation states which value is which).
- **Ordinal 0–10** — 11 features are integer severity scales where 0 appears to
  mean "none" and 10 "highest". `app.py` silently labels these as 0–10 sliders
  with no units.

`BRCA_Mutation` is 1 for only 65 of 2,000 patients (3.3%).

---

## 3. Preprocessing and feature engineering

There is **no feature engineering**. The complete transformation chain is:

1. **Load** the CSV (no encoding argument, no `parse_dates`, no dtype hints).
2. **Drop 3 columns** — `Risk_Level`, `Patient_ID`, `Cancer_Type`.
3. **Encode the target** — `LabelEncoder().fit_transform(df['Risk_Level'])`.
   `LabelEncoder` sorts its classes, so the mapping is deterministic:
   `High=0, Low=1, Medium=2`.
4. **Split** — see §4.
5. **Standardize** — `StandardScaler()`, fit on train, applied to test.
6. **Fit** — `RandomForestClassifier(random_state=42, n_estimators=100)`.
7. **Save** — `joblib.dump(model, 'model_xgb_new.pkl')`.

Feature order is simply **the CSV's column order after the drop in step 2**.
Nothing reorders, encodes, or derives features.

### ⚠ Target leakage in the shipped model

`Overall_Risk_Score` is included as a model input and accounts for **70.3% of
total feature importance** — the model is overwhelmingly a lookup on that one
column.

The notebook author identified this as leakage and rebuilt without it
(cells 18, 25, and 53 all drop `Overall_Risk_Score`, cell 53 comments
*"droping overall_risk_score from the feature list (as it is leaking the
answer)"*). **The shipped artifact was produced before that correction and
still consumes the leaked feature.**

### ⚠ The scaler is not part of the artifact set

Step 5 exists in training but no `scaler.pkl` was ever written. Only three
artifacts exist: `model_xgb_new.pkl`, `label_encoder.pkl`, `feature_names.pkl`.
`app.py` therefore feeds **raw** values to a model fit on **standardized**
values. Measured on the 400 held-out rows:

| Input | Predicted distribution |
|---|---|
| As `app.py` does (raw) | Medium 400 / Low 0 / High 0 |
| As trained (scaled) | Medium 316 / Low 65 / High 19 |

**The app predicts `Medium` for 100% of patients and never predicts `High`.**
This is Issue #2 and was deliberately not fixed here.

---

## 4. Train/test split

Identical in both notebooks:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| Parameter | Value |
|---|---|
| Test size | 0.20 (400 rows train-on, 400 held out) |
| `random_state` | 42 |
| `stratify` | `y` — preserves 78.7 / 16.2 / 5.1 in both splits |
| Shuffle | default `True` |

Deterministic given the dataset. The split is reproducible; the seed is fixed.

---

## 5. Features used to train `model_xgb_new.pkl`

**18 features, in this exact order** (CSV column order after the 3 drops):

```
Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History,
Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake,
Physical_Activity, Air_Pollution, Occupational_Hazards,
BRCA_Mutation, H_Pylori_Infection, Calcium_Intake,
Overall_Risk_Score, BMI, Physical_Activity_Level
```

### The order is not stored in the model

```python
model.feature_names_in_   # -> None
```

The model was fit on a **numpy array** (`scaler.fit_transform` discards column
names), so the estimator retains no record of its own input order. The order
survives only in the separate `feature_names.pkl` — and that file is
**hand-written**, not generated by training:

```python
# import joblib.py
feats = getattr(model, "feature_names_in_", None)
if feats is None:
    feats = ["Age", "Gender", ...]   # <-- hardcoded fallback list
joblib.dump(feats, "feature_names.pkl")

le = LabelEncoder().fit(["Low", "Medium", "High"])   # <-- hardcoded classes
joblib.dump(le, "label_encoder.pkl")
```

Both the feature order and the label mapping are **reconstructed by hand**,
months after training. They happen to be correct today — `verify_dataset.py`
confirms the order equals the derived CSV order and the mapping equals
`{High:0, Low:1, Medium:2}` — but nothing *enforces* that, and the
`label_encoder.pkl` was in fact rebuilt with the class list written
`["Low","Medium","High"]`, which only works because `LabelEncoder` re-sorts.

---

## 6. How the dataset connects to `model_xgb_new.pkl`

Confirmed by in-memory reproduction (nothing written):

```
estimator type match : True  (RandomForestClassifier)
n_estimators         : 100
max_depth            : None
feature_importances_ : identical
all 100 trees        : identical
classes_             : [0 1 2]
```

`model_xgb_new.pkl` is exactly `RandomForestClassifier(random_state=42,
n_estimators=100)` fit on step-5 output. The chain is proven.

### Which model was actually saved?

`Cancer_Risk_Prediction_(ML).ipynb` runs ~15 experiments that all rebind the
same globals. The final save is:

```python
# cell 50
joblib.dump(model, 'model_xgb_new.pkl')
```

`model` is assigned at notebook top level only in **cell 10**. Cells 35 and 47
also assign `model`, but both are *function-local* inside `def objective(trial)`
and never escape. So the save serialises **cell 10's untuned baseline**.

Consequences:

- Every Optuna study, the tuned `final_rf` (`n_estimators=347, max_depth=19`),
  and all XGBoost experiments in the notebook were **discarded**.
- The shipped artifact is the notebook's *first attempt*, not its best result.
- `Multiclass_Classfication-checkpoint.ipynb` contains no `joblib.dump` at all,
  so it produced no artifact.

The filename is actively misleading: `model_xgb_new.pkl` is a
**`RandomForestClassifier`**, and `README.md` describes it as "Trained XGBoost
classifier". `requirements.txt` still pins `xgboost` for a model that does not
use it.

---

## 7. Reproducibility problems

| # | Problem | Severity |
|---|---|---|
| 1 | Dataset is **untracked in git**. It exists in no commit, no branch, no tag. One `git clean` destroys it permanently. | **Critical** |
| 2 | No version, source URL, or retrieval date recorded. The file is the sole copy; upstream cannot be re-fetched or re-verified. | **Critical** |
| 3 | Feature order is recorded only in a **hand-written** `.pkl`, never in the model (`feature_names_in_` is `None`). Silent-wrong-answer risk if the list is ever edited. | **High** |
| 4 | Label encoder classes are **hand-written** (`fit(["Low","Medium","High"])`), decoupled from training. | **High** |
| 5 | Fitted `StandardScaler` was never saved; the model is unusable as shipped and `app.py` mis-scores every input. | **High** (Issue #2) |
| 6 | Notebooks read a Colab-only absolute path `/content/...`; neither runs elsewhere unmodified. | Medium |
| 7 | Training is **not a pipeline** — ~15 experiments mutate shared globals, so the saved model is an accident of execution order. | Medium |
| 8 | Shipped model is the untuned first baseline; all tuning results discarded. | Medium |
| 9 | Artifacts come from **two different environments** — model pickled with scikit-learn 1.6.1, encoder with 1.8.0, neither matching the installed 1.9.1. | Medium |
| 10 | Filename and README both claim XGBoost; artifact is a RandomForest. | Low |
| 11 | `Overall_Risk_Score` leaks the target and carries 70.3% importance. | Medium |
| 12 | Class imbalance 78.7 / 16.2 / 5.1 makes raw accuracy misleading. | Medium |

---

## 8. Files added / changed by this issue

### Added

| File | Purpose |
|---|---|
| `verify_dataset.py` | Read-only verifier. 20 checks over dataset identity, shape, quality, target, feature order, and the model contract. `--json` for CI. Exits non-zero on failure. |
| `docs/DATASET_PROVENANCE.md` | This document. |

### Deliberately NOT changed

`app.py`, `model_xgb_new.pkl`, `label_encoder.pkl`, `feature_names.pkl`,
`Cancer_Risk_Prediction_(ML).ipynb`, and the dataset itself are all untouched,
per the "investigate first" constraint. Problems 1–12 remain open.

### Recommended next (Issue #1 completion, needs approval)

| Change | Rationale |
|---|---|
| `git add cancer-risk-factors.csv` | Removes problem 1 — the only fix that prevents permanent data loss. |
| `.gitattributes` → `cancer-risk-factors.csv -text` | Pins stored bytes so the SHA-256 is stable across platforms; without it `core.autocrlf` rewrites LF→CRLF on checkout and the recorded hash breaks on every other machine. |
| Pin `requirements.txt` to exact versions | Removes problem 9. |
| Correct the README "XGBoost" claim | Removes problem 10. |
| Extract a real `train.py` pipeline that emits one bundle (model + scaler + encoder + features) | Removes problems 3, 4, 5, 7 together, by making the contract a single artifact that cannot drift. |

---

## 9. Verification commands

```powershell
cd "C:\Users\Ahmed\Desktop\cancer prediction"

# Full read-only report (expect: 17 passed, 3 warnings, 0 failed, exit 0)
python verify_dataset.py

# Machine-readable, for CI
python verify_dataset.py --json

# Confirm the dataset hash independently of the script
Get-FileHash cancer-risk-factors.csv -Algorithm SHA256

# Confirm the dataset is NOT yet protected by git
git ls-files --error-unmatch cancer-risk-factors.csv
```

`verify_dataset.py` has been tested against four scenarios:

| Scenario | Result |
|---|---|
| Pristine dataset | 17 passed, 3 warnings, 0 failed — exit 0 |
| One value altered (`Age` 68 → 69) | `dataset sha256` **FAIL** — exit 1 |
| Dataset deleted | `dataset present` **FAIL** — exit 1 |
| CRLF-only rewrite (line endings) | 0 failures, warns only — exit 0 |

The three standing warnings are the known gaps: `feature_names_in_` is `None`,
`scaler.pkl` is absent, and the artifacts were pickled under mismatched
scikit-learn versions.
