# Tier 3 — Reproducible training, the shipped model, and the resampling audit

Covers GitHub Issues **#7**, **#8**, **#9**.

Everything below was produced by running the scripts in this repository. No
number in this document was typed by hand.

---

# Issue #7 — Convert the notebook into a reproducible `train.py`

## 1.1 The training workflow, as traced from the notebook

`Cancer_Risk_Prediction_(ML).ipynb` contains 51 code cells. The chain that
actually produced the deployed model was identified in Tier 2 (Issue #4) as
**cell 10 (execution 55)**, saved to disk by **cell 50 (execution 102)**:

| Cell | Step | Code |
|---|---|---|
| 2 | load | `df = pd.read_csv('/content/cancer-risk-factors.csv')` |
| 4 | features / target | `X = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type'])`, `y = LabelEncoder().fit_transform(df['Risk_Level'])` |
| 8 | split | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` |
| 9 | scale | `StandardScaler().fit_transform(X_train)` |
| 10 | fit | `RandomForestClassifier(random_state=42, n_estimators=100)` |
| 50 | save | `joblib.dump(model, 'model_xgb_new.pkl')` |

`model` has exactly **one** top-level binding in the whole notebook (verified by
AST scope analysis). Cells 35 and 47 rebind the name `model` only inside the
body of `objective()`, so they cannot affect what cell 50 wrote.

Everything after cell 10 — the Optuna study, `final_rf(n_estimators=347,
max_depth=19)`, and all XGBoost work — was **discarded by the save in cell 50**
and is deliberately not reproduced. `train.py` therefore keeps
`n_estimators=100`, not 347.

## 1.2 What `train.py` changes, and why

Nothing changed silently. Each difference traces to a Tier 1 or Tier 2 finding.

| # | Change | Justification | Issue |
|---|---|---|---|
| 1 | Drop `Overall_Risk_Score` from the feature set | `Risk_Level` is a 100% accurate threshold function of it (thresholds `0.329978` / `0.659964`, 2000 rows, **0 mismatches**). The column *is* the target, quantized. | #5 |
| 2 | `class_weight="balanced"` | Target is 78.7% Medium / 16.2% Low / 5.1% High. Unweighted, `High` recall was **0.050** — 1 of 20 High patients found. | #6, #9 |
| 3 | **Persist** the fitted `StandardScaler` | The notebook fit a scaler and never saved it, so `app.py` fed raw values to a model fit on standardized values. Trees are not scale-invariant across a train/serve mismatch. | #1, #2 |
| 4 | Derive the feature list from the CSV, don't hand-write it | `model.feature_names_in_` was `None` (the model was fit on a bare numpy array), so column order survived only in a `.pkl` that `import joblib.py` reconstructed by hand months later. | #1, #2 |
| 5 | Ship model + scaler + encoder + feature order as **one** bundle | Three loose files let the input contract drift apart from the model. | #2 |
| 6 | Write metrics, environment, and checksums | Reproducibility. | #7 |

**Deliberately not done:** no hyperparameter search. The notebook's Optuna
result cannot be reused — its objective was computed on the leaked feature
(Issue #5) *and*, in cells 35 and 47, on cross-validation folds drawn from
already-resampled data (Issue #9, worth **+0.2775** macro-F1 of pure
inflation). Re-tuning on the clean feature set is separate, deliberate work.

**Also deliberately not done:** no change to the dataset, the target, the
`LabelEncoder`, the split (`test_size=0.2`, `random_state=42`, `stratify=y`),
the scaler choice, or the estimator family. Those are the same as cell 10.

## 1.3 Validation the script performs before fitting

`train.py` refuses to train and exits non-zero on any of these:

- dataset file missing
- SHA-256 mismatch against `01291f8b…b82a5` (checked EOL-normalized so Windows
  `autocrlf` cannot cause a false failure, while a real edit still fails)
- shape other than 2000 × 21
- any missing value or duplicate row
- target classes other than exactly `High`, `Low`, `Medium`
- a feature count other than 17
- `Overall_Risk_Score` reappearing in the derived feature list

---

# Issue #8 — Retrain and ship the model you actually want to serve

## 2.1 Final training configuration

| Parameter | Value | Source |
|---|---|---|
| Dataset | `cancer-risk-factors.csv`, sha256 `01291f8b…b82a5`, 2000 × 21 | Issue #1 |
| Target | `Risk_Level` → `LabelEncoder` → `High=0, Low=1, Medium=2` | unchanged |
| Features | **17** — all columns except `Patient_ID`, `Cancer_Type`, `Risk_Level`, `Overall_Risk_Score` | **changed**, Issue #5 |
| Split | `test_size=0.2`, `random_state=42`, `stratify=y` → 1600 train / 400 test | unchanged |
| Scaler | `StandardScaler`, fit on the **train split only** | unchanged, now persisted |
| Estimator | `RandomForestClassifier(n_estimators=100, max_depth=None, criterion="gini", max_features="sqrt", class_weight="balanced", random_state=42)` | `class_weight` added, Issue #6/#9 |
| Model version | `2.0.0` | new |
| Trained at | 2026-09-28, Python 3.14.7 / scikit-learn 1.9.1 / numpy 2.5.3 / pandas 3.0.6 | recorded in the bundle |

Feature order (this *is* the input contract):

```
Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History, Diet_Red_Meat,
Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity, Air_Pollution,
Occupational_Hazards, BRCA_Mutation, H_Pylori_Infection, Calcium_Intake,
BMI, Physical_Activity_Level
```

## 2.2 Why `Overall_Risk_Score` was removed rather than retained

Not on assumption — on the Tier 2 measurement:

- `Risk_Level` is recovered from `Overall_Risk_Score` alone with **zero**
  errors across all 2000 rows.
- The column is itself ~83.9% linearly reconstructible from the other 17
  features, so it is a composite, not an independent measurement.
- Issue #6 measured the cost of keeping it: the single column scored the **same
  0.9975** as all 18 features together, and the other 17 contributed
  **+0.0000**.

A model handed the answer will always look excellent and learn nothing.

## 2.3 Evaluation results — held-out test set (400 rows, never resampled)

```
              precision    recall  f1-score   support

        High       0.60      0.30      0.40        20
         Low       0.67      0.66      0.67        65
      Medium       0.89      0.92      0.90       315

    accuracy                           0.85       400
   macro avg       0.72      0.63      0.66       400
weighted avg       0.84      0.85      0.84       400
```

Confusion matrix (rows = actual `High, Low, Medium`; columns = predicted):

```
[[  6,   0,  14],     # 6 of 20 High caught
 [  0,  43,  22],     # 43 of 65 Low caught
 [  4,  21, 290]]     # 290 of 315 Medium caught
```

3-fold CV macro-F1 on the training split: **0.6657 ± 0.0342**
(folds 0.7031 / 0.6734 / 0.6205).

**These are the numbers to judge the model on, and they are modest.** That is
the honest result. The previously published 0.9975 was an artifact of leakage,
not performance.

## 2.4 The metric choice, and why accuracy is not the headline

The target is 78.7% Medium. A model that answers *Medium* for everybody scores
**0.79 accuracy** while detecting zero High-risk patients. Accuracy is therefore
the wrong summary statistic here, and **macro-F1** is reported as the headline
because it weights all three classes equally.

Balancing raised `High` recall from **0.050 → 0.300** (1 → 6 of 20 patients).
That is the single most important improvement in this retrain, and it came from
`class_weight="balanced"`, not from tuning.

## 2.5 Streamlit integration changes

`app.py` now reads the bundle and nothing else.

| Change | Reason |
|---|---|
| Loads `artifacts/model_bundle.joblib` instead of `model_xgb_new.pkl` + `label_encoder.pkl` + `feature_names.pkl` | one file cannot drift |
| `preprocess_input()` ends with `scaler.transform(df)` | **Issue #2** — the root cause of the app returning `Medium` 400/400 |
| `Overall_Risk_Score` slider deleted from the manual form | **Issue #5** — it made the form self-answering |
| Batch mode now echoes the user's own columns, not the scaled internal matrix | the old code wrote `X.copy()`, which is now a scaled array |
| Reports unknown extra CSV columns instead of silently dropping them | contract transparency |
| Sidebar model card: version, estimator, held-out metrics, and why the old score was higher | **Issue #8 step 9** |

`model_xgb_new.pkl`, `label_encoder.pkl`, and `feature_names.pkl` are **left on
disk, untouched, no longer read by anything.** They remain in git for
provenance. Removing them is a separate decision — see the open questions at
the end.

## 2.6 Before / after, end to end

| | Before (v1, shipped) | After (v2.0.0) |
|---|---|---|
| Features | 18, incl. `Overall_Risk_Score` | 17, leak-free |
| Held-out F1 macro | 0.9975 *(meaningless — leak)* | 0.6572 *(honest)* |
| `High` recall | 0.050 (leaky features) | 0.300 (leak-free) |
| App output on 400 held-out rows | `Medium` 400/400, `High` 0 | `Medium` 326, `Low` 64, `High` 10 |
| Scaler persisted | no | yes |
| Input contract | 3 loose pickles, order hand-written | 1 bundle, asserted |

---

# Issue #9 — Get resampling right

## 3.1 Current resampling workflow, cell by cell

SMOTE appears in both notebooks. The two notebooks contain **both** a correct
and an incorrect pattern.

### Leakage-free (correct)

| Location | Pattern |
|---|---|
| NB1 cell 27, NB2 cells 46/55 | `SMOTE().fit_resample(X_train, y_train)` — **after** the split, on train only. Correct for a single holdout. |
| NB1 cells 41/42/44/45, NB2 cells 67/68/70-72 | `ImbPipeline([('smote', SMOTE(...)), ('rf', rf)])` with `cross_val_score` / `pipe.fit(X_train, y_train)`. Correct: SMOTE is refitted inside every fold. |
| NB1 cell 47 | `class_weight` recomputed from `y_t` inside the fold loop. Correct. |

### Leaking (incorrect)

| Location | Pattern |
|---|---|
| **NB1 cell 35** | `cv.split(X_train_res, y_train_res)` then `model.fit(X_t, y_t)` |
| **NB2 cell 62** | identical |

`X_train_res` was produced by resampling the **entire** training split *before*
cross-validation. Each "validation" fold therefore contains synthetic samples
that were interpolated from rows belonging to that same fold. Validation scores
are optimistic; anything selected on them is selected on noise.

**The deployed model used no resampling at all.** It came from cell 10, before
any SMOTE existed in the pipeline.

## 3.2 Problem: the inflation, measured

Same estimator, same split, same features — only the resampling *placement*
differs:

| Pattern | 3-fold CV macro-F1 |
|---|---|
| No resampling | 0.4901 |
| **Leaky**: SMOTE → then CV over resampled data (**NB1:35 / NB2:62**) | **0.9549** |
| **Correct**: SMOTE inside an `ImbPipeline`, refitted per fold | **0.6774** |
| `class_weight="balanced"` | 0.6597 |

```
leakage inflation  =  0.9549 − 0.6774  =  +0.2775
```

**+0.2775 macro-F1 is not performance — it is the model grading its own
homework.** Roughly 28 points of the score came from validation rows having
contaminated their own training folds.

This matters beyond the notebook. The Optuna study in cells 35-37 optimized
`objective()` against exactly this number, and the hyperparameters it selected
(`final_rf`, `n_estimators=347`) were chosen on it.

## 3.3 Problem: resampling library was not even a dependency

`imbalanced-learn` was **absent** from `requirements.txt` and **not installed**
in this environment. Every SMOTE cell in both notebooks was unrunnable as
shipped. A reader reproducing the notebook would have hit `ModuleNotFoundError`
at cell 27 and had no way to know the resampling was even supposed to be there.

## 3.4 Corrected workflow — test set never touched

`train.py` enforces the invariant structurally rather than by convention:

```python
# 1. split FIRST -- the test set is never resampled, ever
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# 2. scale on train only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)          # transform, never fit

# 3. resampling lives INSIDE the estimator, so cross_val_score refits it per fold
rf = RandomForestClassifier(..., class_weight="balanced", random_state=42)
model = rf                                      # shipped arm
# model = ImbPipeline([("smote", SMOTE(random_state=42)), ("rf", rf)])   # --resample smote
```

Because the pipeline object is what gets handed to `cross_val_score`, imblearn
refits SMOTE on each fold's training portion. There is no code path in
`train.py` that can resample a validation or test row — resampling is not a
separate step that could be misplaced, it is a step of `fit`.

`--resample smote` exists so the corrected pattern is executable and
auditable, not just described.

## 3.5 Before vs after, on the shipped configuration

Both arms run through the real `train.py` on the held-out 400 rows:

| Metric | `class_weight` **(shipped)** | `smote` (comparison) | Δ |
|---|---|---|---|
| Accuracy | 0.8475 | 0.8600 | +0.0125 |
| Precision macro | 0.7205 | 0.7790 | +0.0585 |
| Recall macro | **0.6274** | 0.6049 | −0.0225 |
| **F1 macro** | **0.6572** | 0.6523 | −0.0049 |
| F1 weighted | 0.8409 | 0.8478 | +0.0069 |
| CV F1 macro | 0.6657 ± 0.0342 | 0.6883 ± 0.0193 | +0.0226 |
| `High` precision / recall / F1 | 0.600 / **0.300** / **0.400** | 0.714 / 0.250 / 0.370 | — |

**The two arms are statistically indistinguishable.** Held-out macro-F1 differs
by 0.0049, on a test set containing 20 `High` patients — one flipped prediction
moves `High` recall by 0.05 and macro-F1 by ~0.04. The CV/test disagreement in
sign is fold noise, not signal.

### Why `class_weight` is shipped

1. It wins the agreed headline metric (macro-F1) on held-out data, and has the
   higher `High` recall — the clinically important class.
2. **No inference-time dependency.** A plain scikit-learn estimator keeps
   `imbalanced-learn` out of the serving path entirely.
3. **No synthetic rows.** Nothing in the model was ever fitted on an
   interpolated patient.

I checked the usual objection to SMOTE here and it did **not** hold: SMOTE
produced **zero** impossible values. It created 0 non-`{0,1}` values in the four
binary features and 0 non-integers in the eleven 0-10 ordinal features, because
k-nearest neighbours in those low-cardinality columns share values. The
fractional cells it did produce (5.6% of the matrix) are confined to `Age` and
`BMI`, which are continuous and legitimately fractional. I had expected this
argument to favour `class_weight`; it turned out to be neutral, and the decision
rests on points 1-3 instead.

## 3.6 Verification that the test set remains untouched

`verify_app_integration.py` and `train.py --verify` both assert the train/test
index sets are disjoint, and every metric in §2.3 comes from `X_test_s`, which
is produced by `scaler.transform` and never by any resampler. `train.py`'s
`cross_validate` only ever receives `X_train_s, y_train`.

---

# Verification steps

```powershell
pip install -r requirements.txt

# 1. dataset integrity, provenance, and the Tier 1/2 audits
python verify_dataset.py
python audit_overall_risk_score.py

# 2. reproduce the model  ->  trains, evaluates, saves, then self-verifies
python train.py

# 3. re-check a saved bundle without retraining
python train.py --verify

# 4. confirm app.py is wired to exactly these artifacts (20 checks)
python verify_app_integration.py

# 5. serve
streamlit run app.py
```

**Reproducibility is bitwise, and was measured.** Across two consecutive runs,
the fitted forest hashed identically
(`sha256(feature|threshold|children_left|children_right)` over all 100 trees =
`aeca50e59b3dc670177a29e05a0e6c84`), and every metric matched to 4 decimals. The
canonical fingerprint function is `forest_sha256()` in `train.py`; it is recorded
in the bundle and manifest, and `train.py --verify` asserts it automatically. The
`model_bundle.joblib` **file hash** does differ between runs, solely because the
bundle embeds a `trained_at` UTC timestamp; `manifest.json` records that hash
per run.

`verify_app_integration.py` goes further than reading the source. It has five
layers:

1. **Static** — parses `app.py` with `ast` and asserts what it loads, what it
   scales, and what it never mentions.
2. **Behavioural** — extracts `app.py`'s real `preprocess_input` via `ast` and
   executes it, confirming the matrix it produces is **bit-identical**
   (sha256 match, max abs diff `0.00e+00`) to `scaler.transform(...)` even when
   fed reversed column order, shuffled rows, and decoy columns including a fake
   `Overall_Risk_Score`.
3. **End to end** — runs the 400 held-out rows through the app's code path and
   asserts all three classes appear.
4. **Purity** — asserts train/test index sets are disjoint.
5. **Live Streamlit** — boots the real `app.py` through Streamlit's own
   `AppTest` runner, switches to manual mode, and clicks the predict button for
   three synthetic patients.

The live check is the one that matters most for §2.5, because it exercises the
widget layer that static analysis cannot reach:

| Profile | Predicted | Old app |
|---|---|---|
| age 80, BMI 45, all factors 10, all three flags on, male | **High** | Medium |
| age 25, BMI 18, all factors 0, no flags, female | **Low** | Medium |
| age 50, BMI 25, all factors 5, no flags, female | **Medium** | Medium |

The old app returned `Medium` for all three, and for all 400 held-out rows.

---

# Conclusions

### Issue #7 — resolved

The notebook's real training chain was traced to cells 2/4/8/9/10 saved by cell
50, and converted to `train.py`. Another developer reproduces the published
model with `python train.py`. Six deviations from the notebook are documented
above, each traced to a Tier 1/2 issue. The dataset is checksum-gated, so a
silent data change fails the run instead of quietly changing the model.

### Issue #8 — resolved

The model that is actually wanted to serve is now trained, shipped as a single
bundle with its scaler, evaluated, recorded, and consumed by `app.py` with a
machine-checked proof of integration. `Overall_Risk_Score` was removed on the
strength of the Issue #5/#6 measurement, not on assumption.

**The honest number is macro-F1 0.6572, not 0.9975.** The gap is the leakage,
and closing it is the point of the exercise. `High` recall of 0.300 — 6 of 20 —
is usable but weak, and with only 102 `High` patients in the dataset (51 in the
training split) that class is data-limited, not model-limited. Better `High`
recall is the obvious next piece of work, and it requires more data rather than
more tuning.

### Issue #9 — resolved

Resampling is audited, the leak is measured at **+0.2775 macro-F1**, the correct
pattern is implemented in `train.py` and executable via `--resample smote`, and
the before/after comparison on the shipped configuration is reported in §3.5.
The test set is never resampled, by construction rather than by care.

The one thing this issue cannot fix retroactively: the Optuna result in
notebook cells 35-37 is not recoverable. Its objective function is a leaked
metric, so re-running it would only reproduce the same fiction.

---

# Open questions for the maintainer

1. **Delete the superseded artifacts?** `model_xgb_new.pkl`, `label_encoder.pkl`,
   and `feature_names.pkl` are no longer read by any code path but are still
   tracked in git and on disk. Keeping them preserves the before/after
   evidence; deleting them prevents anyone from accidentally reloading a broken
   model. Currently left untouched, deliberately.
2. **Keep the notebooks in the repository?** They remain the historical record
   of the leakage, but they are not runnable as written (`imbalanced-learn` was
   not a dependency, Issue #3.3). `train.py` is now the supported path.
3. **The nested `cancer-risk-predictor/` clone** is now listed in `.gitignore`
   so it stops showing up as untracked noise. Nothing was deleted — say the
   word if you want it removed or tracked instead.
