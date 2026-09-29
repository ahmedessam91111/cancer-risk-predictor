# Issue registry — #7, #8, #9 (Tier 3), #10 and #11 (Tier 4)

This file records the GitHub issue definitions and their resolution evidence.
Issues #1–#6 are documented in `DATASET_PROVENANCE.md`,`T2_PROVENANCE_AND_LEAKAGE.md`
and `TRAINING_AND_LEAKAGE.md`.

Status legend: **RESOLVED** = implementation verified and merged to `master`.

---

## Issue #7 — Convert notebook into reproducible `train.py` — RESOLVED

**Definition (as filed):**
> Convert the essential training workflow into a reproducible script.
> Preserve the intended dataset, preprocessing, features, target, split, and
> model unless an audit finding requires a change. Make parameters
> reproducible. Save artifacts consistently.

**Resolution:** `train.py` (640+ lines) re-implements the notebook's training
cells (2 / 4 / 8 / 9 / 10) and the save cell (50) as a checkable pipeline. It
records 6 explicit, documented deviations from the notebook, each required by a
prior audit finding (leaked feature #5, missing scaler #2, resampling placement
#9, resampling strategy #9, environment pinning). It is checksum-gated on the
dataset, pins every parameter as a named constant, saves one atomic bundle, and
self-verifies after saving.

**Evidence:** `train.py`, `docs/TRAINING_AND_LEAKAGE.md` (deviations table),
commit `015368e`. Metrics reproduced bit-for-bit across two machines.

---

## Issue #9 — Get resampling right — RESOLVED

**Definition (as filed):**
> Verify whether SMOTE/resampling is used. Ensure train/test splitting happens
> BEFORE resampling. Ensure the test set is never resampled. If cross-validation
> is used, put resampling inside the CV pipeline. Document and fix any leakage.

**Resolution:** Measured and fixed.

* **Measured leakage:** notebook cell 35 (NB1) / cell 62 (NB2) resample the
  whole dataset before the split. Resampling before CV inflated CV macro-F1 to
  **0.9549** vs 0.6774 for the correct workflow — **+0.2775 of pure leakage**.
  The test set never sees resampling in the shipped pipeline.
* **Correct workflow:** `train.py` splits first (stratified, `random_state=42`),
  and imbalance handling is **inside** the classifier (`class_weight="balanced"`)
  rather than by row synthesis. `--resample smote` is supported and correct
  (resampling fit inside the CV folds, test untouched), but the shipped model
  uses `class_weight` — statistically indistinguishable macro-F1 (0.6572 vs
  0.6523) with no `imbalanced-learn` runtime dependency and no synthetic rows.
* **Honest correction in the record:** an initial hypothesis that SMOTE creates
  impossible fractional binary/ordinal values was measured and found **wrong** —
  0 bad values; the SMOTE path is correct as implemented.

**Evidence:** `docs/TRAINING_AND_LEAKAGE.md` (resampling section), commit
`015368e`.

---

## Issue #8 — Retrain and ship the model actually intended to be served — RESOLVED

**Definition (as filed):**
> Use the findings from Issues #1–#7. Train the final model from the correct
> dataset and features. Apply the corrected preprocessing and resampling
> workflow. Save the final production artifacts. Verify Streamlit uses exactly
> those artifacts. Record model version, features, preprocessing, and training
> configuration.

**Resolution:**

* Final model trained from the leak-free 17 features, `class_weight="balanced"`,
  stratified 80/20 with `random_state=42`, dataset checksum-verified. Published
  metrics (400 held-out): accuracy 0.8475, F1 macro 0.6572, High recall 0.300,
  High precision 0.600.
* **Production artifacts** (the "save the final production artifacts" step, now
  in the layout requested): `artifacts/production/{model,label_encoder,
  feature_names,metadata}.{pkl,json}`, produced by `export_production.py` from
  the verified bundle. `model.pkl` is a `Pipeline [StandardScaler ->
  RandomForestClassifier]` so no caller can feed unscaled input (Issue #2).
* **Verified Streamlit uses exactly those artifacts:** `verify_app_integration.py`
  — 27/27 checks across static/AST, behavioural (real `preprocess_input`),
  end-to-end bitwise equality with the bundle, purity, and a live
  `AppTest` boot of the app. The app runs without retraining.
* **Recorded:** `metadata.json` + `manifest.json` + `metrics.json` carry model
  version (2.0.0), feature list, config, environment (with
  `environment_matches_reference`), dataset sha256, forest fingerprint, and
  metrics including the Issue #10 primary metric (`high_recall`,
  `high_precision`).

**Evidence:** `train.py`, `export_production.py`, `verify_app_integration.py`,
`docs/ARTIFACT_AUDIT.md`, `docs/TRAINING_AND_LEAKAGE.md`, commits `015368e`,
`10107d7`, and the production-artifacts commit.

---

## Issue #11 — Check whether the displayed probabilities are real — RESOLVED (hybrid shipped)

**Decision (user-approved):** option **A — hybrid** — recalibrate the
displayed/exported probabilities while class labels stay on the raw forest's
argmax, so every Issue #10 metric remains byte-for-byte valid.

**Definition (as filed):**
> Check whether the displayed probabilities are real. Find where
> `predict_proba()` is used. Check how probabilities are displayed in the
> Streamlit app. Verify whether the probabilities are calibrated. Evaluate
> calibration using an appropriate calibration curve and Brier score. Check
> whether a displayed probability such as 80% actually corresponds to
> approximately 80% observed frequency. If probabilities are poorly calibrated,
> explain the correct calibration approach. Do not change the model unless
> calibration is actually required.

**Measurement (analysis only — nothing retrained, nothing deployed):**
`analyze_probability_calibration.py` scores the deployed production pipeline on
the exact 400 held-out rows.

* **Where probabilities come from / how they are shown:** `app.py` line 185
  (batch) and line 281 (manual) call `predict_proba()`; manual mode renders each
  class as a bold percentage + bar (`render_probabilities`, lines 152–163),
  batch mode exports `prob_<class>` columns to the downloadable CSV. The values
  are `RandomForestClassifier(class_weight='balanced')` **vote fractions** — a
  confidence proxy, not calibrated frequencies.
* **Brier score:** multiclass **0.2418** vs climatology 0.3509 → the model is
  informative (+31%) but far from perfect. Per class: High 0.0367, Low 0.0799,
  Medium 0.1253.
* **Calibration (predicted vs observed frequency):** ECE macro **0.0912**
  (threshold 0.05) → materially miscalibrated. Per class: High 0.0418
  (noisy, n=20), Low **0.0943**, Medium **0.1376**.
  * Medium is systematically **under**-confident mid/high range: 0.56 shown →
    0.74 observed; 0.79 shown → 0.91–1.00 observed.
  * Low is **over**-confident: 0.20 shown → 0.04 observed; 0.35 shown → 0.23.
  * High is over-confident at low probabilities (0.11 shown → 0.03; 0.19 → 0.11),
    roughly right near 0.41 (→ 0.34).
* **"Does 80% mean ~80%?":** roughly only at the top end — Medium bin mean 0.79
  → observed 0.91–1.00; Low bin mean 0.83 → observed 0.90. Mid-range fails the
  check badly (Low 0.20 → 0.04; Medium 0.60 → 0.76). Displayed percentages are
  **not** reliable as frequencies.
* **What proper recalibration would achieve:** per-class isotonic fitted on
  5-fold out-of-fold predictions of the same forest (fit on train, applied to
  the untouched 400 rows): Brier 0.2418 → **0.2118**, ECE 0.0912 → **0.0203**.

**Problems found**

1. The shipped RF probabilities are materially miscalibrated (ECE 0.091 ≥ 0.05).
2. The app presents them as plain percentages/CSV columns with no caveat, so a
   user can mistake vote fractions for observed frequencies.
3. Worst error is not a sample-size artefact for the majority class: Medium has
   315 held-out rows and is under-confident by up to 21 points (bin 0.79 → 1.00).

**Recommended solution (correct calibration approach)**

If calibration is adopted: fit per-class one-vs-rest calibrators (isotonic, or
Platt/sigmoid for the 82-row High class, which is too sparse for stable isotonic)
on **out-of-fold predictions of the same forest** — never on the held-out set —
then renormalize each row's probabilities to sum to 1 (per-class maps do not
preserve the simplex), and re-verify Brier + ECE on the untouched held-out rows.
Equivalently, `CalibratedClassifierCV`.

**Required code changes — shipped (decision A, hybrid, user-approved)**

* `calibrated_model.py` (new): `CalibratedModel(BaseEstimator)` wrapper —
  `predict()` defers to the raw forest (labels unchanged), `predict_proba()`
  applies per-class isotonic recalibration and renormalises rows to 1. Kept in
  its own importable module so `joblib.load` works from any caller.
* `export_production.py`: calibration is now the **default** export; the
  calibrators are fitted on 5-fold OOF probabilities of the *same* forest
  (clone, identical hyperparameters, fixed scaler, seed 0) on the training
  split only. `--no-calibrate` exports the plain uncalibrated pipeline.
  `metadata.json` records `calibration` (method, fit protocol, array
  fingerprint) and `calibration_holdout` (before/after Brier + ECE measured on
  the untouched 400 held-out rows); `manifest.json` records the calibration
  fingerprint. `--verify-only` checks the calibration fingerprint too.
* `app.py`: one caption under the probability bars explains the hybrid
  (calibrated percentages, uncalibrated-model label).
* **Verified:** production contract PASS; `verify_app_integration.py` 27/27
  incl. live Streamlit boot; `--verify-only` exit 0. Held-out through the
  shipped hybrid: labels bit-identical (accuracy 0.8475; distribution
  `{'Medium': 326, 'Low': 64, 'High': 10}`; every #10 metric unchanged),
  Brier 0.2418 → 0.2118, ECE macro 0.0912 → 0.0203. Calibration fingerprint
  `f589f3b8f01a795c…` is deterministic across processes.

**Final conclusion:** the displayed probabilities were **not** real as frequencies —
informative (31% better than the base rate) but materially miscalibrated. The
hybrid fix (decision A) ships honest calibrated probabilities while preserving
every published decision metric; end-to-end recalibration (option B) was
rejected because it dropped High recall 0.30 → 0.20 (conflicts with #10).

**Evidence:** `analyze_probability_calibration.py` (analysis-only), `calibrated_model.py`,
`export_production.py` (default calibrated export), `app.py` caption, this section.

---

## Issue #10 — Choose a metric that matches the cost of being wrong — RESOLVED

**Resolution:** primary metric = **High-class recall**, reported with
**High-class precision** as its guardrail; macro-F1 and balanced accuracy
tracked as secondary; accuracy reported for context only. Analysis-only (no
retraining, no tuning). Full decision record in `docs/METRIC_SELECTION.md`,
evidence script `analyze_error_costs.py`, commit `f2ee071`.

---

## Issue tracker (all known issues)

| # | title | status | where |
|---|-------|--------|-------|
| 1 | Dataset provenance & integrity | RESOLVED | `docs/DATASET_PROVENANCE.md` |
| 2 | App uses loose `.pkl` artifacts (scaler never saved) | RESOLVED | `docs/T2_PROVENANCE_AND_LEAKAGE.md`, `docs/TRAINING_AND_LEAKAGE.md` |
| 3 | App returned *Medium* for every patient | RESOLVED | same |
| 4 | Untuned baseline shipped as the model | RESOLVED | same |
| 5 | `Overall_Risk_Score` target leakage | RESOLVED | `audit_overall_risk_score.py`, same docs |
| 6 | Evaluation used resampled test rows | RESOLVED | same |
| 7 | Convert notebook into reproducible `train.py` | RESOLVED | this file |
| 8 | Retrain & ship the model actually intended to be served | RESOLVED | this file |
| 9 | Get resampling right | RESOLVED | this file |
| 10 | Choose a metric matching the cost of being wrong | RESOLVED | `docs/METRIC_SELECTION.md` |
| 11 | Check whether the displayed probabilities are real | RESOLVED — hybrid calibration shipped | this file |
| 12–13 | (not yet specified) | OPEN | awaiting definitions |