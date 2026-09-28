# Issue registry — #7, #8, #9 (Tier 3) and #10 (Tier 4)

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
| 11–13 | (not yet specified) | OPEN | awaiting definitions |