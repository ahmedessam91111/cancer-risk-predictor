# Issue registry — #7, #8, #9 (Tier 3), #10–#13 (Tier 4)

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

## Issue #12 — Check what the model is actually keying on — RESOLVED

**Definition (as filed):**
> Goal. Check what the model is actually keying on, and whether it makes
> sense. … What are the top five features by importance? Use permutation
> importance on held-out data, not just feature_importances_ — and make sure
> you know why that distinction matters. Do the top features match what
> medical literature says about cancer risk factors? Is there one feature that
> dominates everything else? If so, what are the possible explanations, and how
> would you tell them apart? Are any of your features near-duplicates of each
> other? Look at BMI and Obesity, and at Physical_Activity and
> Physical_Activity_Level. What does that do to importance scores?

**Resolution (analysis-only; no model change).** Full write-up in
`docs/ISSUE12_FEATURE_IMPORTANCE.md`; evidence script
`analyze_feature_importance.py`; figure `docs/figures/permutation_importance.png`.

* **Top 5 by held-out permutation importance (accuracy):** Alcohol_Use
  (+0.0495), Air_Pollution (+0.0470), Smoking (+0.0435), Diet_Red_Meat
  (+0.0375), Occupational_Hazards (+0.0330). Under f1-macro /
  balanced-accuracy the same cluster reorders slightly (Air_Pollution,
  Smoking, Alcohol_Use, diet, occupational); under High-recall **Obesity**
  enters the top 5. Diet_Salted_Processed swaps with Diet_Red_Meat by scorer.
* **Why the distinction matters:** `feature_importances_` is computed on
  training data during fit and is biased toward high-cardinality/continuous
  and correlated features; permutation importance on the held-out set
  measures how much real predictions need a feature. Visible proof: **BMI is
  rank 8 by impurity (0.0451) and rank ≈17 (≈0) by permutation.**
* **Medical literature:** the top cluster is behavioural/environmental
  exposures — smoking, alcohol, air pollution (IARC Group 1), processed/red
  meat, occupational hazards — all established risk factors with the correct
  direction; Fruit_Veg_Intake is the sole *negative* correlate (−0.124),
  the protective direction the literature predicts.
* **Dominance:** none. Ratios #1/#2 are 1.05–1.42 across scorers; the largest
  single feature (Air_Pollution) contributes ≤ ~19% of total held-out signal.
* **Redundant pairs — surprise:** BMI vs Obesity (Spearman **−0.003**) and
  Physical_Activity vs Physical_Activity_Level (**+0.023**) are **not**
  correlated in this dataset — independent synthetic variables. No dilution
  effect exists; `Obesity` carries the signal (perm +0.030), `BMI` and
  `Physical_Activity_Level` carry **zero**. Decision: keep both columns
  (no retrain); BMI / Physical_Activity_Level documented as drop candidates
  for any future retrain.
* **Also surprising:** Age ≈ 0 importance (a top real-world factor absent
  from this synthetic data); BRCA_Mutation / H_Pylori ≈ 0 (effects are
  conditional on `Cancer_Type`, which is excluded by design — main-effect
  measures under-report them); High-recall permutation noise is high
  (n=20 High class).

**Bottom line:** the model keys on modifiable exposure/lifestyle scores in
medically sensible directions, with no single dominant feature — i.e. a good
score for reasons that make sense *within this synthetic dataset*, which does
not encode age/genetics strongly.

**Evidence:** `analyze_feature_importance.py`, `docs/ISSUE12_FEATURE_IMPORTANCE.md`,
`docs/figures/permutation_importance.png` + `.csv`, this section.

---

## Issue #13 — Anyone who clones this repo can obtain the exact dataset — RESOLVED

**Definition (as filed):**
> Goal. Anyone who clones this repo can obtain the exact dataset your model was
> trained on. … If someone cloned this repo today, what exactly would they have
> to do to get the dataset? Try it, and write down where it breaks. Where did
> the CSV originally come from, and does its licence allow you to commit it
> here? How many rows and columns should it have? How would a future reader
> know the file they downloaded is the same one you used?

**Resolution (evidence first, then fixes):**

* **Fresh clone, tried and documented:** the dataset was already committed in
  `9a66cdd` (Issue #1) and pinned `-text` in `.gitattributes`, so a cloned
  `cancer-risk-factors.csv` is byte-identical (141,851 bytes, SHA-256
  `01291f8b…82a5`) and loads `(2000, 21)`. The only thing that broke in the
  clone: the notebooks read the Colab-only path `/content/cancer-risk-factors.csv`.
* **Origin & licence:** upstream identified as **Tarek Masryo, "Cancer Risk
  Factors & Types (2,000 Rows)"** — Kaggle `tarekmasryo/cancer-risk-factors-dataset`,
  mirrored at Hugging Face `tarekmasryo/cancer-risk-factors-data`. Licence
  **CC BY 4.0 (Attribution)** — sharing/adaptation is permitted with
  attribution, so committing the file is allowed (attribution recorded in
  `docs/DATASET_PROVENANCE.md`).
* **Byte-identity proof:** the upstream Hugging Face file was downloaded; its
  SHA-256 exactly matches the recorded hash — the committed file *is* the
  published file.
* **Identity for a future reader:** 2,000 rows × 21 columns, SHA-256
  `01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5`,
  141,851 bytes — asserted by `verify_dataset.py` and `download_dataset.py`.
* **Fixes:** notebook read cell now uses the local `cancer-risk-factors.csv`
  (no `/content/` path); new `download_dataset.py` (fetch + SHA-256 gate);
  `docs/DATASET_PROVENANCE.md` updated with confirmed origin/licence and a
  clone-to-dataframe runbook (§10); README gained a Dataset section.

**Evidence:** `download_dataset.py`, `docs/DATASET_PROVENANCE.md`, README
"Dataset" section, the edited notebook cell, this section.

---

## Issue #14 — Make the model contract explicit and enforced — RESOLVED

**Resolution (documentation + assertions; no model change).** Full contract in
`docs/MODEL_CONTRACT.md`; assertion script `verify_model_contract.py`
(PASS — 14 checks pass of 15, 1 expected WARN, exit 0), commit `6e9894c`.

* **The contract:** 17 features in one canonical order, each with its documented
  value range; `Overall_Risk_Score` excluded as leakage (#5).
* **Three agreeing witnesses,** so the order is *verified*, not assumed:
  `feature_names.pkl` (sha256 `acd88dbe…`, matching `metadata.files`),
  `bundle["feature_names"]`, and `pipeline.feature_names_in_`. The pipeline and
  scaler `feature_names_in_` agree; the forest's is `None` **by construction** —
  it is fit on a numpy array — which is expected, not a defect.
* **Enforced, not just documented:** `app.py` reorders the incoming frame to
  `FEATURE_NAMES` before predicting. Proven adversarially — a permuted CSV
  *without* the app's reorder flips ~9% of labels; *with* it, predictions are
  bit-identical; a wrong-order DataFrame raises inside the pipeline.

**Evidence:** `docs/MODEL_CONTRACT.md`, `verify_model_contract.py`, this section.

---

## Repo layout — archive the superseded artifacts, track the production set — RESOLVED

The two decisions left open at the end of the tier, resolved by the user.

* **Legacy `v1` pickles archived, not deleted** (commit `869f213`), exactly the
  option `docs/ARTIFACT_AUDIT.md` §5 prescribed: `model_xgb_new.pkl`,
  `label_encoder.pkl` and `feature_names.pkl` were `git mv`-ed to
  `archive/legacy-pickles-2026-09/` with a README recording what each is, why it
  was superseded, and its sha256. The three evidence scripts that read them
  (`verify_dataset.py`, `audit_overall_risk_score.py`, `analyze_error_costs.py`)
  were repointed at the archive path and re-verified. The serving path never
  read them.
* **`artifacts/production/` is now tracked** (commit `5beece5`) so a fresh clone
  runs the app with **no retraining**. The training bundle, `metrics.json` and
  `manifest.json` stay ignored — they are version-bound pickles / regenerable
  outputs, and `python train.py` rebuilds them bit-for-bit. `.gitignore` needed
  `artifacts/*` + `!artifacts/production/` because git cannot re-include a file
  whose parent directory is itself excluded.
* **Binary safety:** `.gitattributes` already marks `*.pkl` / `*.joblib`
  `binary`, so the `core.autocrlf=true` CRLF rewrite cannot touch them — the same
  protection the committed dataset needed. Confirmed empirically: a real clone
  reproduced all three sha256 values byte-exactly.

**Two verifier findings that came out of the clone test** (both fixed, both
honest-reporting bugs rather than model issues):

1. `verify_app_integration.py` **hard-failed on a fresh clone** because it gated
   all 27 checks on the (uncommitted) bundle, though only 2 checks use it. It now
   reports those 2 as explicit `[skip]` lines with the reason and exits 0 —
   `PASS: 25/27 checks passed, 2 skipped`. It never claims a pass it did not
   earn, and never fails for a file the project chose not to track.
2. `export_production.py --verify-only` **silently degraded to `[warn] cannot
   check content fingerprints: bundle missing`**. The bundle was never actually
   required: the fitted `StandardScaler` is embedded as pipeline step 0, so the
   scaler fingerprint is reproducible from `model.pkl` alone (verified — it
   reproduces `72a01b0b…` exactly). The fingerprints are now derived from
   `model.pkl`, and the bundle, when present, is an *independent second witness*.
   A fresh clone now verifies **all three** content fingerprints, which is
   strictly stronger than before; `manifest.json`'s absence is reported as a
   `[note]` instead of passing in silence.

**Fresh-clone proof:** cloned to a clean directory, then `export_production.py
--verify-only` exit 0 (all three content fingerprints + the three file sha256s),
`verify_app_integration.py` exit 0 (25 passed / 2 skipped) **including the live
Streamlit `AppTest` boot that produced all three risk classes**, and
`artifacts/model_bundle.joblib` absent — i.e. the app runs with no training.

---

## Issue #15 — One command that says how good the model is — RESOLVED

**Definition (as filed):** the app ships predictions but states no accuracy.
Needed one number, produced the same way every time. Measure the committed
artifact fresh (not the notebook's printed numbers), report per-class P/R/F1 and
a confusion matrix, decide whether the accuracy is plausible, and say what result
would make us suspicious rather than pleased.

**Resolution:** `evaluate.py` — loads `artifacts/production/`, recreates the
400 held-out rows, prints the classification report + confusion matrix, and
refuses to report if it cannot stand behind the measurement. No retraining, no
refitting, no threshold changes. Headline numbers recorded in the README.

* **Measured per-class (400 held-out rows, `test_size=0.2`, `random_state=42`,
  stratified):** High P 0.600 / R 0.300 / F1 0.400 (n=20); Low 0.672 / 0.662 /
  0.667 (n=65); Medium 0.890 / 0.921 / 0.905 (n=315). Accuracy 0.8475,
  balanced accuracy 0.6274, macro-F1 0.6572, weighted-F1 0.8409.
* **Most-confused pair:** Low→Medium (22) and Medium→Low (21) — the two adjacent
  ordinal classes bleeding into each other, which is the expected shape. But the
  error that *matters* is **High→Medium: 14 of 20 real High-risk patients are
  called Medium** (0 High→Low, 0 Low→High, so the ordering is never inverted).
  The model is never absurd, it is just insensitive where it counts.
* **Not a leak, and the seed is load-bearing — measured, not asserted.** The
  decisive finding: re-scoring the *same committed model* on a different
  `random_state` yields accuracy **0.965–0.975** and High recall **0.85–0.95**,
  because ~78% of the new "test" rows (312–320 of 400) were already seen while
  fitting. The true test rows are 0/400 contaminated. So **~0.97 is the number
  that should make us suspicious** — it is one typo away, and the leaked v1 model
  reached 0.9975 the same way (Issue #5). `evaluate.py` §7 prints the whole
  comparison on every run so the trap stays visible.
* **Is 0.8475 plausible for cancer risk from lifestyle factors? Not on its own.**
  Medium is 78.75% of the data, so always answering *Medium* scores **0.7875**;
  the model beats that by only +0.060 accuracy (but by +0.294 balanced accuracy,
  0.6274 vs 0.3333). So 0.8475 is ~6 points of real signal, not 85. Combined
  with High recall 0.300, the honest summary is a *useful Medium/Low triage aid
  with weak High sensitivity* on **synthetic** data that does not encode age or
  genetics (Age ≈ 0 importance, Issue #12) — not a medical device.
* **Reproducibility:** byte-identical output across separate processes and across
  `PYTHONHASHSEED` values; the held-out row set is pinned by fingerprint
  `8600c6c1d3c44bf4…`.
* **Guards that were tested by deliberately breaking them** (a preflight that
  cannot fail proves nothing): tampered CSV → FAIL on dataset sha256; tampered
  `model.pkl` → FAIL on the artifact sha256; metadata claiming `random_state=7`
  → FAIL on split-config drift, exit 1 with "refusing to print metrics for a
  measurement I cannot stand behind". The artifact-sha256 check was **added after
  testing found a flipped byte in `model.pkl` passed silently with exit 0** — the
  script was reporting metrics for a model it had not verified.
* **Self-consistency:** all 8 headline numbers are re-derived and compared to
  `artifacts/production/metadata.json`; a mismatch fails the run, so the README
  table cannot go stale unnoticed.

**Evidence:** `evaluate.py`, README "How good is it? One command" section,
`artifacts/production/metadata.json` (`metrics`), this section.

---

## Issue #16 — Which code produced `model_xgb_new.pkl`? — RESOLVED

**Answer: notebook cell index 10 (`execution_count=55`), variable `model` — the
first, untuned RandomForest in the notebook.** Not XGBoost, not a tuned model,
not the last thing that ran. Written to disk by cell index 50
(`execution_count=102`), the only cell in the notebook that writes any file.
Full write-up in `docs/MODEL_PROVENANCE.md`; evidence script
`trace_model_provenance.py` (exit 0 = positively identified).

* **Type vs filename:** `sklearn.ensemble._forest.RandomForestClassifier`, not
  XGBoost — `"XGB" in type(model).__name__` is `False`. `repr(model)` is the bare
  `RandomForestClassifier(random_state=42)`, and since scikit-learn prints only
  *non-default* params, that proves **every other hyperparameter is at its
  default** — the model was never tuned. The pickle was written by scikit-learn
  **1.6.1** (vs 1.9.1 now).
* **Hyperparameters, one by one:** 13 notebook cells construct estimators (25
  call sites, 7 of them `RandomForest`). Only cell 10 matches on all 19 params.
  Nearest rival is cell 21, whose constructor is *byte-identical* — which is why
  the input shape had to be checked too. Cells 35/38/41/42 are tuned
  (`max_depth=19`, `n_estimators=347`, …) and cannot match; cells 14/44/45/46/47
  are the wrong type entirely.
* **Input shape:** `n_features_in_ = 18`, which is the **cell 4** schema
  (`drop(['Risk_Level','Patient_ID','Cancer_Type'])` — `Overall_Risk_Score`
  retained), not cell 18's clean 17. Confirmed twice over: the model *raises* on
  17 columns, and `Overall_Risk_Score` holds **70.3%** of all importance with a
  top1/#2 ratio of **17.8×** (the shipped clean model: 1.05–1.42×).
* **Positive identification, not just elimination:** re-running notebook cells
  4 → 8 → 9 → 10 verbatim reproduces the artifact's `feature_importances_` to
  **0.000e+00**, gives **identical predictions on all 400 test rows**, and yields
  the same confusion matrix `[[19,0,1],[0,65,0],[0,0,315]]` that cell 11 itself
  printed — from a different interpreter and a different scikit-learn version.
* **The mechanism.** `model` is assigned in three cells, but only **cell 10 is at
  module scope**. Cells 35 and 47 assign `model` *inside* `def objective(trial):`,
  i.e. a **function-local** name the global can never see. So the Optuna tuning —
  including the XGBoost work the filename advertises, which was the notebook's
  last modelling at `exec 100` — never touched the global. When cell 50 ran
  `joblib.dump(model, 'model_xgb_new.pkl')` at `exec 102`, it serialised the
  `exec 55` forest. The save cell does **0 `.fit()` and 0 `.transform()` calls**
  and prints an unconditional "Model saved successfully", so the mismatch was
  invisible. Four compounding mistakes: a reused variable name, tuning hidden in
  a closure, a save cell that trusted the name, and a success message that
  cannot fail.
* **Not the best model, on either reading.** On the notebook's own leaderboard it
  *scored* highest (0.9975) — but that ranking is itself the leak, and the clean
  models were never comparable since this artifact cannot accept their 17-column
  input at all. Among leak-free models it is beaten by the tuned **SMOTE + XGBoost
  pipeline of cell 44 (accuracy 0.85, macro-F1 0.68)** and by cell 29 (0.84/0.64)
  and cell 42 (0.83/0.65). **None of the tuned models was ever saved.**
* **Not the intended deploy target.** Three signals say the intent was the
  XGBoost work and none of it reached the file: the filename says `xgb`; the save
  ran 2 execution steps after the XGB tuning, so someone meant to save *that*;
  and the `_new` suffix reads as "the improved one" when it is in fact the
  **oldest** model in the notebook.
* **The 0.9975 = 399/400**, rounded to `accuracy 1.00` in cell 11's report — the
  figure the project was long described by, and the number Issue #15 replaces
  with a real 0.8475.

**Evidence:** `trace_model_provenance.py`, `docs/MODEL_PROVENANCE.md`,
`archive/legacy-pickles-2026-09/README.md`, this section.

---

## Issue #17 — Is `Overall_Risk_Score` a legitimate input? — RESOLVED

**Verdict: NO — target leakage.** `Risk_Level` is a deterministic function of
`Overall_Risk_Score`, and `Overall_Risk_Score` is itself computed from the 17
legitimate risk factors. It is the *middle* of the chain
`factors → Overall_Risk_Score → Risk_Level`, handed to a model whose job is to
reconstruct the last link. Full write-up in `docs/OVERALL_RISK_SCORE_VERDICT.md`;
evidence script `measure_risk_score_leakage.py`.

* **Computed, not measured.** `float64`, bounded `[0.029285, 0.852158]`, and
  **2000 distinct values for 2000 rows** — a unit-normalised composite, not an
  observation. `R² = 0.838764` regressed on the other 17 columns, with nine
  risk-factor columns carrying near-equal weight (`Family_History` +0.0150 …
  `Occupational_Hazards` +0.0122) and eight near zero. The unexplained 16% is
  **independent noise**, not a missed term: its largest |Spearman| against any
  input is **0.0103**, skew −0.0095, excess kurtosis −0.1111.
* **Not available at prediction time.** It exists only after the same risk
  factors have already been combined and thresholded — i.e. after the answer is
  known. A model that needs it cannot run on a new patient; a model that is
  handed it is re-reading the label.
* **Measured relationship.** Spearman `ρ = 0.7129` (capped by the label's 3
  levels with 1574/2000 rows tied), but **703572/703572 cross-class pairs are
  ordered correctly (1.000000)** — the ordering is total. Mutual information
  `0.633875` nats against a label entropy of `0.635146` → **normalised MI
  0.9980**, i.e. the column removes 99.8% of the label's uncertainty. Its MI
  exceeds the *sum* of the 17 legitimate features' individual MIs (`0.394426`).
  Grouped: Low `[0.029285, 0.329922]`, Medium `[0.330033, 0.659130]`, High
  `[0.660797, 0.852158]` — disjoint, constraining the cuts to `(0.3299, 0.3300)`
  and `(0.6591, 0.6608)` (i.e. **0.33** and **0.66**). A rule reading only this
  column is **2000/2000 = 100.00%**, macro-F1 **1.000000**, versus the deployed
  model's 0.6572.
* **The notebook saw it and dropped it.** Cell 12 celebrates "only 1 mistake out
  of 400"; cell 17 is titled **"Removing Overall_Risk_Score and retrying"**; cell
  18 drops the column; everything after is leak-free. The judgement was right —
  it just never reached the artifact, because of the save-cell bug of Issue #16.
* **The deployed model reflects the decision.** Verified mechanically:
  `artifacts/production/feature_names.pkl` = **17** features, no
  `Overall_Risk_Score`; `metadata.json` records
  `excluded_features = {'Overall_Risk_Score': 'target leakage, Issue #5'}`;
  `app.py` does not ask for it.
* **The old slider.** The v1 app offered a `0.00–1.00` slider (default `0.5`) in
  a group titled **"📊 Engineered score"**. Holding all 17 other features at
  their medians and moving only the slider, the archived v1 model flips class at
  exactly 0.33 and 0.67: 0.32 → Low, 0.33 → Medium, 0.66 → Medium, 0.67 → High.
  The user was not describing the patient; they were **selecting the diagnosis**.
  It was removed in `015368e`, with the reason documented at `app.py:215`, and
  reintroduction is guarded by the 17-feature contract (`verify_model_contract.py`,
  Issue #14).

**Evidence:** `measure_risk_score_leakage.py`, `docs/OVERALL_RISK_SCORE_VERDICT.md`,
`audit_overall_risk_score.py` (Issues #5/#6), this section.

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
| 12 | Check what the model is actually keying on | RESOLVED — permutation-importance audit | `docs/ISSUE12_FEATURE_IMPORTANCE.md` |
| 13 | Anyone who clones this repo can obtain the exact dataset | RESOLVED — committed + upstream-confirmed (CC BY 4.0) | `docs/DATASET_PROVENANCE.md`, `download_dataset.py` |
| 14 | Make the model contract explicit and enforced | RESOLVED — verified (3 witnesses) + enforced by the app | `docs/MODEL_CONTRACT.md`, `verify_model_contract.py` |
| 15 | One command that says how good the model is | RESOLVED — `evaluate.py`, contamination guard included | README "How good is it?", `evaluate.py` |
| 16 | Which piece of code produced `model_xgb_new.pkl` | RESOLVED - cell 10, `model`; proven by bit-identical refit | `docs/MODEL_PROVENANCE.md`, `trace_model_provenance.py` |
| 17 | Is `Overall_Risk_Score` a legitimate input | RESOLVED - NO, target leakage; deployed model excludes it | `docs/OVERALL_RISK_SCORE_VERDICT.md`, `measure_risk_score_leakage.py` |
| — | Repo layout: archive superseded artifacts, track the production set | RESOLVED — archived + `artifacts/production/` tracked | `archive/legacy-pickles-2026-09/README.md`, `docs/ARTIFACT_AUDIT.md` §5 |