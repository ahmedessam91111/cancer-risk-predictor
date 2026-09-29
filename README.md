# 🧬 Cancer Risk Predictor

A 3-class ML web app (Streamlit) that predicts **Cancer Risk Level** — *Low / Medium / High* — from patient lifestyle and genetic factors.

## 📊 How good is it? One command

```bash
python evaluate.py
```

Measured on the 400 held-out rows the shipped model was never fitted on
(`test_size=0.2, random_state=42, stratified` — imported from `train.py`, never
retyped). Not quoted from the notebook; recomputed from the committed artifacts
every run.

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|----:|--------:|
| **High** | 0.600 | **0.300** | 0.400 | 20 |
| **Low** | 0.672 | 0.662 | 0.667 | 65 |
| **Medium** | 0.890 | 0.921 | 0.905 | 315 |

| Metric | Value |
|--------|------:|
| Accuracy | 0.8475 |
| Balanced accuracy | 0.6274 |
| Macro-F1 | 0.6572 |
| Weighted-F1 | 0.8409 |
| **High recall** (primary, Issue #10) | **0.300** |
| **High precision** (its guardrail) | **0.600** |

Read against a baseline that always answers *Medium*:

| | Accuracy | Balanced accuracy |
|---|---:|---:|
| Majority-class baseline | 0.7875 | 0.3333 |
| **This model** | **0.8475** | **0.6274** |
| Margin | +0.0600 | +0.2941 |

Confusion matrix (rows = true, cols = predicted):

|  | High | Low | Medium |
|---|---:|---:|---:|
| **High** | 6 | 0 | **14** |
| **Low** | 0 | 43 | **22** |
| **Medium** | 4 | **21** | 290 |

`evaluate.py` re-derives all eight headline numbers and **fails** if they disagree
with the values recorded in `artifacts/production/metadata.json`, so the README
table cannot silently go stale. Re-running is byte-identical (verified across
processes and `PYTHONHASHSEED` values); the held-out row set is pinned by the
fingerprint `8600c6c1d3c44bf4…`.

### Does 0.8475 look believable? — honestly, no, on its own

**84.75% accuracy is mostly an artefact of class imbalance, not skill.** Medium is
78.75% of the data, so a model that answered *Medium* for every single patient
would already score **0.7875**. The real model beats that constant baseline by
only ~6 points, and it pays for those points with a dangerous failure mode:
**it misses 14 of 20 genuinely High-risk patients (recall 0.300)**, calling them
*Medium*. For a screening tool that is the error that matters, and it is why
Issue #10 made High recall the primary metric and accuracy context only. Read the
headline as *balanced accuracy 0.6274*, not 0.8475.

**The single most suspicious number in this project is anything near 0.97.** It
is trivially reachable here without any real improvement: re-score the same
committed model on a different random split and accuracy jumps to **0.965–0.975**,
because ~78% of those new "test" rows were already seen while fitting (measured,
`evaluate.py` §7). The leaked v1 model scored 0.9975 for the same family of
reason (Issue #5). So: if a future change reports ~0.97, treat it as a **bug
report until proven otherwise** — check the split seed before celebrating.

What *is* believable: the model keys on real risk factors in medically sensible
directions with no dominant feature (Issue #12), the class ordering is
ordinal-consistent (no Low→High inversion), and the weak-but-nonzero High recall
is what a genuinely hard, imbalanced 3-class problem looks like — a model that
found real signal usually looks like this, not like 0.97. The honest summary is a
*useful Medium/Low triage aid with weak High sensitivity*, on **synthetic data**
that does not encode age or genetics strongly (Age ≈ 0 importance). Not a medical
device.

## 📦 Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app (serving entry point) |
| `evaluate.py` | **One-command evaluation** of the shipped model (Issue #15) — report, confusion matrix, contamination guard |
| `trace_model_provenance.py` | Issue #16: proves which notebook cell produced the legacy `model_xgb_new.pkl` (exit 0 = identified) |
| `train.py` | **Training entry point** — `python train.py` |
| `export_production.py` | Builds the 4-file production artifact set from the verified bundle (calibrated by default) |
| `calibrated_model.py` | Issue #11 hybrid wrapper: raw-forest labels + calibrated probabilities |
| `analyze_probability_calibration.py` | Issue #11 calibration analysis (Brier, ECE, reliability) — read-only |
| `analyze_feature_importance.py` | Issue #12 permutation-importance audit (plot → `docs/figures/`) — read-only |
| `verify_dataset.py` | Dataset integrity + provenance checks |
| `download_dataset.py` | Re-fetch + SHA-256-verify the dataset from upstream (Issue #13) |
| `verify_app_integration.py` | Proves `app.py` uses exactly the production artifacts (27 checks; 25 + 2 skips on a fresh clone) |
| `audit_overall_risk_score.py` | Target-leakage audit (Issues #5, #6) |
| `analyze_error_costs.py` | Cost-aware metric analysis (Issue #10) |
| `cancer-risk-factors.csv` | Training data, 2000 × 21, checksum-verified |
| `artifacts/production/` | **The 4-file serving set the app runs from — tracked in git**, so a fresh clone runs the app with no retraining |
| `artifacts/` (bundle, metrics, manifest) | Generated by `train.py` — not tracked, rebuilt by `python train.py` |
| `docs/` | Provenance, leakage, issues, and metric write-ups |
| `Cancer_Risk_Prediction_(ML).ipynb` | Original training notebook (historical; see `docs/`) |

```
artifacts/
├── model_bundle.joblib  # atomic training artifact (train.py) — NOT tracked
├── metrics.json         # held-out metrics, CV, per-class report — NOT tracked
├── manifest.json        # environment versions + checksums — NOT tracked
└── production/          # THE set the app runs from — TRACKED in git
    ├── model.pkl            # Pipeline [StandardScaler -> CalibratedModel]
    ├── label_encoder.pkl    # High / Low / Medium encoder
    ├── feature_names.pkl    # 17 leak-free features, pipeline order
    └── metadata.json        # version, config, metrics, fingerprints, calibration
```

Only `artifacts/production/` is committed, and deliberately so: it is the serving
contract, it is what lets a fresh clone run the app with **no retraining**, and
its integrity is machine-checked rather than assumed —
`metadata.json` records the sha256 of the three sibling files, and
`python export_production.py --verify-only` re-checks those plus the forest,
scaler and calibration content fingerprints. The training bundle is *not*
committed: it is a joblib pickle bound to exact library versions, and tracking it
would recreate the class of problem Issues #2/#4 documented. `python train.py`
rebuilds it bit-for-bit.

The pickles are marked `binary` in [`.gitattributes`](.gitattributes), so git
never applies end-of-line conversion to them.

The original `model_xgb_new.pkl`, `label_encoder.pkl` and `feature_names.pkl`
are **superseded and archived** under `archive/legacy-pickles-2026-09/` (with a
README explaining each); they used to sit at the repo root and are **not** read by
the app. See [`docs/ARTIFACT_AUDIT.md`](docs/ARTIFACT_AUDIT.md).

## 📦 Dataset

The training data is **committed to the repo** — a fresh `git clone` already
contains it, byte-identical to what produced the published metrics (verified
Issue #13: upstream download matches the committed file exactly).

| Property | Value |
|---|---|
| File | `cancer-risk-factors.csv` (repo root) |
| Shape | **2,000 rows × 21 columns** |
| SHA-256 | `01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5` |
| Size | 141,851 bytes |
| Upstream | Tarek Masryo, "Cancer Risk Factors & Types (2,000 Rows)" — [Kaggle](https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset) / [Hugging Face](https://huggingface.co/datasets/tarekmasryo/cancer-risk-factors-data) |
| Licence | **CC BY 4.0 (Attribution)** — share/adapt with attribution |

Verify any copy: `python verify_dataset.py` (full provenance report) or
`python download_dataset.py --verify-only`. Re-download a byte-identical copy:
`python download_dataset.py`. Full write-up:
[`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md).

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py          # that's it — artifacts/production/ is committed
```

**No training is required to run the app.** `artifacts/production/` is committed,
so a fresh clone already contains the four files the app loads. To confirm the
cloned artifacts are intact before you trust them:

```bash
python export_production.py --verify-only
```

### Retraining (optional)

Only if you actually want to rebuild the model from the committed dataset:

```bash
python train.py              # builds + verifies artifacts/model_bundle.joblib
python export_production.py  # re-exports artifacts/production/ from the bundle
```

The app will not start if the production set is missing — it loads exactly
`artifacts/production/{model,label_encoder,feature_names,metadata}.{pkl,json}`
and says so plainly rather than guessing.

**Calibrated probabilities (Issue #11, hybrid):** by default the exported
`model.pkl` wraps the forest in a `CalibratedModel` — `predict()` returns the
raw forest's class label (all published metrics stay valid), while
`predict_proba()` returns probabilities recalibrated out-of-fold (Brier
0.2418 → 0.2118, ECE macro 0.0912 → 0.0203 on the held-out set). Pass
`--no-calibrate` to export the plain uncalibrated pipeline, or
`--verify-only` to probe the existing production set (incl. the calibration
fingerprint).

## 🎯 Features

- 📁 **Batch CSV upload** — upload a CSV of patient records, get predictions + probabilities, download results
- 🧑 **Manual input** — enter patient features via widgets and get an instant risk prediction
- 📋 **Model card** in the sidebar — version, configuration, and held-out metrics, straight from the bundle

## 📊 Model performance

Random Forest, 17 features, trained on a stratified 80/20 split. **400 held-out
rows that were never resampled.**

| Metric | Score |
|---|---|
| Accuracy | 0.8475 |
| **Macro F1** | **0.6572** |
| Macro recall | 0.6274 |
| `High` precision / recall | 0.600 / 0.300 |

The target is 78.7% *Medium*, so accuracy overstates the model — macro-F1 is an
honest summary, but **the primary metric is `High`-class recall** (0.300), reported
with `High` precision (0.600) as its guardrail. It is the metric that matches the
cost of being wrong in a screening tool: a missed *High*-risk patient is the
expensive error, and it is the weakest part of this model. The full reasoning is
in [`docs/METRIC_SELECTION.md`](docs/METRIC_SELECTION.md) (Issue #10).

> **The previously published score of 0.9975 was a leakage artifact, not
> performance.** The model was being handed `Overall_Risk_Score`, a column that
> turned out to be a direct encoding of the answer, and the app was feeding it
> unscaled inputs. Both are fixed. See
> [`docs/TRAINING_AND_LEAKAGE.md`](docs/TRAINING_AND_LEAKAGE.md) for the full
> investigation and [`docs/T2_PROVENANCE_AND_LEAKAGE.md`](docs/T2_PROVENANCE_AND_LEAKAGE.md)
> for the Tier 2 findings.

## 🔍 Verify the claims

```bash
python evaluate.py                  # the headline numbers, re-derived from the artifacts
python trace_model_provenance.py    # prove which cell wrote the legacy model_xgb_new.pkl
python verify_dataset.py            # dataset checksum, schema, leakage pre-checks
python export_production.py --verify-only   # integrity of the production artifact set
python verify_app_integration.py    # 27 checks that app.py == production artifacts
python train.py --verify            # re-check a saved bundle's input contract + forest fingerprint
python audit_overall_risk_score.py  # reproduce the leakage findings
python analyze_error_costs.py       # reproduce the Issue #10 metric analysis
```

The first three work on a **fresh clone with no training** — that is the point of
committing `artifacts/production/`. On a clone, `verify_app_integration.py`
reports **25 passed, 2 skipped** (exit 0): the two skips only compare the
production set against `artifacts/model_bundle.joblib`, which is not committed.
After `python train.py` the same command reports **27/27**, with the bundle
acting as an independent second witness. The script never reports a pass it did
not earn, and never fails for a file the project chose not to track.

> ⚠️ For research/education only — **not** a medical diagnosis tool. The dataset is synthetic.
