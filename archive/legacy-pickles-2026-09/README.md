# Archive — superseded legacy artifacts (the "v1" pickles)

These are the **original artifacts from the first shipped version of the app**.
They were superseded during Tier 2/3 (Issues #2, #4, #5, #8) and are kept here
as **evidence of what was wrong**, per the project rule: nothing is deleted
silently. **No code in the app reads any file in this directory.**

They moved here from the repository root in **September 2026** (finalized
after all verification layers passed: 27/27 app-integration checks, model-contract
PASS, dataset verified). The root was left holding only the live production
contract in `artifacts/production/`.

## What each file is

| File | Size | What it is | Why it was superseded |
|------|-----:|------------|------------------------|
| `model_xgb_new.pkl` | 1.16 MB | Despite the *xgb* name, a `RandomForestClassifier` (sklearn 1.6.1), expecting **18** features (17 real + the leaked `Overall_Risk_Score`). | Issue #5 (target leakage via `Overall_Risk_Score`) and Issue #4 (untuned first baseline). The 0.9975 accuracy it produced was really leakage. No scaler was ever saved for it (Issue #2). |
| `label_encoder.pkl` | 399 B | Hand-built `LabelEncoder` (High/Low/Medium), pickled under sklearn 1.8.0. | Issue #2 — hand-built and decoupled from training; replaced by the encoder inside the verified bundle. |
| `feature_names.pkl` | 307 B | Hand-written list of **18** names (includes the leaked `Overall_Risk_Score`). | Issue #2 — hand-written order, never enforced; replaced by the bundle/production `feature_names.pkl`. |

## Provenance (checksums at archive time, 2026-09-29)

```
model_xgb_new.pkl   sha256 b75d02840c04a323b5dbc6b6c12d0c2e8b103d8f15bfecf440b7b76b5acced02
label_encoder.pkl  sha256 5cbe697be930be2c724922b5e95c42ca0e9b48448d4dc40c807841e3bc190c29
feature_names.pkl  sha256 e10bad9f34ca1063dbf5eb8f1c394ae58774a81ee37ea0d188477056f830eb98
```

These files are git-tracked and the moves were `git mv` (history preserved).

## What replaced them

- Training: `python train.py` → `artifacts/model_bundle.joblib` (model + scaler +
  encoder + feature list in one atomic bundle, contract-verified).
- Serving: `python export_production.py` → `artifacts/production/` — the four-file
  production set the app actually runs from, with no retraining.

The only scripts that still read these legacy files do so **as evidence** (the
dataset/feature audits that documented their flaws): `verify_dataset.py`,
`audit_overall_risk_score.py`, and `analyze_error_costs.py`. They now point at
this directory. The serving path (`app.py`, `export_production.py`,
`verify_app_integration.py`, `verify_model_contract.py`) does **not** touch them.
