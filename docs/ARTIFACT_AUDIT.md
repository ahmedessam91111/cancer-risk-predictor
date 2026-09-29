# Artifact audit — what exists, what the app uses, what is superseded

**Status:** audit complete; production layout in use; superseded files kept as
evidence (deletion/archival deferred until the full tier is verified, per the
project rule "investigate first, never silently delete").

Reviewed: 2026-09-28

---

## 1. Inventory at audit time

| path | bytes | produced by | app uses it? | status |
|------|------:|-------------|:---:|--------|
| `model_xgb_new.pkl` (now `archive/legacy-pickles-2026-09/`) | 1.16 MB | notebook cell: `joblib.dump(model,'model_xgb_new.pkl')` | **no** | **superseded → archived** |
| `label_encoder.pkl` (now `archive/legacy-pickles-2026-09/`) | 399 B | hand-built (no notebook trace found) | **no** | **superseded → archived** |
| `feature_names.pkl` (now `archive/legacy-pickles-2026-09/`) | 307 B | hand-written list (no notebook trace found) | **no** | **superseded → archived** |
| `artifacts/model_bundle.joblib` | 3.77 MB | `python train.py` (traces notebook cells 2/4/8/9/10 → save cell 50) | **no** (replaced below) | **intermediate** |
| `artifacts/production/{model,label_encoder,feature_names,metadata}.{pkl,json}` | ~3.8 MB total | `python export_production.py` | **yes** | **production** |
| `cancer-risk-predictor/` (nested) | — | stale local clone at commit `9a66cdd` | no | untracked, gitignored |

## 2. What each legacy artifact actually is

Inspected by loading (documented evidence — all three were already superseded in
Tier 2/3 and no code reads them):

* **`model_xgb_new.pkl`** — despite the *xgb* name it is a
  `RandomForestClassifier`, fitted by **scikit-learn 1.6.1** (a version long
  gone from `requirements.txt`), expecting **18 features** — i.e. the 17 real
  factors **plus `Overall_Risk_Score`**, the leaked column proven in Issue #5.
  This is the artifact that produced the 0.9975 that was really target leakage.
  Reloading it under the current scikit-learn raises version warnings, and no
  scaler was ever saved for it (Issue #2), so it cannot be scored fairly.
* **`label_encoder.pkl`** — a hand-built `LabelEncoder` (`High/Low/Medium`),
  pickled under scikit-learn 1.8.0. Superseded by the encoder inside the bundle.
* **`feature_names.pkl`** — a hand-written list of **18** names; the clean 17
  match the current feature set but the list also contains `Overall_Risk_Score`.

All three are git-tracked and were deliberately **kept as evidence, not
deleted**. In September 2026 (after every verification layer passed) they were
`git mv`-ed to **`archive/legacy-pickles-2026-09/`** with a README explaining
each one; the repository root now carries only the live contract.

## 3. Why the production layout looks the way it does

The production set is exactly the four files requested:

```
artifacts/production/
    model.pkl            full inference pipeline  StandardScaler -> RandomForestClassifier
    label_encoder.pkl    LabelEncoder (High / Low / Medium)
    feature_names.pkl    the 17 leak-free feature names, in pipeline input order
    metadata.json        model version, config, metrics, environment, fingerprints, checksums
```

Two engineering notes, both `export_production.py`:

1. **`model.pkl` is a Pipeline that embeds the fitted scaler.** The requested
   layout has no separate scaler file, and that is a *feature*: there is no way
   for any caller to reach the forest with unscaled values — the exact failure
   mode Issue #2 documented. The layout still contains exactly four files.
2. **The file names `label_encoder.pkl` / `feature_names.pkl` deliberately
   collide with the deprecated root files only in name, not in path.** The app
   resolves them as `artifacts/production/<name>` and never by bare relative
   name. This separation is why the legacy files are described as *outside the
   production artifact path*.

## 4. Traceability (each production file → evidence)

| production file | derives from | evidence |
|---|---|---|
| `model.pkl` | bundle `model` + `scaler` (re-wrapped) | fingerprint `aeca50e59b3dc670…` matches bundle record; predictions **bit-identical** to bundle path on the 400 held-out rows (checked automatically by `export_production.py`) |
| `label_encoder.pkl` | bundle `label_encoder` | round-trip asserts classes `['High','Low','Medium']` |
| `feature_names.pkl` | bundle `feature_names` | 17 leak-free names asserted |
| `metadata.json` | bundle `metadata` + export record | records dataset sha256, forest fingerprint, scaler fingerprint, environment (incl. `environment_matches_reference`), metrics, and per-file sha256 |

**A reproducibility nuance, measured and documented.** Pickled **file bytes**
for `model.pkl` are NOT stable across python processes: `PYTHONHASHSEED` is
randomized per process and pickle's byte stream depends on it. Two exports of
the identical pipeline from the identical bundle in separate processes produce
different `model.pkl` hashes. This does **not** mean the model changed. The
project's identity checks are therefore defined on **array bytes**, which are
process-stable:

* `forest_sha256` — `sha256(feature|threshold|children_left|children_right)`
  over all 100 trees = `aeca50e59b3dc670…` (defined in `train.py`).
* `scaler_fingerprint` — `sha256(mean_|var_|scale_|n_features_in_)` as float64
  arrays (defined in `export_production.py`).

`export_production.py --verify-only` verifies BOTH the file-level sha256 (files
unchanged since export) and the content fingerprints (the model content is the
canonical bundle's content), so either direction of drift is caught.

## 5. Lifecycle plan for the superseded files

Per the project rule, nothing is deleted while the tier is unverified, and
nothing is ever deleted silently.

1. Keep `model_xgb_new.pkl`, `label_encoder.pkl`, `feature_names.pkl` as evidence
   while all verification layers pass (they do: 27/27 app-integration checks,
   model-contract PASS, dataset verified; app runs from the production set
   without retraining).
2. After the full tier (including issues #10–#13) is verified, the options are:
   * **archive**: move under `archive/legacy-pickles-2026-09/` with a
     `README.md` documenting why each was superseded (keeps the audit trail in
     git), or
   * **delete**: `git rm` the three files in a documented commit.
   → **Done: archived** (2026-09, `git mv`, README included). The three
   evidence scripts (`verify_dataset.py`, `audit_overall_risk_score.py`,
   `analyze_error_costs.py`) were repointed at the archive path; the serving
   path never reads them.
3. The nested `cancer-risk-predictor/` clone is a stale, untracked workspace,
   not project data; it is gitignored and can be deleted locally at any time
   without touching the repository.

Recommended: **archive**, because the leak-era artifacts are the evidence
behind Issues #2/#4/#5 and are cheap to keep.

## 6. How to rebuild the production artifacts (no retraining needed)

```bash
python train.py                  # 1. (only when retraining is actually wanted)
python export_production.py      # 2. always: export -> production/ from the bundle
python export_production.py --verify-only   # 3. integrity re-check
streamlit run app.py             # 4. app loads artifacts/production only
```

If `export_production.py` is run after a retrain, every metric, fingerprint and
checksum in `metadata.json` is regenerated from the new bundle automatically.