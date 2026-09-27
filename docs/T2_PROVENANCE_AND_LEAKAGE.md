# T2 — Model Provenance and Leakage Audit

Covers GitHub Issues **#4** (which cell produced `model_xgb_new.pkl`),
**#5** (`Overall_Risk_Score` leakage audit) and **#6** (impact measurement).

**Reproduce with:**

```bash
python audit_overall_risk_score.py
python audit_overall_risk_score.py --json
```

Nothing was retrained, tuned, or modified. The audit trains models in memory
only, and never writes or deletes any artifact in this repository.

---

# Issue #4 — Which notebook cell produced `model_xgb_new.pkl`?

## #4.1 Where it is saved

Exactly one place in the whole repository writes the file:

| File | Cell | `execution_count` | Code |
|---|--:|--:|---|
| `Cancer_Risk_Prediction_(ML).ipynb` | **50** | **102** | `joblib.dump(model,'model_xgb_new.pkl')` |

Cell 50 carries the execution output `Model saved successfully as
'model_xgb_new.pkl'`, confirming it actually ran. `Multiclass_Classfication-
checkpoint.ipynb` contains **no** `joblib.dump` at all and produced no artifact.

## #4.2 The exact cell that produced it — and why it is not the obvious one

`model_xgb_new.pkl` was **not** produced by any of the notebook's tuned models.
It was produced by the notebook's **first, untuned baseline**.

The save cell serialises the bare name `model`. An AST scope analysis of every
binding of that name across all 52 cells:

| Cell | exec | Scope | Binding |
|---|--:|---|---|
| **10** | **55** | **top level** | `model = RandomForestClassifier(random_state=42, n_estimators=100)` |
| 35 | 89 | **local to `objective()`** | `model = RandomForestClassifier(**params)` |
| 47 | 100 | **local to `objective()`** | `model = XGBClassifier(**params, ...)` |
| 50 | 102 | top level | `joblib.dump(model, ...)` |

Cells 35 and 47 assign `model` only *inside* `def objective(trial)`. Python
function locals never escape into notebook globals, so at execution 102 the name
`model` still referred to the object created at execution 55 in cell 10.

**Conclusion: `Cancer_Risk_Prediction_(ML).ipynb` cell 10 (exec 55), saved by
cell 50 (exec 102).**

## #4.3 Model and data lineage

Every top-level rebinding of a data or model name, in execution order.
`<==` marks the state that was live when cell 10 ran:

```
 exec  cell  name        code
    48     2  df          df = pd.read_csv('/content/cancer-risk-factors.csv')   <==
    50     4  X           X = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type'])  <==
    50     4  le          le = LabelEncoder()                                     <==
    50     4  y           y = le.fit_transform(df['Risk_Level'])                 <==
    53     8  X_train     X_train, X_test, y_train, y_test = train_test_split(   <==
                              X, y, test_size=0.2, random_state=42, stratify=y)
    54     9  scaler      scaler = StandardScaler()                              <==
    54     9  X_train     X_train = scaler.fit_transform(X_train)                <==
    54     9  X_test      X_test = scaler.transform(X_test)                      <==
    55    10  model       model = RandomForestClassifier(random_state=42,        <==
                              n_estimators=100)   ... .fit(X_train, y_train)
    57    14  log_reg     LogisticRegression(random_state=42)
    80    25  X, y        rebuilt, now dropping Overall_Risk_Score
    83    28  rf_model    RandomForestClassifier(random_state=42)                -- SMOTE
    92    38  final_rf    RandomForestClassifier(n_estimators=347, max_depth=19) -- DISCARDED
    97    44  xgb         XGBClassifier(...)                                     -- DISCARDED
    98    45  final_xgb   XGBClassifier(**best_params)                           -- DISCARDED
   100    47  final_xgb   XGBClassifier(**best_params)                           -- DISCARDED
```

Nothing after exec 55 rebinds `model` at top level, so the object saved at
exec 102 is provably the cell-10 baseline.

**The shipped model:**

| Property | Value |
|---|---|
| Estimator | `RandomForestClassifier` — **not XGBoost**, despite the filename |
| `n_estimators` | 100 |
| `max_depth` | `None` (untuned) |
| `random_state` | 42 |
| `criterion` / `max_features` | `gini` / `sqrt` (sklearn defaults) |
| `classes_` | `[0, 1, 2]` |
| `n_features_in_` | 18 |
| `feature_names_in_` | **`None`** — fit on a numpy array, order not stored |
| Fit on | `X_train` = 1600 rows, `StandardScaler`-transformed, 18 features |
| Target encoding | `High=0, Low=1, Medium=2` |

Everything later in the notebook — the Optuna study, `final_rf`
(`n_estimators=347, max_depth=19`), and all four XGBoost experiments — was
**discarded** by the save in cell 50. The shipped artifact is the notebook's
first attempt, not its best result.

## #4.4 Is the shipped `.pkl` the artifact the app uses?

Yes, the same file, byte for byte.

| Check | Result |
|---|---|
| `app.py` `MODEL_PATH` | `os.path.join(BASE_DIR, "model_xgb_new.pkl")` |
| sha256 of that file | `b75d02840c04a323b5dbc6b6c12d0c2e8b103d8f15bfecf440b7b76b5acced02` |
| `joblib.load` in `app.py` | reads exactly that path |
| Scaling applied at inference | **none** — no scaler is loaded, so raw values are fed to a model fit on standardized values |

The app and the notebook therefore disagree at inference. Quantified in
Issue #1: the model collapses to `Medium` 400/400 on the held-out split.

## #4.5 Can the notebook reproduce the artifact?

**Yes, exactly.** Rebuilding cells 2/4/8/9/10 from the tracked CSV reproduces
the shipped model:

```
estimator            : RandomForestClassifier   match
n_estimators         : 100        max_depth: None    random_state: 42
feature_importances_ : identical
all 100 trees        : identical
classes_             : [0 1 2]    identical
held-out predictions : identical on 400/400 rows   -> MATCH
```

The audit script re-checks this every run against the real pickle, so any future
drift is caught.

## #4.6 Reproducibility gaps for #4

| Gap | Severity |
|---|---|
| Training is not a pipeline. ~15 experiments mutate shared globals (`X`, `y`, `X_train`, `model`), so *which* model gets saved depends on execution order. Saving by bare name is the root cause. | High |
| No `train.py`; the only executable record of training is a 52-cell exploratory notebook. | High |
| Notebooks read the Colab-only path `/content/cancer-risk-factors.csv`. | Medium |
| The saved model is the untuned baseline; all tuning discarded. Nothing in the repo says so. | Medium |
| `feature_names_in_` is `None`; feature order survives only in a hand-written `.pkl`. | High (Issue #2) |
| Filename and README both claim XGBoost. | Low |

---

# Issue #5 — `Overall_Risk_Score` target-leakage audit

## #5.1 How is `Overall_Risk_Score` created?

**Not in this repository.** A search for any assignment to `Overall_Risk_Score`
or `Risk_Level` across both notebooks and all `.py` files returns **zero hits**.
Both columns are read verbatim from `cancer-risk-factors.csv`.

Consequence: the formula relating them lives upstream, in whatever generated
the synthetic dataset. No download URL, dataset version, or generator script is
recorded, so the formula **cannot be read — only recovered empirically**. That
is what the rest of this section does.

The only places the column is *referenced*:

| File | Cell | Line |
|---|--:|---|
| `Cancer_Risk_Prediction_(ML).ipynb` | 18, 25 | `df.drop(columns=[... 'Overall_Risk_Score'])` |
| `Multiclass_Classfication-checkpoint.ipynb` | 28, 51, 53 | correlation, and another drop |

## #5.2 How is the target `Risk_Level` created?

Also not in this repository — a plain CSV column with three string values.

## #5.3 The relationship, recovered from data

The three classes occupy **completely disjoint** ranges of `Overall_Risk_Score`:

| Class | n | min | max |
|---|--:|--:|--:|
| Low | 324 | 0.029285 | 0.329922 |
| Medium | 1574 | 0.330033 | 0.659130 |
| High | 102 | 0.660797 | 0.852158 |

The gaps between classes are tiny but strictly positive:

```
Low    max 0.329922  ->  Medium min 0.330033    gap +0.000111
Medium max 0.659130  ->  High   min 0.660797    gap +0.001667
```

Fitting thresholds into those gaps gives a rule:

```
Overall_Risk_Score < 0.329978  ->  Low
                  < 0.659964  ->  Medium
                  else        ->  High
```

**This rule classifies all 2000 rows with 100.00% accuracy and 0 mismatches.**

## #5.4 Is `Overall_Risk_Score` itself derived from the risk factors?

Regressing `Overall_Risk_Score` on the other 17 features gives linear
**R² = 0.8388**, with strikingly structured coefficients:

| Feature | weight | | Feature | weight |
|---|--:|---|---|--:|
| Family_History | +0.01504 | | Fruit_Veg_Intake | −0.00081 |
| BRCA_Mutation | +0.01479 | | Gender | +0.00064 |
| Alcohol_Use | +0.01301 | | H_Pylori_Infection | −0.00053 |
| Smoking | +0.01272 | | BMI | +0.00049 |
| Air_Pollution | +0.01249 | | Calcium_Intake | −0.00032 |
| Diet_Salted_Processed | +0.01245 | | Physical_Activity | −0.00031 |
| Diet_Red_Meat | +0.01241 | | Age | −0.00011 |
| Obesity | +0.01220 | | Physical_Activity_Level | −0.00001 |
| Occupational_Hazards | +0.01217 | | intercept | −0.00077 |

The nine risk-factor columns carry near-equal positive weights
(~+0.012…+0.015) while Age, BMI, activity and calcium sit essentially at zero.
`Overall_Risk_Score` is an upstream **weighted composite of the risk factors**.

## #5.5 Verdict: the leakage chain

```
lifestyle + genetic factors
        │
        ▼  weighted composite  (R² = 0.84 on the 17 features)
Overall_Risk_Score          ← given to the model as an INPUT
        │
        ▼  thresholds at ~0.33 and ~0.66
Risk_Level                  ← the model is asked to predict THIS
```

`Overall_Risk_Score` is not a *proxy* for the target. It **is** the target,
continuously encoded and then quantized into three bins. The model is handed the
middle of the generating chain and asked to reproduce its output.

**Leakage exists, and it is total.** The audit script reports
`>>> VERDICT: LEAKAGE`.

### Corroboration

- Feature importance: `Overall_Risk_Score` = **0.7034** of the shipped model
- `Overall_Risk_Score` **alone** reaches accuracy **0.9975** — identical to using
  all 18 features (see Issue #6)
- The notebook author independently identified it. `Multiclass_Classfication-
  checkpoint.ipynb` cell 53 comments: *"droping overall_risk_score from the
  feature list (as it is leaking the answer)"*. Cells 18 and 25 of the other
  notebook drop it too. **The shipped model was produced before that
  correction and still consumes the leaked feature.**

## #5.6 Impact

| | |
|---|---|
| Reported accuracy 0.9975 | Measures how well the model reads the answer, not risk prediction |
| `High` class (5.1% of data) | Recovered almost entirely via the leak; genuine drivers contribute ~nothing measurable |
| `app.py` exposure | The leak is surfaced as a user input: an `Overall_Risk_Score` slider (0.0–1.0, default 0.5) under a group literally titled **"📊 Engineered score"**. The app asks the user to supply a value that already determines the label. |
| Any tuning performed on this target | Invalid. Every Optuna study in the notebook optimised against a metric computed from a leaked feature. |
| Downstream trust | Any reported result is unusable for clinical or decision support. |

## #5.7 Recommended action (not applied)

1. **Drop `Overall_Risk_Score` from the feature set** and retrain. The
   notebook already did this three times; the fix was simply never shipped.
2. **Stop asking for it in `app.py`.** Remove the "📊 Engineered score" group
   and the `Overall_Risk_Score` rule from `RULES`.
3. **Regenerate the dataset without the column**, or document it as
   target-derived and quarantine it — otherwise any future notebook cell that
   re-adds it silently reintroduces the leak.
4. **Re-baseline expectations honestly.** The leakage-free number is
   macro-F1 ≈ 0.49, not 0.99 (Issue #6).
5. **Address the 78.7 / 16.2 / 5.1 imbalance** — without the leak the model
   collapses to `Medium`, so class weighting or resampling is needed before the
   minority classes are usable at all.

---

# Issue #6 — Measured impact of `Overall_Risk_Score`

## #6.1 Method

Three arms, **controlled**. Identical in every respect except the feature list:

| Held constant | Value |
|---|---|
| Dataset | `cancer-risk-factors.csv`, sha256 `01291f8b…` |
| Split | `test_size=0.2, random_state=42, stratify=y` |
| Preprocessing | `StandardScaler`, fit on train only, applied to test |
| Estimator | `RandomForestClassifier(random_state=42, n_estimators=100)` |
| Tuning | **none** |

Because the split depends only on `y` and the seed, **all three arms get the
exact same 400 held-out rows** — a genuinely controlled comparison.

| Arm | Features |
|---|---|
| **A** | all 18 (shipped configuration) |
| **B** | 17 — `Overall_Risk_Score` removed |
| **C** | 1 — `Overall_Risk_Score` alone (added to isolate its share) |

## #6.2 Results

| Metric | A (18) | B (17) | C (ORS alone) | Δ A−B |
|---|--:|--:|--:|--:|
| Accuracy | **0.9975** | 0.8350 | **0.9975** | +0.1625 |
| Precision (macro) | 0.9989 | 0.8933 | 0.9949 | +0.1056 |
| Recall (macro) | **0.9833** | 0.4586 | 0.9989 | +0.5247 |
| F1 (macro) | **0.9909** | 0.4943 | 0.9969 | +0.4966 |
| F1 (weighted) | 0.9975 | 0.7953 | 0.9975 | +0.2022 |
| `ORS` importance | 0.7034 | — | 1.0000 | — |

### Per-class, arm A (with the feature)

| Class | Precision | Recall | F1 | n |
|---|--:|--:|--:|--:|
| High | 1.000 | 0.950 | 0.974 | 20 |
| Low | 1.000 | 1.000 | 1.000 | 65 |
| Medium | 0.997 | 1.000 | 0.998 | 315 |

```
true \ pred      High       Low    Medium
High                19         0         1
Low                  0        65         0
Medium               0         0       315
```

### Per-class, arm B (without it)

| Class | Precision | Recall | F1 | n |
|---|--:|--:|--:|--:|
| High | 1.000 | **0.050** | 0.095 | 20 |
| Low | 0.846 | 0.339 | 0.483 | 65 |
| Medium | 0.834 | 0.987 | 0.904 | 315 |

```
true \ pred      High       Low    Medium
High                 1         0        19
Low                  0        22        43
Medium               0         4       311
```

### Per-class recall delta

| Class | A | B | Δ |
|---|--:|--:|--:|
| High | 0.9500 | 0.0500 | **+0.9000** |
| Low | 1.0000 | 0.3385 | **+0.6615** |
| Medium | 1.0000 | 0.9873 | +0.0127 |

## #6.3 Faithfulness of the harness

Arm A is not a reimplementation that merely scores similarly — it is the same model:

```
shipped model_xgb_new.pkl accuracy on this split : 0.9975
arm A rebuilt-from-notebook accuracy              : 0.9975
predictions identical on 400/400 held-out rows    -> MATCH
```

## #6.4 Interpretation

**The feature is not merely disproportionately responsible — it is entirely
responsible.**

Arm C uses `Overall_Risk_Score` and *nothing else*, and still reaches
accuracy 0.9975. The contribution of the other 17 features to accuracy is
**+0.0000**. The shipped model's reported performance is a direct readout of a
column that was thresholded to produce the label.

Three things follow:

1. **Accuracy is the wrong lens here.** Arm B's 0.8350 looks respectable but is
   almost entirely the majority class: `Medium` recall is 0.987 while `High`
   recall collapses to 0.050. On the class that carries 19 of 20 missed
   `High` patients, the model is close to useless. Macro-F1 0.4943 is the
   honest figure.

2. **The leak is doing the work on the minority classes.** `High` recall moves
   0.05 → 0.95, a swing of 0.90. Since `High` is only 5.1% of the data, the
   leak is precisely what makes the model look clinically useful. Remove it and
   the class is not predicted at all in any meaningful quantity (1 of 20).

3. **The residual 0.8350 is not a consolation.** It is 78.7% majority-class
   baseline plus a little genuine signal. `Overall_Risk_Score` is 83.9%
   linearly reconstructible from the legitimate features, so a *leakage-free*
   pipeline could still recover much of it — legitimately, by predicting the
   score from the factors and thresholding afterwards. That path is not
   available to the model as shipped, because the score is handed to it
   pre-computed.

## #6.5 Conclusion for #6

`Overall_Risk_Score` accounts for **100% of measured performance**. Its
disproportionate responsibility is not a matter of degree: a model given that
column alone matches the full 18-feature model exactly, and removing it costs
0.1625 accuracy, 0.4966 macro-F1, and 0.90 recall on the `High` class.

Any metric computed with this feature present — including every Optuna study in
the training notebook — is invalid and should not be reported.
