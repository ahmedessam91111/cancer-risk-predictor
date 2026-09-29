# MODEL_PROVENANCE.md — which code produced `model_xgb_new.pkl`

**Issue #16. Answer: notebook cell index 10 (`execution_count=55`), variable
`model`.** Not XGBoost, not a tuned model, not the last thing that ran.

Everything below is reproducible:

```bash
python trace_model_provenance.py     # exit 0 = positively identified
```

The script re-derives each claim from the artifact and the notebook, and
**re-fits the notebook's own cells verbatim** to confirm the identification at
bit level. Nothing here is taken from the file's name.

The artifact in question is the superseded v1 model, preserved as evidence at
[`archive/legacy-pickles-2026-09/model_xgb_new.pkl`](../archive/legacy-pickles-2026-09/model_xgb_new.pkl)
(sha256 `b75d02840c04a323b5dbc6b6c12d0c2e8b103d8f15bfecf440b7b76b5acced02`,
1,162,849 bytes). It is no longer served; `app.py` reads only
`artifacts/production/`.

---

## 1. What type of estimator is it? Does the filename match?

| | |
|---|---|
| **Actual type** | `sklearn.ensemble._forest.RandomForestClassifier` |
| **`repr(model)`** | `RandomForestClassifier(random_state=42)` |
| **Filename implies** | XGBoost — gradient boosting |
| **Match?** | **No.** `"XGB" in type(model).__name__.upper()` → `False` |

The filename is wrong. This is not a case of a mislabelled-but-equivalent model:
XGBoost is a different algorithm (boosted sequential trees) and is a different
library. The file contains a bagged random forest from scikit-learn. Evidence
that the object is genuinely a forest, not a wrapper: `len(model.estimators_) ==
100`, and `estimators_[0]` is a
`sklearn.tree._classes.DecisionTreeClassifier`; 12,734 tree nodes in total,
`max_depth=None` on every tree.

**The `repr` line is the most useful single fact in this document.** scikit-learn
prints *only non-default* parameters in `repr`. Because the repr is bare, we know
**every hyperparameter except `random_state` is at its library default** — this
model was never tuned, and `n_estimators=100` is itself the default.

The pickle also carries a version stamp: the only version string in the file is
`1.6.1` (scikit-learn), against `1.9.1` in the current environment. It was
written by a different, older install — which is why it can no longer be loaded
without warnings, and why it cannot be trusted to interoperate with the code
around it.

## 2. Hyperparameters, compared one by one against every model in the notebook

The notebook constructs estimators in **13 cells**, at **25 constructor call
sites**; **7** of those are `RandomForest`. Every `RandomForest` candidate,
checked against the artifact's `get_params()`:

| # | cell (exec) | variable | constructor | vs. artifact |
|---|---|---|---|---|
| **1** | **10 (55)** | **`model`** | `RandomForestClassifier(random_state=42, n_estimators=100)` | ✅ **MATCH** — all 19 params equal |
| 2 | 21 (62) | `model_new` | `RandomForestClassifier(random_state=42, n_estimators=100)` | ⚠️ **identical constructor**, but different variable name and 17 features |
| 3 | 28 (83) | `rf_model` | `RandomForestClassifier(random_state=42)` | ❌ fit on SMOTE-resampled rows |
| 4 | 35 (89) | `model` *(local)* | `RandomForestClassifier(**params)` from Optuna | ❌ tuned: `max_depth`, `min_samples_*`, `bootstrap`, `criterion` all non-default |
| 5 | 38 (92) | `final_rf` | `RandomForestClassifier(**best_params)` | ❌ tuned: `n_estimators=347`, `max_depth=19` |
| 6 | 41 (93) | `rf` | `RandomForestClassifier(**params)` | ❌ tuned, inside `ImbPipeline` |
| 7 | 42 (94) | `rf` | `RandomForestClassifier(**best, n_jobs=-1)` | ❌ tuned, inside `ImbPipeline` |
| 8 | 14 (57) | `log_reg` | `LogisticRegression(random_state=42)` | ❌ wrong type |
| 9 | 44 (97) | `xgb` | `XGBClassifier(...)` in `ImbPipeline` | ❌ wrong type |
| 10 | 45 (98) | `xgb`/`final_xgb` | `XGBClassifier(**best_params, n_jobs=-1)` | ❌ wrong type |
| 11 | 46 (99) | `xgb_weighted` | `XGBClassifier(...)`, `sample_weight` | ❌ wrong type |
| 12 | 47 (100) | `model` *(local)* | `XGBClassifier(**params, n_jobs=1, random_state=42)` | ❌ wrong type **and** function-local |
| 13 | 27 (82) | `smote` | `SMOTE(random_state=42)` | ❌ not a model |

Only **one** candidate matches on every parameter: cell 10. The nearest rival,
**cell 21, has a byte-for-byte identical constructor** and is the reason the
input shape had to be checked as well as the parameters — see §3.

## 3. How many input features, and which training set has that shape?

`model.n_features_in_ == 18`.

| cell (exec) | line | feature set | width |
|---|---|---|---|
| **4 (50)** | `X = df.drop(columns=['Risk_Level','Patient_ID', 'Cancer_Type'])` | keeps `Overall_Risk_Score` | **18** ✅ |
| 18 (59) | `X_new = df.drop(columns=[…, 'Overall_Risk_Score'])` | leak-free | 17 ❌ |

The artifact expects the **cell-4** schema: the 17 real factors **plus
`Overall_Risk_Score`**. Two confirmations that this is not a matter of taste:

* **The model refuses the clean schema.**
  `model.predict(np.zeros((1, 17)))` raises
  `ValueError: X has 17 features, but RandomForestClassifier is expecting 18
  features as input.` The v1 app could not have served the leak-free model.
* **The importance is concentrated exactly where leakage would put it.**
  `Overall_Risk_Score` carries **70.3%** of all importance, and the top
  feature is **17.8×** the runner-up. For comparison, the shipped clean model
  tops out at **1.05–1.42×** (Issue #12). A 17.8× ratio is not a real risk
  factor; it is the target echoing back at itself (Issue #5).

## 4. Which cell wrote the file, and which variable did it save?

**One cell writes a file in the entire notebook:**

```
cell index 50, execution_count=102
    import joblib
    joblib.dump(model,'model_xgb_new.pkl')
    print("Model saved successfully as 'model_xgb_new.pkl'")
```

`model_xgb_new.pkl` is also the only `.pkl` filename the notebook ever mentions.
The save cell contains **0 `.fit()` calls and 0 `.transform()` calls** — it does
no fitting, no feature selection, and no sanity check. It serialises whatever the
global name `model` happened to refer to, and prints a success message either
way.

## 5. The mechanism — how the wrong model got saved

This is the part worth understanding, because the filename is not merely
inaccurate: it is *inverted* relative to the notebook's own history.

`model` is assigned in three cells, but **only one of them is at module scope**:

| cell (exec) | scope | assignment |
|---|---|---|
| 10 (55) | **module (global)** | `model = RandomForestClassifier(random_state=42, n_estimators=100)` |
| 35 (89) | **inside `def objective(trial):`** | `model = RandomForestClassifier(**params)` |
| 47 (100) | **inside `def objective(trial):`** | `model = XGBClassifier(**params, n_jobs=1, random_state=42)` |

In a Jupyter kernel, `model = …` inside a function body creates a **local**
name. The Optuna tuning functions each build their candidate under that local
name, score it, and throw it away. **The global `model` is never rebound after
cell 10.**

So at `execution_count=102`, when cell 50 ran `joblib.dump(model, …)`, it
dumped the global — still cell 10's untuned forest from `execution_count=55`.
The tuning that ran *after* it (including the XGBoost work the filename
advertises, which was the last modelling in the notebook at `exec 100`) never
touched the object that got written to disk.

The chain of mistakes, in order:

1. **A reused variable name across unrelated experiments.** `model` is
   overwritten again and again as the notebook moves from one experiment to the
   next.
2. **Tuning work hidden inside a closure.** The later `model = …` assignments
   were function-local, so they read as "the model was replaced" while leaving
   the global untouched.
3. **A save cell that trusts the name instead of the object.** It dumped a
   variable called `model` and named the file after an algorithm that variable
   never held.
4. **A success message that cannot fail.** `print("Model saved successfully…")`
   reports on serialisation, not on whether the right model was serialised. The
   run looked clean.

The net effect is an artifact whose name promises a tuned gradient-boosting
model and whose contents are the first untuned baseline in the notebook — the
one number in the whole project that was never meant to be believed.

## 6. Positive identification — not just elimination

Excluding the other candidates is suggestive; matching them is proof. So
`trace_model_provenance.py` re-runs the notebook's cells 4 → 8 → 9 → 10
**verbatim** and compares the result against the committed artifact:

| check | result |
|---|---|
| all 19 hyperparameters equal | ✅ |
| `n_features_in_` equal (18) | ✅ |
| `feature_importances_` max abs difference | **0.000e+00** |
| predictions identical on all 400 test rows | ✅ (0 rows differ) |
| confusion matrix — refit vs artifact vs cell 11's printed log | **all three agree** |

```
refit      : [[19, 0, 1], [0, 65, 0], [0, 0,315]]
artifact   : [[19, 0, 1], [0, 65, 0], [0, 0,315]]
cell 11 log: [[19, 0, 1], [0, 65, 0], [0, 0,315]]
```

A refit from a clean interpreter under a different scikit-learn version (1.9.1
vs the 1.6.1 that wrote the file) reproduces the artifact's fitted state exactly.
That is identification, not inference.

**The 0.9975.** Those matrices give 399/400 = **0.9975** correct. Cell 11's
classification report rounds this to `accuracy 1.00` — the number the project
was long described by. It is target leakage (Issue #5), not a 99.75%-accurate
model. Note the shape of the confusion matrix: **perfect on every class**, with
one single error. A real model on a hard 3-class problem does not do that.

## 7. Is this the best model you trained? No — on either reading

The honest answer has two parts, because the naive one is misleading.

**On the notebook's own leaderboard, it scored highest — and that is the
problem.** At 0.9975 it beat every other model by a mile, so on the numbers as
recorded it looks like the obvious deploy candidate. That ranking is itself an
artefact of leakage: it won by consuming a column that is a near-copy of the
label. The other models were handicapped by comparison because they were
evaluated on the clean 17-feature schema, which this model **cannot accept at
all** (it raises on 17 columns). The two groups were never on comparable
footing, so the leaderboard was meaningless.

**Among models evaluated on comparable, leak-free inputs, it was not even
eligible — and it is beaten by several.** From the notebook's own saved outputs:

| cell (exec) | model | accuracy | macro-F1 |
|---|---|---:|---:|
| **10 (55)** | **untuned RF, 18 features (the shipped artifact)** | **0.9975** | 0.99 |
| 11 → 15 (58) | logistic regression, 18 features | 0.98 | 0.95 |
| 22 (63) | `model_new` — same constructor, clean 17 | 0.83 | 0.49 |
| 29 (85) | SMOTE + RF, clean 17 | 0.84 | 0.64 |
| 42 (94) | tuned RF + SMOTE pipeline, clean 17 | 0.83 | 0.65 |
| 44 (97) | **SMOTE + XGBoost pipeline, clean 17** | **0.85** | **0.68** |
| 46 (99) | class-weighted XGBoost, clean 17 | 0.64 | 0.54 |

The best *legitimate* model is the **tuned SMOTE + XGBoost pipeline of cell 44**
(accuracy 0.85, macro-F1 0.68) — which is, notably, also the model the filename
was reaching for. **None of the tuned models was ever saved.** The notebook
wrote exactly one `.pkl`, and it was the wrong object.

*(Scores are read directly from each cell's saved output. Cell 10's model is
reported in cell 11, one cell later; cell 14's in cell 15. Rows marked
"18 features" were trained with `Overall_Risk_Score` present and are therefore
not comparable with the clean rows — that is the point of the next paragraph.)*

So the intended deploy target and the best available model were both
well-documented in the notebook, and the artifact was neither.

## 8. Was it the model you intended to deploy? No

Three independent signals say the intent was the XGBoost work, and none of it
reached the file:

1. **The filename.** `model_xgb_new.pkl` says XGBoost. The notebook's last
   modelling was XGBoost (cells 44–47, `exec 97–100`). Nothing in the notebook
   ever puts an `XGBClassifier` into this file.
2. **Chronology.** The file was written at `exec 102`, immediately after the
   XGBoost tuning at `exec 100` — someone clearly meant to save the freshly
   tuned model. The scoping bug meant they saved the `exec 55` forest instead.
3. **The filename's `new`.** It reads as a deliberate "the improved one",
   superseding a prior `model_xgb.pkl`. It is in fact the *oldest* model in the
   notebook.

**The intended artifact, by every available signal, was the tuned XGBoost
pipeline. What shipped was the first untuned baseline — trained on a leaked
column, never tuned, and mislabelled as the algorithm it does not contain.**

## 9. Why this mattered downstream

The consequences were not cosmetic, and they are the reason this document exists
rather than a one-line commit message:

* **A wrong headline.** 0.9975 was reported as the model's accuracy. The real
  figure for a leak-free model is 0.8475 (Issue #15). An inflated number is
  worse than no number, because it stops anyone from asking.
* **A serving contract that could not be met.** The artifact needs 18 columns;
  the app's feature set has 17. The app and its model disagreed about their own
  interface, and no scaler was ever saved for it (Issue #2), so the two cannot
  even be made to agree after the fact.
* **A model nobody could reproduce.** Because the producing cell was never
  identified, the artifact looked hand-made until now. It is not: it is exactly
  reproducible, and `trace_model_provenance.py` demonstrates it.

## 10. What replaced it

| | |
|---|---|
| v1 (this document) | cell 10's untuned RF, 18 leaky features, 0.9975 by leakage |
| v2 (current) | `python train.py` → `artifacts/model_bundle.joblib` → `python export_production.py` → `artifacts/production/` |

The current artifact is **class-weighted** (`class_weight='balanced'`), trained
on the **17 leak-free** features, evaluated on a held-out split that is never
resampled (Issues #6/#9), and reports **0.8475 accuracy / High recall 0.300**
(Issue #15). It is worse on paper than the leaky v1 model. That gap is the
entire value of the exercise.

**The habit this issue is really about:** an artifact should be able to name the
code that made it, and that code should be able to reproduce the artifact. Both
now hold — `trace_model_provenance.py` exits 0 only if the refit matches the
committed bytes' behaviour, and `export_production.py --verify-only` checks the
same fingerprints for the model that ships today.

---

**Related:** [`docs/T2_PROVENANCE_AND_LEAKAGE.md`](T2_PROVENANCE_AND_LEAKAGE.md)
(Issue #4, the first pass at this question) ·
[`docs/ARTIFACT_AUDIT.md`](ARTIFACT_AUDIT.md) (lifecycle) ·
[`../archive/legacy-pickles-2026-09/README.md`](../archive/legacy-pickles-2026-09/README.md)
(what each legacy file is) · [`docs/ISSUES.md`](ISSUES.md)
