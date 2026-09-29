# OVERALL_RISK_SCORE_VERDICT.md — is `Overall_Risk_Score` a legitimate model input?

**Issue #17. Verdict: NO. It is target leakage, and the deployed model does not
use it.**

`Risk_Level` is a deterministic function of `Overall_Risk_Score`, and
`Overall_Risk_Score` is itself computed from the seventeen legitimate risk
factors. The column is the *middle* of the chain `factors → Overall_Risk_Score →
Risk_Level`, handed to a model whose entire job is to reconstruct the last link
from the first. It is not an observation about a patient; it is the answer, one
step upstream of the label.

Reproduce every figure below with:

```bash
python measure_risk_score_leakage.py     # exit 0 = measurement complete
```

The deeper audit from Issue #5 — including the controlled with/without
comparison — is in `audit_overall_risk_score.py`.

---

## 1. How is `Overall_Risk_Score` produced — a measurement, or a computation?

**A computation.** Two independent lines of evidence.

**(a) Its shape is the shape of a score, not of an observation.** A clinical
measurement (a blood pressure, an age, a lab value) is not confined to the unit
interval, and it repeats across patients. This column:

| property | value |
|---|---|
| dtype | `float64` |
| distinct values | **2000 of 2000 rows** — different for every patient |
| range | `[0.029285, 0.852158]` |
| mean / std | `0.454449` / `0.123074` |

Bounded in `[0, 1]`, continuous, and unique per row: that is a
unit-normalised composite.

**(b) It is computable from the other columns.** A linear fit of
`Overall_Risk_Score` on the 17 legitimate factors gives

```
R² = 0.838764        residual std = 0.049407
```

with nine risk-factor columns carrying near-equal positive weight
(`Family_History` +0.0150, `BRCA_Mutation` +0.0148, `Alcohol_Use` +0.0130,
`Smoking` +0.0127, `Air_Pollution` +0.0125, `Diet_Salted_Processed` +0.0125,
`Diet_Red_Meat` +0.0124, `Obesity` +0.0122, `Occupational_Hazards` +0.0122) and
the other eight (`Age`, `Gender`, `BMI`, `Fruit_Veg_Intake`,
`Physical_Activity`, `Physical_Activity_Level`, `Calcium_Intake`,
`H_Pylori_Infection`) sitting at essentially zero.

The unexplained 16% is **not** a term we failed to model. Its correlation with
every one of the 17 inputs is negligible (largest |Spearman| = **0.0103**), and
it is close to normal (skew −0.0095, excess kurtosis −0.1111; 66.9% of cases
within 1σ vs a normal's 68.3%). That is the signature of **independent noise
added after the factors were combined**. So the generator behaves as:

```
Overall_Risk_Score ≈ weighted_sum(9 risk factors) + noise
Risk_Level          = threshold(Overall_Risk_Score)      # at 0.33 and 0.66
```

The formula lives upstream, in whoever produced the dataset — neither column is
created anywhere in this repository (`audit_overall_risk_score.py`, §5a). It is
recovered here empirically rather than read from a spec.

## 2. The real-patient test

> Picture a patient in front of you, before anyone has assessed their risk. Do
> you know this number? Who computes it, and from what?

**No — and that is the whole problem.** At the moment a prediction is needed,
nobody has produced `Overall_Risk_Score`:

* If the number does not exist yet, you cannot supply it. A model that needs it
  cannot run.
* If it *does* exist, it exists because someone already ran the risk model —
  because they already computed the weighted sum of the same seventeen factors
  and thresholded it. In that case the answer is already known, and asking the
  model for it again is theatre.

There is no third option. There is no clinician, device, or lab that measures
`Overall_Risk_Score` independently of the risk factors; the column *is* the
factors, compressed into the one form that makes the label readable off it. A
feature that is only available after the outcome is known is, by definition,
target leakage.

## 3. Its measured relationship to `Risk_Level`

Measured, not eyeballed. Note the label is encoded **ordinally** (Low=0,
Medium=1, High=2); an alphabetical encoding would report a near-zero correlation
and mean nothing.

**(a) Rank correlation.** Pearson `r = +0.770886`, Spearman `ρ = +0.712862`
(p ≈ 3e-310), Kendall `τ = +0.593264`.

That "0.71" looks moderate, and it is worth understanding why: the label has
only **3 levels**, and **1574 of 2000 rows** sit in one of them. Most pairs are
therefore *tied in the label* and cannot contribute to a rank correlation, which
caps the statistic. Asking the sharper question — of pairs drawn from
*different* classes, how often does the score order them correctly? —

```
703572 / 703572 cross-class pairs ordered correctly = 1.000000
```

The score **never** places a higher-risk patient below a lower-risk one. The
ordering is total; only the summary statistic is coarse.

**(b) Mutual information.**

```
I(Overall_Risk_Score ; Risk_Level) = 0.633875 nats
H(Risk_Level)                      = 0.635146 nats   (the label's total uncertainty)
normalised MI                      = 0.9980
```

Knowing this one column removes **99.8%** of the uncertainty about the risk
class. For scale, the strongest single legitimate feature is `Air_Pollution` at
`0.086437` nats, and the individual mutual informations of **all seventeen**
legitimate features sum to `0.394426` nats — *less than this one column alone*.

**(c) Grouped summary.** The class ranges do not touch:

| class | n | min | max | mean |
|---|---:|---:|---:|---:|
| Low | 324 | 0.029285 | 0.329922 | 0.269458 |
| Medium | 1574 | 0.330033 | 0.659130 | 0.476616 |
| High | 102 | 0.660797 | 0.852158 | 0.699991 |

The gaps constrain the decision thresholds to
`(0.329922, 0.330033)` and `(0.659130, 0.660797)` — i.e. cuts at **0.33** and
**0.66**. A rule that reads *only* this column and nothing else:

```
2000 / 2000 correct = 100.00%      macro-F1 = 1.000000
```

For comparison, the deployed seventeen-feature model scores macro-F1 **0.6572**
on its held-out split (Issue #15). A single column, with no model at all,
outscores it.

## 4. The notebook dropped it — so what happened?

The notebook is not naive about this. The decision is visible in three places:

| cell | what it does |
|---|---|
| **4** (exec 50) | `X = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type'])` — **keeps** `Overall_Risk_Score` |
| **12** (markdown) | *"The model accurately distinguishes all risk levels, with only 1 mistake out of 400"* — the leaky result is celebrated, before its cause is suspected |
| **17** (markdown) | **"## Removing Overall_Risk_Score and retrying"** |
| **18** (exec 59) | `X_new = df.drop(columns=[…, 'Overall_Risk_Score'])` — **drops** it |
| **25** (exec 80) | drops it again, for the SMOTE/resampling work that follows |

So the author's reasoning was sound: the 1-in-400 result was *too* good, it was
flagged, and the column was removed. **Everything after cell 18 is built on the
leak-free 17-feature schema.** The same instinct appears in cell 7's markdown,
which removes `Cancer_Type` precisely because *"the model will cheat by learning
the mapping"*.

The failure was not a wrong judgement — it was that the judgement never reached
the artifact. Notebook cell 4 still defines the 18-feature set, and because of
the variable-shadowing bug in the save cell (Issue #16), the one model trained
on it — cell 10's untuned forest — is the one that got written to
`model_xgb_new.pkl`. The author removed the column and then shipped a model that
used it anyway.

**Does the model currently in this repo reflect the decision? Yes.** This is
verified mechanically, not asserted:

| check | result |
|---|---|
| `artifacts/production/feature_names.pkl` | **17** features |
| contains `Overall_Risk_Score`? | **False** |
| `metadata.json` → `excluded_features` | `{'Overall_Risk_Score': 'target leakage, Issue #5'}` |
| `app.py` asks the user for it? | **False** |

The deployed model is trained on the seventeen leak-free factors, with
`class_weight='balanced'`, a persisted `StandardScaler`, and a real, much lower
score (macro-F1 0.6572). Removing the leak cost a great deal of apparent
accuracy, and bought back a number that means something.

## 5. What is a user actually doing when they move that slider?

The v1 app presented `Overall_Risk_Score` as a **0.00–1.00 slider, default
0.5**, in a form group literally titled **"📊 Engineered score"** (git `904cf6f`,
later removed in `015368e`). The app was, to its credit, telling the truth in
the group name.

Because `Risk_Level` is a threshold function of the slider's value, moving the
slider is not describing the patient — **it is choosing the diagnosis.** Holding
all 17 other features fixed at their medians and scoring them with the archived
v1 model:

| slider | predicted class |
|---:|---|
| 0.20 | Low |
| 0.32 | Low |
| **0.33** | **Medium** ← flips |
| 0.50 | Medium |
| 0.66 | Medium |
| **0.67** | **High** ← flips |
| 0.80 | High |

The other inputs never moved, yet the answer tracked the slider across the two
cuts. A user who wanted a "High risk" verdict dragged the slider past 0.66; one
who wanted "Low" dragged it below 0.33. The form was self-answering, and any
accuracy measured through it was a measurement of the slider, not of the model.

## 6. What should happen to that slider

**It has already been removed** (commit `015368e`), and the removal is documented
in the code itself, at `app.py:215`:

```python
# Issue #5 proved Overall_Risk_Score is a 100% accurate threshold function of
# Risk_Level, so asking the user for it made the form trivially self-answering.
# It is no longer a model input, so it is no longer asked for.
```

The correct disposition is exactly this: **delete the input, do not merely
ignore it.** An input that is collected but unused is a trap for the next
person; an input that is collected and *used* is the leak. The removal is now
enforced, not just remembered: `verify_model_contract.py` (Issue #14) check 10
asserts `LEAKY_FEATURE not in features`, and check 9 re-derives the feature list
as *CSV column order minus the non-features **and** `Overall_Risk_Score`*. Put
the column back and both checks fail with a non-zero exit. `verify_app_integration.py`
separately fails if the app and the model disagree about their interface.

There is one honest residual to keep in view: `Overall_Risk_Score` is ~84%
recoverable from the seventeen legitimate factors. That is not a leak — it is
the model legitimately learning a real association from real inputs, and it is
precisely the signal the deployed model is *supposed* to find. The difference
between "the factors predict the risk" and "we handed the model the answer" is
the difference between `R² = 0.84` with a genuinely uncertain label, and a
single column that classifies all 2000 rows with zero errors.

---

## Summary of the verdict

| question | answer |
|---|---|
| Measurement or computation? | **Computation** — bounded in [0,1], unique per row, `R² = 0.8388` on the other 17 factors, residual uncorrelated noise |
| Available at prediction time? | **No** — it exists only after the risk factors have already been scored |
| Measured relationship to the label | `H`-normalised MI **0.9980**; **703572/703572** cross-class pairs ordered correctly; class ranges disjoint; a one-column rule scores **100.00%** / macro-F1 **1.000000** |
| Legitimate input? | **No — target leakage** |
| Does the deployed artifact use it? | **No** — 17 features, `excluded_features` recorded in `metadata.json` |
| What about the slider? | **Removed** in `015368e`, with the reason in the code; guarded by the feature contract |

---

**Related:** `measure_risk_score_leakage.py` (this issue's measurements) ·
`audit_overall_risk_score.py` (Issue #5/#6 audit and the controlled
with/without comparison) · [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md)
(Issue #16 — how the leaky model came to be shipped) ·
[`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) (the 17-feature contract) ·
[`ISSUES.md`](ISSUES.md)
