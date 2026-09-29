# Issue #12 — What is the model actually keying on?

**Status:** RESOLVED (analysis-only; no model change). Evidence script:
`analyze_feature_importance.py` (read-only, loads the deployed bundle). Figure:
`docs/figures/permutation_importance.png` (+ machine-readable
`permutation_importance.csv`).

---

## 1. Top-five features — two measures, one honest question

**Impurity-based** (`feature_importances_`, computed during fit **on training
data**):

| rank | feature | importance |
|---|---|---|
| 1 | Air_Pollution | 0.1477 |
| 2 | Alcohol_Use | 0.1325 |
| 3 | Smoking | 0.1081 |
| 4 | Diet_Salted_Processed | 0.1069 |
| 5 | Occupational_Hazards | 0.0958 |

**Permutation importance** (shuffle each feature on the **held-out 400 rows**,
n_repeats=10, drop in score when the feature loses its information):

| rank | feature | Δ accuracy | Δ f1-macro | Δ High-recall |
|---|---|---|---|---|
| 1 | Air_Pollution | +0.0470 ± 0.014 | +0.1340 ± 0.041 | +0.135 ± 0.098 |
| 2 | Alcohol_Use | +0.0495 ± 0.010 | +0.0933 ± 0.022 | +0.085 ± 0.045 |
| 3 | Smoking | +0.0435 ± 0.012 | +0.1033 ± 0.027 | +0.095 ± 0.047 |
| 4 | Diet_Red_Meat | +0.0375 ± 0.010 | +0.0754* | +0.080* |
| 5 | Occupational_Hazards | +0.0330 ± 0.012 | +0.0655 ± 0.026 | +0.085* |

\* = slot 4/5 differs slightly per scorer: for f1-macro slot 4 is
Diet_Salted_Processed (+0.0754); for High-recall slot 4 is **Obesity**
(+0.085 ± 0.050). The core cluster — **Air_Pollution, Alcohol_Use, Smoking,
diet, Occupational_Hazards — is stable across all four scorers.**

### Why the distinction between the two measures matters

* **Impurity-based importance** is computed during training from the trees'
  split history. It never sees a held-out example and is biased toward
  high-cardinality/continuous features (easy to split cleanly) and toward
  correlated features (credit is split between them). It measures "how useful
  this feature was while fitting" — not "how much a real prediction needs it."
* **Permutation importance** shuffles one feature's values on the *held-out*
  set and measures the drop in a chosen metric. It asks the more honest
  question: *"if this feature stopped carrying any information, how much worse
  would real predictions get?"* — but note the caveat that shuffling breaks
  the feature's joint distribution with anything correlated with it.

This distinction is not academic here — it is visible in the data (see §4:
**BMI rank 8 by impurity, rank ≈ last by permutation**).

---

## 2. Do the top features match the medical literature?

Yes, and in the *right direction*. Every leading feature is an established
cancer-risk exposure:

| feature | literature role | direction in data (Spearman vs risk level) |
|---|---|---|
| Smoking | lung, head/neck, bladder, pancreas… | +0.335 (higher → riskier) ✔ |
| Alcohol_Use | oral, esophageal, liver, breast, colorectal… | +0.313 ✔ |
| Air_Pollution | lung cancer (IARC Group 1, PM2.5) | +0.381 ✔ |
| Occupational_Hazards | occupational carcinogens | +0.254 ✔ |
| Diet_Red_Meat / Diet_Salted_Processed | colorectal cancer (processed meat IARC Group 1) | +0.215 / +0.287 ✔ |
| Fruit_Veg_Intake | protective (fiber, antioxidants) | **−0.124** (higher intake → lower risk) ✔ protective direction |
| Obesity | endometrium, breast (post-menopausal), colorectal… | +0.179 ✔ |

The model keys on **modifiable behavioural/environmental exposures**, with
directions that match epidemiology. The protective direction of
fruit/vegetable intake is a nice consistency check: the variable with a
*negative* relationship to risk is exactly the one the literature would call
protective.

---

## 3. Is there one feature that dominates everything else?

**No.** The signal is spread across a cluster of similar-sized exposure
features.

| scorer | #1 | #2 | ratio #1/#2 |
|---|---|---|---|
| accuracy | Alcohol_Use | Air_Pollution | 1.05 |
| f1-macro | Air_Pollution | Smoking | 1.30 |
| High-recall | Air_Pollution | Smoking | 1.42 |
| balanced-acc | Air_Pollution | Smoking | 1.25 |

Air_Pollution is consistently first (or tied) and is the largest single
contributor by non-accuracy scorers — but its drop is at most ~19% of the
total held-out signal (sum of all drops ≈ 0.26 for accuracy, 0.59 for
f1-macro), far from "the model is just Air_Pollution." Nothing in the model
collapses when any single feature is removed.

**Possible explanations for a *would-be* dominant feature, and how to tell
them apart** (for the record, since none dominates here):

1. *Genuine strong causal signal* — would show a monotone dose-response in a
   partial-dependence-style plot (mean risk rises steadily with the feature).
2. *Proxy for the leaked/omitted target* — a feature carrying target
   information; check correlation with the target and whether its importance
   survives when correlated features are removed.
3. *Interaction-dependent importance* — one feature unlocks many splits only
   in combination (e.g., cancer-type-conditional genes); would show up in
   feature-pair importance, not in single-feature permutation.

The join of these checks: Spearman with the target (+0.38 for Air_Pollution,
monotone), removal robustness, and pair importance. For this dataset the
answer is (1): exposure scores are simply the strongest *main-effect* drivers
the synthetic data encodes.

---

## 4. Redundant pairs — the surprise of this audit

The question expects BMI≈Obesity and Physical_Activity≈
Physical_Activity_Level to be near-duplicates. **They are not — numerically
these pairs are essentially independent:**

| pair | Spearman(a,b) | Spearman(a, risk) | Spearman(b, risk) | perm-imp a | perm-imp b | joint |
|---|---|---|---|---|---|---|
| BMI / Obesity | **−0.003** | +0.028 | +0.179 | **0.0000** | +0.0303 | +0.020 |
| Physical_Activity / Activity_Level | **+0.023** | +0.065 | +0.019 | +0.003 | −0.004 | +0.005 |

What this means:

* The **names** imply redundancy; the **values** show two independently
  generated variables. `Obesity` is not derived from `BMI` (nor `Activity_Level`
  from `Physical_Activity`) in this dataset. Whatever importance story exists
  belongs to `Obesity` alone; `BMI` contributes **zero** held-out signal.
* **Consequence for importance scores: none of the classic "dilution"** (a
  pair splitting credit) applies, because the pair members are uncorrelated.
  The error would be to read the names and *assume* correlation.

**The sharpest illustration of the entire audit** (why §1's distinction
matters): `BMI` is **rank 8 by impurity-based importance (0.0451)** — the
trees happily split on it during training, e.g. exploiting local
interactions/noise — yet **rank 17 (≈0) by held-out permutation importance**.
Impurity said "useful"; permutations say "the model doesn't actually need it."

**Decision on the pairs (no model change):** keep both columns. The deployed
model is unchanged — the only legitimate changes are documented drop
*candidates* for a future retrain: `BMI`, `Physical_Activity_Level` (and to a
weaker extent `Physical_Activity`) add no held-out signal. We do **not**
retrain in this issue; `metadata.json` and all metrics stay byte-valid.

---

## 5. Anything that surprised me

1. **Age barely matters (−0.015 vs risk; ~0 permutation importance).** Age is
   the strongest real-world cancer risk factor; this synthetic dataset encodes
   it as noise. A real-world version of this model would not look like this.
2. **BMI ≈ 0 held-out importance despite Obesity being 6th.** And the two are
   uncorrelated — a naming/reality mismatch to flag to any user of this data.
3. **BRCA_Mutation ≈ 0 and H_Pylori ≈ 0.** Real-world-important (BRCA for
   breast/ovarian, H. pylori for gastric), but their effects are conditional on
   cancer type — and `Cancer_Type` is excluded from features (by design).
   Main-effect measures (correlation, permutation) will under-report
   interaction-dependent factors like these.
4. **The High-recall top-5 adds Obesity and is very noisy** (±0.098 for
   Air_Pollution). With n=20 High patients, per-class importance for the rare
   class should be read as indicative, not precise.

**Bottom line:** the model is doing what its (synthetic) training data asks —
keying on behavioural/environmental exposure scores in medically sensible
directions — and it does not hide behind a single dominant feature. The
caveat: importance is only as medically sensible as the dataset; factors that
real medicine ranks first (age, genetics) are nearly absent from this data and
therefore from the model. This is the raw material for explaining a
prediction: "your risk is driven mostly by exposure/lifestyle scores —
smoking, alcohol, air pollution, diet, occupational hazards."

---

## Deliverables (per Issue #12 definition)

- ✔ Permutation importance plot committed: `docs/figures/permutation_importance.png`
  (+ `permutation_importance.csv`)
- ✔ Written interpretation incl. surprises: this document
- ✔ Redundant-pair note + decision: §4 (pairs are *not* duplicates in this
  data; keep columns; BMI / Physical_Activity_Level documented as drop
  candidates for a future retrain — no retrain performed here)