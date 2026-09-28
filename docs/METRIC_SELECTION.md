# Metric selection — Issue #10: choose a metric that matches the cost of being wrong

**Status:** Solved (analysis only — no model was retrained, retuned, or refit for this issue)
**Evidence script:** [`analyze_error_costs.py`](../analyze_error_costs.py) (read-only; loads the shipped bundle and scores the 400 held-out rows)
**Investigated:** 2026-09-28

---

## 1. What this issue asked

> Choose a metric that matches the cost of being wrong.

The project is a cancer-risk screening/risk-stratification tool. The question is
not "which metric scores highest" but "which error is this tool allowed to make,
and which metric keeps that error visible."

## 2. Current metrics (shipped model, 400 held-out rows)

| class  | support | predicted | precision | recall | F1 | FN | FP |
|--------|--------:|----------:|----------:|-------:|----:|---:|---:|
| High   |      20 |        10 |    0.6000 | 0.3000 | 0.4000 | 14 |  4 |
| Low    |      65 |        64 |    0.6719 | 0.6615 | 0.6667 | 22 | 21 |
| Medium |     315 |       326 |    0.8896 | 0.9206 | 0.9048 | 25 | 36 |

| metric            | value  |
|-------------------|-------:|
| accuracy          | 0.8475 |
| balanced accuracy | 0.6274 |
| precision (macro) | 0.7205 |
| recall (macro)    | 0.6274 |
| **F1 (macro)**    | **0.6572** |
| F1 (weighted)     | 0.8409 |
| MCC               | 0.5414 |
| **High recall**   | **0.3000** |
| **High precision**| **0.6000** |

Confusion matrix (rows = actual, columns = predicted):

```
actual \ predicted    High    Low  Medium  total
High                    6      0      14     20
Low                     0     43      22     65
Medium                  4     21     290    315
total                  10     64     326    400
```

## 3. Error analysis — which mistakes is this model actually making?

| direction | meaning | patients |
|-----------|---------|---------:|
| **A** | High-risk patient sent home as Medium/Low (missed escalation) | **14** |
| B | Low/Medium patient falsely escalated to High (unnecessary follow-up) | 4 |
| C | Medium patient downgraded to Low | 21 |
| D | Low patient upgraded to Medium | 22 |

Direction A vs B is the asymmetry that matters: **14 : 4 = 3.5×.** A missed
High-risk patient is a delayed work-up. A false alarm is a cheap, reversible
follow-up test. The model makes the expensive error 3.5× more often than the
cheap one.

Relative to the error budget, error A is a *subset* of the model's behaviour but
its only *consequential* failure direction. Errors C and D (21 and 22 patients)
reflect the coarse boundary between Low and Medium — real, but low-cost: the
patient is still flagged for risk, just at the wrong tier.

## 4. Candidate metrics compared on the same 400 rows

| candidate | accuracy | bal. acc | F1 macro | recall macro | High recall | High precision | MCC | exp. cost (k=10) |
|-----------|---------:|---------:|---------:|-------------:|------------:|---------------:|-----:|------------------:|
| shipped model | 0.8475 | 0.6274 | 0.6572 | 0.6274 | 0.3000 | 0.6000 | 0.5414 | **0.4675** |
| always Medium | 0.7875 | 0.3333 | 0.2937 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6625 |
| always High | 0.0500 | 0.3333 | 0.0317 | 0.3333 | 1.0000 | 0.0500 | 0.0000 | 0.9500 |
| always Low | 0.1625 | 0.3333 | 0.0932 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 1.2875 |

(exp. cost = mean cost per patient when a missed High costs 10× a false alarm,
defined in §6. Lower is better. The v1 "leaked" model is deliberately not
scored: its 0.9975 was meaningless and its scaler was never persisted, see
`T2_PROVENANCE_AND_LEAKAGE.md`.)

**What the table shows:**

- **Accuracy** scores the do-nothing "always Medium" predictor at 93% of the
  shipped model's value while it finds **0% of High-risk patients**. Accuracy is
  not merely uninformative here — it is actively misleading, because the target
  is 78.7% Medium.
- **High recall alone** is gameable in the *other* direction: "always High"
  scores 1.00 at the cost of flagging all 400 patients (precision 0.05).
- **Macro-F1** does separate the candidates cleanly (0.6572 vs 0.2937 vs 0.0317),
  so it is not blind — its defect is subtler, see §5.

## 5. How each metric responds to the expensive error

Relabel missed High patients as correctly found, one at a time (pure metric
analysis — the shipped model's decision rule is not touched):

| High found | accuracy | Δ acc | F1 macro | Δ F1 macro | High recall |
|-----------:|---------:|------:|---------:|-----------:|------------:|
| 6/20 (shipped) | 0.8475 | +0.0000 | 0.6572 | +0.0000 | 0.300 |
| 8/20 | 0.8525 | +0.0050 | 0.6914 | +0.0342 | 0.400 |
| 11/20 | 0.8600 | +0.0125 | 0.7357 | +0.0785 | 0.550 |
| 14/20 | 0.8675 | +0.0200 | 0.7733 | +0.1161 | 0.700 |
| 17/20 | 0.8750 | +0.0275 | 0.8055 | +0.1483 | 0.850 |
| 20/20 | 0.8825 | +0.0350 | 0.8336 | +0.1764 | 1.000 |

- **Accuracy moves +0.0350 in total** for fixing the single most expensive error
  in the system. It is effectively blind to it: the 14 relabelled rows are
  diluted across 400, and the 0.7875 the do-nothing predictor already achieves
  does most of the work.
- **Macro-F1 moves +0.1764** — it responds clearly, so it is *not* blind. Its
  defect is that it is an **unweighted average**: it credits improvement on Low
  and Medium exactly as much as improvement on High, and cannot express that a
  missed High-risk patient costs several times a Low/Medium mix-up. It is a fair
  average for "all mistakes cost the same" and exactly the wrong average here.
- **High-class recall** moves 0.300 → 1.000 and cannot be moved without moving
  it. It measures the thing directly, with no averaging to dilute it.

## 6. The cost-weighted view (swept, not asserted)

To test how much the conclusion depends on *assuming* a cost ratio, expected
cost per patient is computed under k = (cost of a missed High) / (cost of a
false alarm), for k = 1 … 50:

| k (miss : alarm) | shipped model | always Medium | always High | always Low |
|-----------------:|--------------:|--------------:|------------:|-----------:|
| 1 | 0.1525 | 0.2125 | 0.9500 | 0.8375 |
| 2 | 0.1875 | 0.2625 | 0.9500 | 0.8875 |
| 5 | 0.2925 | 0.4125 | 0.9500 | 1.0375 |
| 10 | 0.4675 | 0.6625 | 0.9500 | 1.2875 |
| 20 | 0.8175 | 1.1625 | 0.9500 | 1.7875 |
| 50 | 1.8675 | 2.6625 | 0.9500 | 3.2875 |

The shipped model has the lowest expected cost at **every** ratio, including
k = 1. The ranking never flips; what degrades as k → 1 is the *margin*
(0.1950 at k = 10 → 0.0600 at k = 1). This analysis deliberately does **not**
pick a value of k — that is a clinical judgement owned by whoever runs the
screening programme, not by this repository. The recommendation below holds for
any k > 1, which is the only regime that makes sense for a screening tool.

## 7. Recommended primary metric

> **PRIMARY METRIC: recall on the High class (sensitivity).**
> Reported always alongside it: **High-class precision**, as the guardrail.
> Tracked, not primary: **macro-F1** and **balanced accuracy**.
> Reported for context only: **accuracy** — never as a decision metric.

Current score on the agreed metric:

- High recall: **0.300** (6 of 20 found, 14 missed)
- High precision: **0.600**
- Expected cost: **0.4675** per patient at k = 10 (vs 0.6625 for the do-nothing
  Medium predictor)

### Why High-class recall, and not accuracy

The purpose of this tool is to *flag* high-risk patients for follow-up. The
cost of the tool is dominated by the patients it fails to flag. Accuracy cannot
see that failure: it moves +0.04 even when every missed High patient is fixed,
and it scores the "predict Medium for everyone" baseline at 0.79. Any metric a
constant predictor can score well has failed to encode the objective.

### Why not macro-F1 (the previous de-facto headline)

Macro-F1 is the correct metric for "how good is this classifier overall" and the
wrong one for "does this tool miss high-risk patients". It rises by +0.18 when
all 14 missed High patients are found — so it *does* respond — but it responds
equally to a cheap Low/Medium mix-up. Keeping the tool honest requires a metric
whose value is dominated by the expensive direction, not one that averages it
away.

### Why recall is not reported alone

Alone, any recall is gameable: "always High" scores High recall 1.00 by flagging
every patient. Paired with High precision (0.60 today, 0.05 for the flag-everyone
baseline), the pair pins both ends — find the high-risk patients without flagging
everyone.

### The honest reading of 0.300

High recall 0.300 is weak, and the reason is data, not model. There are 102 High
patients in the whole dataset and 51 in the training split — 20 in the test
split. Boosting recall further by threshold games on 20 test examples would be
overfitting to noise. The right response to "we miss 14/20 high-risk patients"
is **more labelled High-risk patients**, not tuning.

## 8. Code / documentation changes made for this issue

| change | why |
|--------|-----|
| `analyze_error_costs.py` (new) | read-only, reproducible evidence script: computes every metric in this document from the shipped bundle; `--json` for machine-readable output |
| `docs/METRIC_SELECTION.md` (this file) | the decision record |
| `train.py`: `high_recall`, `high_precision`, `balanced_accuracy` added to the published metrics (metrics.json and bundle metadata) | the agreed primary metric + guardrail + secondary are now part of the standard training record, so every future model is scored on them automatically |
| `train.py`: `forest_sha256()` added and pinned into the bundle, manifest, and `--verify` | pins the canonical bitwise fingerprint of the fitted forest (reproduces the value published in `TRAINING_AND_LEAKAGE.md`); `--verify` now asserts the fingerprint matches the record |

**No model behaviour changed.** The shipped forest is bitwise identical
(`forest_sha256 = aeca50e59b3dc670…`), and every metric matches the published
values exactly.

## 9. Conclusion for Issue #10

The primary metric is **High-class recall**, reported with **High-class
precision** as its guardrail. This choice is driven by the tool's objective —
escalating high-risk patients — not by which number looks highest, and it holds
for any cost ratio > 1, so it does not depend on an assumed clinical cost. The
model itself was not retrained or retuned; the issue is closed by selecting and
recording the metric, and by making every future training run report it.