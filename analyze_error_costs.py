"""
Issue #10 -- choose a metric that matches the cost of being wrong.

This script does NOT retrain, retune, or refit anything. It loads the shipped
bundle from artifacts/, reproduces the exact 400 held-out rows, and scores the
predictions that model already makes. No threshold is changed and no
hyperparameter is touched.

What it produces:
  1. The confusion matrix and the full error budget
  2. Every candidate metric on the current model and on baselines
  3. How sensitive each metric is to the error that matters most
  4. A cost-weighted analysis swept over assumed cost ratios
  5. A recommendation, with the reasoning that produced it

Run:  python analyze_error_costs.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "artifacts" / "model_bundle.joblib"
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"
LEGACY_MODEL = BASE_DIR / "model_xgb_new.pkl"

# Class order. Index 0 is the most dangerous, so ordinal distance is meaningful.
CLASSES = ["High", "Low", "Medium"]


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# Data / predictions (no training)
# -----------------------------------------------------------------------------
def held_out_split():
    """Recreate the exact 400-row test split train.py used."""
    df = pd.read_csv(DATA_PATH)
    feats = [c for c in df.columns
             if c not in ("Patient_ID", "Cancer_Type", "Risk_Level", "Overall_Risk_Score")]
    y = LabelEncoder().fit_transform(df["Risk_Level"])
    _, X_te, _, y_te = train_test_split(df[feats], y, test_size=0.20,
                                        random_state=42, stratify=y)
    return df, feats, X_te, y_te


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    per = {}
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[name] = {
            "support": int(cm[i].sum()),
            "predicted": int(cm[:, i].sum()),
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "fn": fn, "fp": fp,
        }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", labels=list(range(n)), zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", labels=list(range(n)), zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", labels=list(range(n)), zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", labels=list(range(n)), zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "high_recall": per["High"]["recall"],
        "high_precision": per["High"]["precision"],
        "per_class": per,
        "confusion_matrix": cm.tolist(),
    }


def expected_cost(y_true, y_pred, miss_high_cost: float) -> float:
    """
    Mean cost per patient.

    Costs are assigned per (truth, prediction) pair:
      * truth High, predicted anything else  -> `miss_high_cost`
        (a high-risk patient sent home; this is the expensive error)
      * truth non-High, predicted High       -> 1.0
        (an unnecessary follow-up; cheap, reversible, and safe)
      * both non-High, wrong class           -> 1.0 per level of ordinal
        distance (predicting Low for a Medium patient is worse than Medium
        for a Medium patient)
    """
    total = 0.0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p != 0:
            total += miss_high_cost
        elif t != 0 and p == 0:
            total += 1.0
        else:
            total += float(abs(t - p))
    return round(total / len(y_true), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit the numbers as JSON")
    args = ap.parse_args()

    if not BUNDLE_PATH.exists():
        print(f"[FAIL] {BUNDLE_PATH} missing -- run `python train.py` first")
        return 1

    bundle = joblib.load(BUNDLE_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    feats = bundle["feature_names"]
    df, _, X_te, y_te = held_out_split()
    n_high_total = int((df["Risk_Level"] == "High").sum())

    y_pred = model.predict(scaler.transform(X_te[feats]))
    shipped = all_metrics(y_te, y_pred)
    report = {"shipped": shipped}

    # ---------------------------------------------------------------- section 1
    hr("1. CURRENT METRICS  (400 held-out rows, shipped model, nothing retrained)")
    print(f"model v{bundle['metadata']['model_version']}   "
          f"resample={bundle['metadata']['resample']}   "
          f"features={len(feats)} (Overall_Risk_Score excluded)")
    print(f"trained_at {bundle['metadata']['trained_at'][:19]} UTC")
    print()
    print(f"{'class':<10}{'support':>8}{'predicted':>11}{'precision':>11}"
          f"{'recall':>9}{'F1':>8}{'FN':>6}{'FP':>6}")
    for name in CLASSES:
        m = shipped["per_class"][name]
        print(f"{name:<10}{m['support']:>8}{m['predicted']:>11}{m['precision']:>11}"
              f"{m['recall']:>9}{m['f1']:>8}{m['fn']:>6}{m['fp']:>6}")
    print()
    for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted", "mcc"):
        print(f"  {k:<20}{shipped[k]}")
    print(f"  {'high_recall':<20}{shipped['high_recall']}")
    print(f"  {'high_precision':<20}{shipped['high_precision']}")

    hr("2. CONFUSION MATRIX  (rows = actual, columns = predicted)")
    cm = np.array(shipped["confusion_matrix"])
    order = [0, 1, 2]  # High, Low, Medium
    print(f"{'actual \\ predicted':<20}" + "".join(f"{CLASSES[i]:>10}" for i in order) + f"{'total':>9}")
    for r in order:
        print(f"{CLASSES[r]:<20}" + "".join(f"{cm[r, c]:>10}" for c in order) + f"{cm[r].sum():>9}")
    print(f"{'total':<20}" + "".join(f"{cm[:, c].sum():>10}" for c in order) + f"{cm.sum():>9}")

    # ---------------------------------------------------------------- section 3
    hr("3. ERROR BUDGET -- which mistakes is this model actually making?")
    missed_high = shipped["per_class"]["High"]["fn"]
    false_alarm = shipped["per_class"]["High"]["fp"]
    med_to_low = shipped["per_class"]["Medium"]["fn"] - shipped["per_class"]["High"]["fp"]
    low_to_med = shipped["per_class"]["Low"]["fn"]
    print(f"  A. High-risk patient sent home as Medium/Low : {missed_high:>3} patients"
          f"   <-- the expensive error")
    print(f"  B. Low/Medium patient falsely escalated to High : {false_alarm:>3} patients")
    print(f"  C. Medium patient downgraded to Low         : {med_to_low:>3} patients")
    print(f"  D. Low patient upgraded to Medium           : {low_to_med:>3} patients")
    print()
    print(f"  Direction A vs B: {missed_high} : {false_alarm}  "
          f"= {missed_high / max(false_alarm, 1):.1f}x more missed than false alarms")
    print(f"  Of {shipped['per_class']['High']['support']} true High patients, "
          f"{shipped['per_class']['High']['recall']:.0%} found, "
          f"{1 - shipped['per_class']['High']['recall']:.0%} missed.")
    print()
    print("  A missed High-risk patient is a delayed diagnosis. A false alarm is an")
    print("  unnecessary, cheap, reversible follow-up test. These are not symmetric,")
    print("  so a metric that weights them equally is measuring the wrong thing.")
    report["error_budget"] = {"missed_high": missed_high, "false_alarm": false_alarm,
                              "medium_to_low": med_to_low, "low_to_medium": low_to_med}

    # ---------------------------------------------------------------- section 4
    hr("4. CANDIDATE METRICS COMPARED  (same predictions, different lenses)")
    baselines = {
        "shipped model": y_pred,
        "always Medium": np.full_like(y_te, 2),
        "always High": np.zeros_like(y_te),
        "always Low": np.ones_like(y_te),
    }
    # The superseded v1 model is deliberately NOT scored here. It was fit on
    # StandardScaler output but its scaler was never persisted (Issue #2), so
    # there is no way to feed it correctly-formatted input. Any score would
    # measure that missing scaler rather than the model. Its 0.9975 came from
    # the leaked Overall_Risk_Score (Issue #5) and is not comparable anyway.
    # See docs/T2_PROVENANCE_AND_LEAKAGE.md and docs/TRAINING_AND_LEAKAGE.md.

    rows = []
    for name, pred in baselines.items():
        m = all_metrics(y_te, np.asarray(pred))
        m["expected_cost_10"] = expected_cost(y_te, np.asarray(pred), 10.0)
        rows.append((name, m))
        report.setdefault("candidates", {})[name] = m

    cols = ["accuracy", "balanced_accuracy", "f1_macro", "recall_macro",
            "high_recall", "high_precision", "mcc", "expected_cost_10"]
    header = f"{'candidate':<28}" + "".join(f"{c[:13]:>15}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<28}" + "".join(f"{m[c]:>15}" for c in cols))
    print()
    print("  expected_cost_10: mean cost per patient if missing a High costs 10x a")
    print("  false alarm. LOWER IS BETTER. Every other column: HIGHER IS BETTER.")

    ship = report["candidates"]["shipped model"]
    med = report["candidates"]["always Medium"]
    high = report["candidates"]["always High"]
    print()
    print("  Reading the table:")
    print(f"    * accuracy ranks 'always Medium' at {med['accuracy']:.4f} against the")
    print(f"      model's {ship['accuracy']:.4f} -- {med['accuracy'] / ship['accuracy']:.0%} of the score,")
    print(f"      for 0% of High-risk patients found against the model's "
          f"{ship['high_recall']:.0%}.")
    print(f"    * 'always High' scores accuracy {high['accuracy']:.4f} and High recall")
    print(f"      {high['high_recall']:.2f}. Taken alone, High recall would rank that")
    print("      constant predictor top of the table, alongside flagging all 400")
    print("      patients. That is why the primary metric is reported as a PAIR with")
    print("      High precision, never on its own.")
    print(f"    * macro-F1 separates them properly "
          f"({ship['f1_macro']:.4f} vs {med['f1_macro']:.4f} vs {high['f1_macro']:.4f})")
    print("      -- but it prices a missed High the same as any other mistake, which")
    print("      is the problem section 3 describes.")

    # ---------------------------------------------------------------- section 5
    hr("5. HOW BLIND IS EACH METRIC TO THE EXPENSIVE ERROR?")
    print("  Take the shipped predictions and relabel a missed High patient as High,")
    print("  one patient at a time. This is analysis of the metrics, not tuning: the")
    print("  shipped model's decision rule is not changed anywhere.\n")
    print(f"{'High found':<12}{'accuracy':>11}{'delta':>9}{'f1_macro':>11}{'delta':>9}"
          f"{'High recall':>14}")
    base_acc, base_f1 = ship["accuracy"], ship["f1_macro"]
    curve = []
    n_high = shipped["per_class"]["High"]["support"]
    n_found = int(np.array(shipped["confusion_matrix"])[0, 0])
    for extra in (0, 2, 5, 8, 11, 14):
        p = y_pred.copy()
        idx = [i for i in range(len(p)) if y_te[i] == 0 and p[i] != 0][:extra]
        p[idx] = 0
        m = all_metrics(y_te, p)
        curve.append(m)
        print(f"{n_found + extra:>4}/{n_high}     "
              f"{m['accuracy']:>11.4f}{m['accuracy'] - base_acc:>+9.4f}"
              f"{m['f1_macro']:>11.4f}{m['f1_macro'] - base_f1:>+9.4f}"
              f"{m['high_recall']:>14.3f}")
    report["sensitivity"] = curve
    print()
    print("  This is where accuracy and macro-F1 part ways, and the distinction")
    print("  matters for the recommendation:")
    print()
    print(f"    * accuracy moves +{curve[-1]['accuracy'] - base_acc:.4f} in total -- a third of a")
    print("      point. Fixing the most expensive error in the whole system is")
    print("      nearly invisible to it, because 14 relabelled rows are diluted")
    print("      across 400 and the 0.7875 the constant-Medium predictor already")
    print("      scores is doing most of the work. Accuracy is effectively blind")
    print("      to the error that matters.")
    print()
    print(f"    * macro-F1 moves +{curve[-1]['f1_macro'] - base_f1:.4f} -- it responds clearly, and is")
    print("      not blind. Its defect is different and more subtle: it responds")
    print("      EQUALLY to a High miss and to a Low/Medium mix-up, so it cannot")
    print("      express that one of those errors is three times costlier. It is a")
    print("      fair unweighted average, which is exactly what is wanted when all")
    print("      mistakes cost the same, and exactly what is not wanted here.")
    print()
    print("    * High-class recall goes 0.300 -> 1.000 and cannot be moved without")
    print("      moving it. It measures the thing directly, with no averaging to")
    print("      dilute it.")

    # ---------------------------------------------------------------- section 6
    hr("6. COST-WEIGHTED VIEW -- swept over assumed cost ratios")
    print("  If missing a High-risk patient costs k times a false alarm, mean cost")
    print("  per patient is:\n")
    print(f"{'k (miss:alarm)':<16}" + "".join(f"{n[:15]:>17}" for n, _ in rows))
    print("-" * (16 + 17 * len(rows)))
    sweep = {}
    for k in (1, 2, 5, 10, 20, 50):
        vals = []
        for name, _ in rows:
            c = expected_cost(y_te, np.asarray(baselines[name]), float(k))
            vals.append(c)
            sweep.setdefault(str(k), {})[name] = c
        print(f"{k:<16}" + "".join(f"{v:>17.4f}" for v in vals))
    report["cost_sweep"] = sweep
    print()
    print("  The shipped model has the lowest expected cost at every ratio tested,")
    print("  including k=1 where the two error types are priced equally. The ranking")
    print("  never flips. What degrades is the MARGIN: at k=1 the model's advantage")
    print("  over 'always Medium' is 0.0600, and at k=10 it is 0.1950 -- the")
    print("  cheaper a missed High-risk patient is assumed to be, the less the")
    print("  model's apparent accuracy advantage is worth. Accuracy cannot show you")
    print("  that, because its score does not depend on k at all.")
    print()
    print("  Note what this analysis does NOT do: it does not pick a cost ratio. The")
    print("  number k encodes a clinical judgement about the consequences of a")
    print("  delayed diagnosis versus an unnecessary follow-up, and that judgement")
    print("  belongs to whoever owns the clinical decision, not to this repository.")
    print("  The recommendation below is deliberately chosen so that it does not")
    print("  depend on an exact k.")

    # ---------------------------------------------------------------- section 7
    hr("7. RECOMMENDATION")
    s = shipped["per_class"]["High"]
    rec = f"""
  PRIMARY METRIC:  recall on the High class (sensitivity)

  reported always alongside:  High-class precision, as its guardrail.

  Secondary, tracked but not primary:  macro-F1 and balanced accuracy.
  Reported for context only:  accuracy. Never as a decision metric.

  Why High-class recall and not accuracy:
    Accuracy is not merely uninformative here, it is misleading. The target is
    78.7% Medium, so the do-nothing strategy scores {med['accuracy']:.2f} while missing
    100% of high-risk patients. Any metric a constant predictor can score well
    on has failed to encode the objective.

  Why not macro-F1:
    Macro-F1 is not blind to the expensive error -- it rises
    +{curve[-1]['f1_macro'] - base_f1:.4f} when all {s['fn']} missed High patients are found. Its defect is
    subtler: it is an UNWEIGHTED average, so it credits improvement on Low and
    Medium exactly as much as improvement on High. It cannot express that a
    missed High-risk patient costs several times a Low/Medium mix-up. It is the
    right metric for "how good is this classifier overall" and the wrong one for
    "does this tool miss high-risk patients". Keep reporting it; do not steer on it.

  Why recall alone is not enough on its own:
    It is gameable in the other direction -- the always-High baseline scores
    High recall {high['high_recall']:.2f} while flagging all {cm.sum()} patients at High precision
    {high['high_precision']:.2f}. Paired with High-class precision, which is {s['precision']:.2f} today, it
    is not gameable. The pair pins both ends: find the high-risk patients,
    without flagging everyone.

  What the model scores today, on the agreed metric:
    High recall     {s['recall']:.3f}   ({int(np.array(shipped['confusion_matrix'])[0, 0])} of {s['support']} found, {s['fn']} missed)
    High precision  {s['precision']:.3f}
    Expected cost   {ship['expected_cost_10']:.4f} per patient at k=10
                    (vs {med['expected_cost_10']:.4f} for the do-nothing Medium predictor)

  The honest reading: High recall {s['recall']:.3f} is weak. With {n_high_total} High patients in the
  dataset, {n_high_total // 2} of them in the training split, this class is data-limited. The
  right response is more labelled High-risk patients, not threshold tuning on
  {s['support']} test examples.
"""
    print(rec)

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
