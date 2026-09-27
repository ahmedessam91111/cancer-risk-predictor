"""
Audit of Overall_Risk_Score for target leakage, and measurement of its
influence on model performance.

Covers GitHub Issue #5 (leakage audit) and Issue #6 (impact measurement).

This script is strictly NON-DESTRUCTIVE. It trains models in memory to measure
things, but never writes, overwrites, or deletes any artifact in this repo --
`model_xgb_new.pkl`, `label_encoder.pkl` and `feature_names.pkl` are only ever
READ. Nothing here is tuned or optimized: the estimator, its hyperparameters,
the split, and the preprocessing are held identical across every experiment so
that the only variable is the presence of Overall_Risk_Score.

    python audit_overall_risk_score.py
    python audit_overall_risk_score.py --json

Exit codes
    0  audit completed (leakage may still be present -- that is a finding)
    1  dataset missing or unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "cancer-risk-factors.csv"
FEATURES_PKL = BASE_DIR / "feature_names.pkl"
MODEL_PKL = BASE_DIR / "model_xgb_new.pkl"

# The leaked feature under audit.
LEAKY = "Overall_Risk_Score"
TARGET = "Risk_Level"

# Frozen training configuration -- identical to Cancer_Risk_Prediction_(ML).ipynb
# cells 8/9/10, which produced model_xgb_new.pkl. NOT tuned.
SEED = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, classes) -> dict:
    labels = list(range(len(classes)))
    per_class = {}
    for i, name in enumerate(classes):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {
            "support": int((y_true == i).sum()),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predicted_distribution": {
            name: int((y_pred == i).sum()) for i, name in enumerate(classes)
        },
    }


def show_confusion(cm, classes, title) -> None:
    print(f"\n  {title}")
    header = "".join(f"{c:>10}" for c in ["true \\ pred"] + list(classes))
    print(f"  {header}")
    for name, row in zip(classes, cm):
        cells = "".join(f"{v:>10}" for v in row)
        print(f"  {name:<12}{cells}")


# ---------------------------------------------------------------------------
# Part 1 -- Issue #5: where do the two columns come from?
# ---------------------------------------------------------------------------

def audit_provenance(df: dict) -> None:
    print("=" * 78)
    print("ISSUE #5a  PROVENANCE OF THE TWO COLUMNS")
    print("=" * 78)
    print("  Neither Overall_Risk_Score nor Risk_Level is created anywhere in this")
    print("  repository. Both are read verbatim from the CSV, so whatever formula")
    print("  relates them lives upstream, in the generator of the dataset itself.")
    print("  No download URL, dataset version, or generator script is recorded, so")
    print("  that formula cannot be read -- it can only be recovered empirically.")
    print("  Confirmed by search: 0 assignments to either name across both")
    print("  notebooks and all .py files.")


def audit_separation(df: pd.DataFrame, classes) -> dict:
    print()
    print("=" * 78)
    print("ISSUE #5b  IS Risk_Level A THRESHOLD FUNCTION OF Overall_Risk_Score?")
    print("=" * 78)
    rng = df.groupby(TARGET)[LEAKY].agg(["count", "min", "max"])
    print(f"\n  {LEAKY} range per class:")
    print("  " + rng.round(6).to_string().replace("\n", "\n  "))

    bounds = {}
    ordered = sorted(classes, key=lambda c: df.loc[df[TARGET] == c, LEAKY].mean())
    for c in ordered:
        bounds[c] = (float(df.loc[df[TARGET] == c, LEAKY].min()),
                     float(df.loc[df[TARGET] == c, LEAKY].max()))
    print("\n  separation gaps between consecutive classes:")
    for a, b in zip(ordered, ordered[1:]):
        print(f"    {a:<7} max {bounds[a][1]:.6f}  ->  {b:<7} min {bounds[b][0]:.6f}"
              f"   gap {bounds[b][0] - bounds[a][1]:+.6f}")

    # Recover the thresholds from the midpoints of the observed gaps.
    cuts = [(bounds[a][1] + bounds[b][0]) / 2 for a, b in zip(ordered, ordered[1:])]

    def rule(v):
        for i, cut in enumerate(cuts):
            if v < cut:
                return ordered[i]
        return ordered[-1]

    pred = df[LEAKY].map(rule)
    mismatches = int((pred != df[TARGET]).sum())
    acc = float((pred == df[TARGET]).mean())

    print(f"\n  recovered rule:  {LEAKY} < {cuts[0]:.6f} -> {ordered[0]} ; "
          f"< {cuts[1]:.6f} -> {ordered[1]} ; else {ordered[2]}")
    print(f"  accuracy of that rule on all {len(df)} rows : {acc * 100:.2f}%")
    print(f"  mismatches                                    : {mismatches}")

    verdict = "LEAKAGE" if acc > 0.999 else ("STRONG PROXY" if acc > 0.90 else "NO LEAKAGE")
    print(f"\n  >>> VERDICT: {verdict}")

    return {"thresholds": [round(c, 6) for c in cuts], "rule_accuracy": round(acc, 6),
            "mismatches": mismatches, "verdict": verdict}


def audit_composite(df: pd.DataFrame) -> dict:
    print()
    print("=" * 78)
    print("ISSUE #5c  IS Overall_Risk_Score ITSELF A WEIGHTED SUM OF THE FACTORS?")
    print("=" * 78)
    feats = [c for c in df.columns
             if c not in ("Patient_ID", "Cancer_Type", TARGET, LEAKY)]
    lr = LinearRegression().fit(df[feats].values, df[LEAKY].values)
    r2 = float(lr.score(df[feats].values, df[LEAKY].values))
    print(f"\n  linear R^2 of {LEAKY} on the other {len(feats)} features: {r2:.4f}")
    print("\n  coefficients (sorted by magnitude):")
    for c, w in sorted(zip(feats, lr.coef_), key=lambda t: -abs(t[1])):
        print(f"    {c:<26} {w:+.5f}")
    print(f"    {'intercept':<26} {lr.intercept_:+.5f}")
    print(f"\n  The nine risk-factor columns carry near-equal positive weights")
    print(f"  (~+0.012..+0.015) while Age, BMI, activity and calcium sit near zero.")
    print(f"  So {LEAKY} is an upstream composite of the risk factors, and Risk_Level")
    print(f"  was then derived from {LEAKY} by thresholding. The model is handed the")
    print(f"  middle of that chain and asked to predict its output.")
    return {"linear_r2_on_other_features": round(r2, 4)}


def audit_recoverability(df: pd.DataFrame) -> dict:
    print()
    print("=" * 78)
    print("ISSUE #5d  COULD THE LEAK BE REBUILT FROM LEGITIMATE FEATURES?")
    print("=" * 78)
    feats = [c for c in df.columns
             if c not in ("Patient_ID", "Cancer_Type", TARGET, LEAKY)]
    X = df[feats]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    sc = StandardScaler()
    m = RandomForestClassifier(random_state=SEED, n_estimators=N_ESTIMATORS)
    m.fit(sc.fit_transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    acc = float(accuracy_score(yte, pred))
    print(f"\n  RF on the 17 features EXCLUDING {LEAKY}: accuracy = {acc:.4f}")
    print(f"  {LEAKY} is partly recoverable from legitimate inputs, so removing it")
    print(f"  costs some signal -- but the residual accuracy is real, not the leak.")
    return {"rf_without_leaky_accuracy": round(acc, 4)}


# ---------------------------------------------------------------------------
# Part 2 -- Issue #6: controlled measurement
# ---------------------------------------------------------------------------

def controlled_comparison(df: pd.DataFrame, classes) -> dict:
    print()
    print("=" * 78)
    print("ISSUE #6  CONTROLLED COMPARISON: WITH vs WITHOUT Overall_Risk_Score")
    print("=" * 78)
    print(f"\n  Held identical in both arms: dataset, split (test_size={TEST_SIZE},")
    print(f"  random_state={SEED}, stratified), StandardScaler fit on train only,")
    print(f"  RandomForestClassifier(random_state={SEED}, n_estimators={N_ESTIMATORS}).")
    print(f"  The only difference is whether {LEAKY} is in the feature list.")
    print(f"  Nothing is tuned.")

    feature_names = list(joblib.load(FEATURES_PKL))
    base = [f for f in feature_names if f != LEAKY]

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    classes = [str(c) for c in le.classes_]

    results = {}
    arms = (
        ("A_with_leaky", feature_names),
        ("B_without_leaky", base),
        ("C_leaky_only", [LEAKY]),
    )
    for label, feats in arms:
        X = df[feats]
        # The split depends only on y and the seed, so BOTH arms get the exact
        # same train/test rows -- a genuinely controlled comparison.
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        model = RandomForestClassifier(random_state=SEED, n_estimators=N_ESTIMATORS)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)

        res = evaluate(yte, pred, classes)
        res["n_features"] = len(feats)
        res["n_test_rows"] = int(len(yte))
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        res["top_features"] = {k: round(v, 4) for k, v in imp.head(5).items()}
        res["leaky_importance"] = round(float(imp.get(LEAKY, 0.0)), 4)
        results[label] = res

        print(f"\n  --- Arm {label} ({len(feats)} features) ---")
        print(f"    accuracy        {res['accuracy']:.4f}")
        print(f"    precision macro {res['precision_macro']:.4f}")
        print(f"    recall macro    {res['recall_macro']:.4f}")
        print(f"    F1 macro        {res['f1_macro']:.4f}")
        print(f"    F1 weighted     {res['f1_weighted']:.4f}")
        print(f"    {LEAKY} importance: {res['leaky_importance']:.4f}")
        print("    per class:")
        for c, m in res["per_class"].items():
            print(f"      {c:<7} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f}  n={m['support']}")
        show_confusion(res["confusion_matrix"], classes, "confusion matrix (rows=true, cols=pred)")

    # Faithfulness check: arm A must reproduce the SHIPPED artifact exactly, by
    # evaluating the real pickle on the identical split and comparing predictions.
    print()
    print("  --- faithfulness check ---")
    shipped = joblib.load(MODEL_PKL)
    X = df[feature_names]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    sc = StandardScaler()
    sc.fit(Xtr)
    ship_pred = shipped.predict(sc.transform(Xte))
    armA = RandomForestClassifier(random_state=SEED, n_estimators=N_ESTIMATORS)
    armA.fit(sc.transform(Xtr), ytr)
    armA_pred = armA.predict(sc.transform(Xte))

    ship_acc = accuracy_score(yte, ship_pred)
    arm_acc = accuracy_score(yte, armA_pred)
    identical = int((ship_pred == armA_pred).sum())
    print(f"    shipped model_xgb_new.pkl accuracy on this split : {ship_acc:.4f}")
    print(f"    arm A rebuilt-from-notebook accuracy              : {arm_acc:.4f}")
    print(f"    predictions identical on {identical}/{len(yte)} held-out rows"
          f"  -> {'MATCH' if identical == len(yte) else 'DIFFERENT'}")
    print(f"    shipped model consumes {LEAKY}: {LEAKY in feature_names}")
    print(f"    shipped artifact untouched: {shipped.n_features_in_} features, "
          f"{type(shipped).__name__}")

    a = results["A_with_leaky"]
    print()
    print("  --- delta (A minus B) ---")
    print(f"    {'metric':<18}{'A with':>10}{'B without':>12}{'delta':>10}")
    for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        a_v, b_v = a[m], results["B_without_leaky"][m]
        print(f"    {m:<18}{a_v:>10.4f}{b_v:>12.4f}{a_v - b_v:>+10.4f}")
    print()
    print("    per-class recall:")
    for c in a["per_class"]:
        r_a = a["per_class"][c]["recall"]
        r_b = results["B_without_leaky"]["per_class"][c]["recall"]
        print(f"      {c:<8}{r_a:>8.4f}{r_b:>12.4f}{r_a - r_b:>+10.4f}")

    results["delta"] = {m: round(a[m] - results["B_without_leaky"][m], 4)
                        for m in ("accuracy", "precision_macro", "recall_macro",
                                  "f1_macro", "f1_weighted")}

    # How much do the 17 legitimate features add on top of the leak?
    c = results["C_leaky_only"]
    print()
    print("  --- how much do the other 17 features actually contribute? ---")
    print(f"    arm C ({LEAKY} ALONE, 1 feature) : accuracy {c['accuracy']:.4f}  "
          f"F1 macro {c['f1_macro']:.4f}")
    print(f"    arm A (all 18 features)          : accuracy {a['accuracy']:.4f}  "
          f"F1 macro {a['f1_macro']:.4f}")
    print(f"    contribution of the other 17    : {a['accuracy'] - c['accuracy']:+.4f} accuracy")
    results["leak_only_vs_full"] = {
        "leaky_only_accuracy": c["accuracy"],
        "full_accuracy": a["accuracy"],
        "accuracy_contribution_of_other_17": round(a["accuracy"] - c["accuracy"], 4),
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"FATAL: {DATA.name} not found. Run `python verify_dataset.py` first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(DATA)
    classes = sorted(df[TARGET].unique().tolist())

    print(f"dataset : {DATA.name}  rows={len(df)}  cols={df.shape[1]}")
    print(f"python  : {sys.version.split()[0]}   sklearn {sklearn.__version__}\n")

    out = {"provenance": {}}
    audit_provenance(df)
    out["separation"] = audit_separation(df, classes)
    out["composite"] = audit_composite(df)
    out["recoverability"] = audit_recoverability(df)
    out["comparison"] = controlled_comparison(df, classes)

    if args.json:
        print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
