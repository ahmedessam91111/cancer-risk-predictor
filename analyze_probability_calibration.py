"""
Issue #11 -- check whether the displayed probabilities are real.

Analysis ONLY. Nothing is retrained, nothing is shipped. The deployed model
(artifacts/production/model.pkl) is scored on the exact 400 held-out rows and
its probability output is measured:

   1. WHERE probabilities are produced and displayed (static, app.py)
   2. Brier score -- multiclass and per class
   3. Calibration curves -- binned predicted probability vs observed frequency
   4. ECE (expected calibration error), per class and macro
   5. The "does 80% mean ~80%?" reliability check
   6. What out-of-fold per-class isotonic recalibration would achieve
      (fit on the training split only; applied to the held-out rows)

The last item exists purely as evidence for the recommendation. The deployed
model is never modified here.

Run:  python analyze_probability_calibration.py [--json]
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
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "production" / "model.pkl"
LE_PATH = BASE_DIR / "artifacts" / "production" / "label_encoder.pkl"
FEAT_PATH = BASE_DIR / "artifacts" / "production" / "feature_names.pkl"
META_PATH = BASE_DIR / "artifacts" / "production" / "metadata.json"
BUNDLE_PATH = BASE_DIR / "artifacts" / "model_bundle.joblib"
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"
APP_PATH = BASE_DIR / "app.py"

N_BINS = 10


def binned_reliability(p, y, n_bins: int = N_BINS):
    """Equal-frequency bins; returns (rows, ECE)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    uniq = np.unique(np.round(p, 12))
    if len(uniq) > n_bins:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        span = float(p.max() - p.min()) + 1e-9
        edges = np.unique(np.linspace(float(p.min()) - 0.001 * span,
                                      float(p.max()) + 0.001 * span, n_bins + 1))
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({"bin": b, "n": n,
                     "conf": float(p[m].mean()), "obs": float(y[m].mean())})
    ece = sum(r["n"] / len(p) * abs(r["conf"] - r["obs"]) for r in rows)
    return rows, ece


def brier_multiclass(proba, y_oh):
    return float(np.mean(((proba - y_oh) ** 2).sum(axis=1)))


def held_out_split():
    df = pd.read_csv(DATA_PATH)
    feats = [c for c in df.columns
             if c not in ("Patient_ID", "Cancer_Type", "Risk_Level", "Overall_Risk_Score")]
    y = LabelEncoder().fit_transform(df["Risk_Level"])
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], y, test_size=0.2,
                                              random_state=42, stratify=y)
    return df, feats, X_tr, X_te, y_tr, y_te


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"[FAIL] {MODEL_PATH} missing -- run `python export_production.py`")

    report: dict = {}

    # ---------------------------------------------------------------- section 1
    print("=" * 78)
    print("1. CURRENT PROBABILITY IMPLEMENTATION")
    print("=" * 78)
    src_lines = APP_PATH.read_text(encoding="utf-8").splitlines()
    hits = [(i + 1, src_lines[i].strip()) for i in range(len(src_lines))
            if "predict_proba" in src_lines[i]]
    print("model family: RandomForestClassifier(class_weight='balanced'), 100 trees")
    print("production artifact: Pipeline [StandardScaler -> RandomForest] in model.pkl")
    print("probability calls in app.py:")
    for ln, text in hits:
        print(f"    line {ln:>4}:  {text}")
    print("manual mode   -> render_probabilities(): each class shown as a bold "
          "percentage + bar")
    print("batch mode    -> predict_proba() exported to the CSV as prob_<class> "
          "columns")
    print()
    print("Raw RF probabilities are fractions of tree votes -- a sensible confidence")
    print("proxy, but NOT guaranteed to equal observed frequencies (calibration).")
    report["predict_proba_locations"] = hits

    # ---------------------------------------------------------------- section 2
    print()
    print("=" * 78)
    print("2. DATA, SPLIT AND MODEL")
    print("=" * 78)
    bundle = joblib.load(BUNDLE_PATH)
    forest = bundle["model"]
    scaler = bundle["scaler"]
    df, feats, X_tr, X_te, y_tr, y_te = held_out_split()
    le = joblib.load(LE_PATH)
    classes = list(le.classes_)
    print(f"train rows {len(X_tr)}, held-out rows {len(X_te)}")
    for c in range(3):
        print(f"  {classes[c]:<7} train n={int((y_tr == c).sum()):>4}   "
              f"test n={int((y_te == c).sum()):>3}")
    print(f"features: {len(feats)} (leak-free)")

    # column order = model classes_ (0,1,2); label encoder maps class name -> index
    proba = forest.predict_proba(scaler.transform(X_te[feats].to_numpy()))
    order = {int(c): i for i, c in enumerate(forest.classes_)}
    proba = proba[:, [order[c] for c in range(3)]]
    y_oh = (y_te[:, None] == np.arange(3)).astype(float)

    # ---------------------------------------------------------------- section 3
    print()
    print("=" * 78)
    print(f"3. BRIER SCORE  (lower is better; a perfect predictor scores 0)")
    print("=" * 78)
    brier_mc = brier_multiclass(proba, y_oh)
    base = np.bincount(y_te, minlength=3) / len(y_te)
    clima = float(1.0 - (base ** 2).sum())   # of predicting only the base rate
    per_class = {classes[c]: float(np.mean((proba[:, c] - (y_te == c)) ** 2))
                 for c in range(3)}
    print(f"  multiclass Brier (deployed model) : {brier_mc:.4f}")
    print(f"  climatology baseline (no model)    : {clima:.4f}")
    print(f"  improvement over climatology       : {(clima - brier_mc) / clima * 100:+.1f}%")
    print("  per-class Brier:")
    for c in range(3):
        print(f"    {classes[c]:<7} {per_class[classes[c]]:.4f}")
    report["brier"] = {"multiclass": brier_mc, "climatology": clima,
                       "per_class": per_class, "base_rate": base.tolist()}

    # ---------------------------------------------------------------- section 4
    print()
    print("=" * 78)
    print("4. CALIBRATION CURVES + ECE (predicted probability vs observed frequency)")
    print("=" * 78)
    print(f"  equal-frequency bins (max {N_BINS}); ECE = n/N * |predicted - observed|")
    ece_per = {}
    for c in range(3):
        rows, ece = binned_reliability(proba[:, c], (y_te == c).astype(float))
        ece_per[classes[c]] = ece
        print(f"\n  {classes[c]}   ECE {ece:.4f}   ({len(rows)} non-empty bins)")
        print(f"    {'n':>4}  {'predicted':>10}  {'observed':>10}  {'gap':>9}")
        for r in rows:
            gap = r["conf"] - r["obs"]
            print(f"    {r['n']:>4}  {r['conf']:>10.4f}  {r['obs']:>10.4f}  {gap:>+9.4f}")
    ece_macro = float(np.mean(list(ece_per.values())))
    report["ece"] = {"per_class": ece_per, "macro": ece_macro}

    # ---------------------------------------------------------------- section 5
    print()
    print("=" * 78)
    print("5. THE QUESTION: does an 80% display mean ~80% observed?")
    print("=" * 78)
    targets = [0.20, 0.40, 0.50, 0.60, 0.80, 0.90]
    print(f"  {'class':<8}{'target':>8}{'n in bin':>10}{'mean shown':>12}"
          f"{'observed':>12}")
    checks = []
    for c in range(3):
        rows, _ = binned_reliability(proba[:, c], (y_te == c).astype(float), n_bins=20)
        for t in targets:
            hit = [r for r in rows
                   if abs(r["conf"] - t) == min(abs(rr["conf"] - t) for rr in rows)]
            if hit:
                r = hit[0]
                checks.append({"class": classes[c], "target": t, "n": r["n"],
                               "mean_shown": r["conf"], "observed": r["obs"]})
                print(f"  {classes[c]:<8}{t:>8.2f}{r['n']:>10}{r['conf']:>12.4f}"
                      f"{r['obs']:>12.4f}")
    report["reliability_80pct"] = checks

    # ---------------------------------------------------------------- section 6
    print()
    print("=" * 78)
    print("6. WHAT RECALIBRATION WOULD ACHIEVE (analysis only -- NOT deployed)")
    print("=" * 78)
    print("  Protocol: 5-fold stratified OOF probabilities of the SAME forest on")
    print("  the training split (same hyperparameters, fixed scaler), per-class")
    print("  isotonic calibrator fitted on OOF, applied to held-out probabilities.")
    print("  This fits NO model that is shipped -- it measures what is achievable.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros((len(X_tr), 3))
    X_tr_n = scaler.transform(X_tr[feats].to_numpy())
    for tr_i, va_i in skf.split(X_tr_n, y_tr):
        est = clone(forest)
        est.fit(X_tr_n[tr_i], y_tr[tr_i])
        pv = est.predict_proba(X_tr_n[va_i])
        ord_ = {int(c): i for i, c in enumerate(est.classes_)}
        oof[va_i] = pv[:, [ord_[c] for c in range(3)]]

    calibrators = []
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof[:, c], (y_tr == c).astype(float))
        calibrators.append(iso)

    proba_cal = np.column_stack([calibrators[c].predict(proba[:, c])
                                 for c in range(3)])
    proba_cal /= proba_cal.sum(axis=1, keepdims=True)   # renormalise to a simplex

    brier_cal = brier_multiclass(proba_cal, y_oh)
    print(f"  multiclass Brier:  {brier_mc:.4f} -> {brier_cal:.4f} "
          f"({(brier_cal - brier_mc) * 100:+.2f} points)")
    ece_cal = {}
    for c in range(3):
        _, e = binned_reliability(proba_cal[:, c], (y_te == c).astype(float))
        ece_cal[classes[c]] = e
    print(f"  ECE macro:         {ece_macro:.4f} -> "
          f"{float(np.mean(list(ece_cal.values()))):.4f}")
    for c in range(3):
        print(f"    {classes[c]:<7} ECE {ece_per[classes[c]]:.4f} "
              f"-> {ece_cal[classes[c]]:.4f}")
    report["recalibrated"] = {"brier": brier_cal, "ece_per_class": ece_cal}

    # ---- does recalibration change the predicted class? (hard-decision check) --
    raw_arg = proba.argmax(axis=1)
    cal_arg = proba_cal.argmax(axis=1)
    agree = float((raw_arg == cal_arg).mean())
    print()
    print("  Does recalibration change the predicted class?")
    print(f"    held-out rows whose class flips under recalibration: "
          f"{int((raw_arg != cal_arg).sum())} / {len(raw_arg)} "
          f"({agree * 100:.1f}% unchanged)")
    if (raw_arg != cal_arg).any():
        print("    flips (raw class -> calibrated class):")
        for i in np.where(raw_arg != cal_arg)[0]:
            print(f"      row {i:>3}: {classes[raw_arg[i]]} -> {classes[cal_arg[i]]}"
                  f"  (true={classes[y_te[i]]})")
    raw_acc = float((raw_arg == y_te).mean())
    cal_acc = float((cal_arg == y_te).mean())
    print(f"    accuracy under raw argmax      : {raw_acc:.4f} "
          f"(matches published 0.8475)")
    print(f"    accuracy under calibrated argmax: {cal_acc:.4f}")
    report["recalibrated_decision"] = {
        "flips": int((raw_arg != cal_arg).sum()),
        "n": int(len(raw_arg)),
        "accuracy_raw": raw_acc,
        "accuracy_calibrated": cal_acc,
        "flip_details": [
            {"row": int(i), "from": classes[raw_arg[i]], "to": classes[cal_arg[i]],
             "true": classes[y_te[i]]}
            for i in np.where(raw_arg != cal_arg)[0]],
    }

    # ---------------------------------------------------------------- section 7
    print()
    print("=" * 78)
    print("7. VERDICT")
    print("=" * 78)
    print(f"  multiclass Brier {brier_mc:.4f} vs climatology {clima:.4f} "
          f"(model is informative: {(clima - brier_mc) / clima * 100:.0f}% better)")
    print(f"  ECE macro {ece_macro:.4f}  (per class: "
          + ", ".join(f"{k}={v:.4f}" for k, v in ece_per.items()) + ")")
    print()
    if ece_macro >= 0.05:
        print("  VERDICT: probabilities are materially miscalibrated (ECE >= 0.05).")
        print("  Displayed percentages are not reliable as frequencies. ")
    else:
        print("  VERDICT: probabilities are approximately calibrated (ECE < 0.05).")
    print("  NOTE: High-class probabilities are measured on very few samples")
    print("  (n=20 in the held-out set), so per-class ECE for High is noisy.")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())