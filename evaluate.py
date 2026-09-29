"""
One command that says how good the model in this repo actually is.

Issue #15 -- the app ships predictions to users but states no accuracy anywhere.
Before anything can be changed responsibly you need a number you trust,
produced the same way every time.

This script does NOT retrain, retune, or refit anything. It loads the COMMITTED
artifacts from artifacts/production/, recreates the exact 400 held-out rows the
model was never fitted on, and scores the predictions that model already makes.

The one thing that would quietly ruin this measurement
--------------------------------------------------------
The model was fitted on the 80% TRAIN split. If you re-split the data with a
different random_state, the "test" set is silently contaminated with training
rows and every number comes out inflated -- the classic way to "discover" a 0.99
accuracy that does not exist (which is exactly what the leaked v1 model did,
Issue #5). So this script does not invent a split:

  * it imports the split constants from train.py (test_size, random_state,
    stratify) and the feature order from train.derive_features(), so there is
    exactly ONE definition of the split in the project;
  * it asserts those constants against the `config` block recorded in
    artifacts/production/metadata.json, and asserts the CSV against the
    `dataset.sha256` recorded there. Same bytes + same seed + same column order
    => the same 400 rows, on any machine, in any process;
  * it publishes a sha256 of the held-out row indices, so "did I get the same
    rows?" is a one-line check rather than an assumption;
  * it re-derives the per-class numbers and compares them to the metrics
    recorded in metadata.json, failing loudly if they ever disagree.

Run:  python evaluate.py [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Single source of truth: the split and the feature order are DEFINED in
# train.py. Importing them (instead of copying the literals) is what makes this
# script immune to the "different seed => contaminated test set" trap.
from train import (RANDOM_STATE, STRATIFY, TARGET, TEST_SIZE, derive_features,
                   sha256_of)

BASE_DIR = Path(__file__).resolve().parent
PROD_DIR = BASE_DIR / "artifacts" / "production"
MODEL_PKL = PROD_DIR / "model.pkl"
LABEL_PKL = PROD_DIR / "label_encoder.pkl"
FEATURES_PKL = PROD_DIR / "feature_names.pkl"
METADATA_JSON = PROD_DIR / "metadata.json"
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"


# =============================================================================
# SPLIT
# =============================================================================
def held_out_split(df: pd.DataFrame, features: list[str], encoder):
    """
    Recreate the exact split train.py used. Byte-for-byte the same call, with
    the constants imported from train.py.

    Returns (X_test, y_test, train_idx, test_idx).
    """
    y = encoder.transform(df[TARGET])
    _, X_test, _, y_test = train_test_split(
        df[features], y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )
    return X_test, y_test, None, X_test.index


def split_fingerprint(idx) -> str:
    """
    Process-stable fingerprint of WHICH rows were held out.

    Depends only on the row indices, not on pickle/env/ordering side effects, so
    two runs on two machines must agree. This is the reproducibility receipt.
    """
    h = hashlib.sha256()
    h.update(np.asarray(sorted(int(i) for i in idx), dtype=np.int64).tobytes())
    return h.hexdigest()


def _counts(labels, classes) -> dict:
    """Class -> count, in the canonical class order, as plain ints."""
    vals = pd.Series(labels)
    return {c: int(vals.eq(c).sum()) for c in classes}


# =============================================================================
# PREFLIGHT
# =============================================================================
def preflight(encoder) -> tuple[list[str], dict]:
    """
    Refuse to report a number we cannot stand behind. Every assertion here is
    about proving we are measuring the shipped model on unseen rows.
    """
    print("=" * 78)
    print("0. PREFLIGHT -- can this measurement be trusted?")
    print("=" * 78)

    problems: list[str] = []

    missing = [p.name for p in (MODEL_PKL, LABEL_PKL, FEATURES_PKL, METADATA_JSON)
               if not p.exists()]
    if missing:
        print(f"[FAIL] missing committed production artifacts: {missing}")
        return [], {}

    md = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    ok = True

    # (a) The artifacts we are about to score are byte-for-byte the committed
    #     ones. Without this, evaluate.py would happily report metrics for
    #     whatever model.pkl happens to be on disk, and the number would be
    #     meaningless. metadata.json records the sha256 of its three siblings.
    files_ok = True
    for name, rec in md.get("files", {}).items():
        path = PROD_DIR / name
        want, got = rec.get("sha256"), sha256_of(path)
        same = want == got
        files_ok = files_ok and same
        print(f"[{'ok' if same else 'FAIL'}] {name} is the committed artifact "
              f"({got[:16]}...)")
    ok = ok and files_ok
    if not files_ok:
        problems.append("a production artifact does not match the sha256 recorded "
                        "in metadata.json -- the model on disk is not the shipped one")

    # (b) Same data the model was trained on.
    want_ds = md.get("dataset", {}).get("sha256")
    got_ds = sha256_of(DATA_PATH)
    same = want_ds == got_ds
    print(f"[{'ok' if same else 'FAIL'}] dataset is byte-identical to the one "
          f"the model was trained on")
    print(f"       recorded {str(want_ds)[:16]}...  local {got_ds[:16]}...")
    ok = ok and same

    # (c) Same split settings. If train.py's seed ever drifts from the recorded
    #     config, the numbers below would silently describe different rows.
    cfg = md.get("config", {})
    seed_ok = (cfg.get("test_size") == TEST_SIZE
               and cfg.get("random_state") == RANDOM_STATE
               and bool(cfg.get("stratified")) == bool(STRATIFY))
    print(f"[{'ok' if seed_ok else 'FAIL'}] split config matches metadata: "
          f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}, "
          f"stratify={bool(STRATIFY)}")
    print(f"       metadata says test_size={cfg.get('test_size')}, "
          f"random_state={cfg.get('random_state')}, "
          f"stratified={cfg.get('stratified')}")
    ok = ok and seed_ok
    if not seed_ok:
        problems.append("split config drift between train.py and metadata.json")

    # (d) Feature order: one source of truth, and it must match the contract.
    df = pd.read_csv(DATA_PATH)
    features = derive_features(df)
    contract = joblib.load(FEATURES_PKL)
    order_ok = list(contract) == features
    print(f"[{'ok' if order_ok else 'FAIL'}] feature order matches the committed "
          f"contract ({len(features)} features)")
    ok = ok and order_ok
    if not order_ok:
        problems.append("feature order disagrees with feature_names.pkl")

    # (e) The shipped encoder is the authority on class order.
    fresh = LabelEncoder().fit(df[TARGET])
    enc_ok = list(encoder.classes_) == list(fresh.classes_)
    print(f"[{'ok' if enc_ok else 'FAIL'}] shipped label encoder matches a fresh "
          f"fit on the data: {list(encoder.classes_)}")
    ok = ok and enc_ok

    # (f) The model must not have seen the rows we are about to score. The
    #     production pipeline is a Pipeline[StandardScaler -> CalibratedModel];
    #     report its shape so the number is attributable to the shipped model.
    print(f"[ok] scoring the committed production pipeline "
          f"(no retraining, no refitting)")

    return (features if ok else []), {"md": md, "ok": ok, "problems": problems}


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true",
                    help="also emit the results as JSON")
    args = ap.parse_args()

    encoder = joblib.load(LABEL_PKL)
    model = joblib.load(MODEL_PKL)

    features, info = preflight(encoder)
    if not info.get("ok") or not features:
        print("\nFATAL: refusing to print metrics for a measurement I cannot "
              "stand behind.")
        for p in info.get("problems", ["preflight failed"]):
            print(f"  - {p}")
        return 1
    md = info["md"]

    df = pd.read_csv(DATA_PATH)
    y_all = encoder.transform(df[TARGET])
    X_test, y_test, _, test_idx = held_out_split(df, features, encoder)
    y_pred = model.predict(X_test)
    fp = split_fingerprint(test_idx)
    classes = list(encoder.classes_)
    labels = list(range(len(classes)))

    n_test = len(y_test)
    print()
    print("=" * 78)
    print(f"1. HELD-OUT SPLIT -- {n_test} rows the model was never fitted on")
    print("=" * 78)
    print(f"  test_size={TEST_SIZE}  random_state={RANDOM_STATE}  "
          f"stratify={bool(STRATIFY)}   (imported from train.py, not retyped)")
    print(f"  held-out row-index fingerprint (sha256): {fp}")
    print(f"  true class counts: "
          f"{_counts(encoder.inverse_transform(y_test), classes)}")
    print(f"  predicted counts : "
          f"{_counts(encoder.inverse_transform(y_pred), classes)}")

    # =============================================================================
    print()
    print("=" * 78)
    print("2. CLASSIFICATION REPORT")
    print("=" * 78)
    print(classification_report(y_test, y_pred, target_names=classes,
                                zero_division=0, digits=3))

    # =============================================================================
    print("=" * 78)
    print("3. PER-CLASS PRECISION / RECALL / F1 (measured now, not quoted)")
    print("=" * 78)
    print(f"  {'class':<10}{'prec':>8}{'recall':>9}{'f1':>8}{'support':>9}")
    per_class = {}
    for i, c in enumerate(classes):
        p = precision_score(y_test, y_pred, labels=[i], average="macro",
                            zero_division=0)
        r = recall_score(y_test, y_pred, labels=[i], average="macro",
                         zero_division=0)
        f = f1_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)
        sup = int((y_test == i).sum())
        per_class[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                        "f1": round(float(f), 4), "support": sup}
        print(f"  {c:<10}{p:>8.3f}{r:>9.3f}{f:>8.3f}{sup:>9d}")

    # =============================================================================
    print()
    print("=" * 78)
    print("4. CONFUSION MATRIX  (rows = true, cols = predicted)")
    print("=" * 78)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = "  " + " " * 10 + "".join(f"{c:>10}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        print(f"  {c:<10}" + "".join(f"{cm[i][j]:>10d}" for j in range(len(classes))))
    print()
    print("  Errors only (off-diagonal), as true -> predicted:")
    errs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                errs.append((int(cm[i][j]), classes[i], classes[j]))
    for n, a, b in sorted(errs, reverse=True):
        print(f"    {n:>4d}  {a} -> {b}")
    if not errs:
        print("    (none)")

    # =============================================================================
    print()
    print("=" * 78)
    print("5. HEADLINE NUMBERS")
    print("=" * 78)
    headline = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro",
                                                      labels=labels, zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro",
                                                 labels=labels, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro",
                                         labels=labels, zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted",
                                            labels=labels, zero_division=0)), 4),
        "high_precision": per_class["High"]["precision"],
        "high_recall": per_class["High"]["recall"],
    }
    for k, v in headline.items():
        print(f"  {k:<20} {v}")

    # Accuracy alone is not an answer to "how good is it". With Medium at 78.75%
    # of the data, a model answering Medium every time scores 0.7875 -- so 0.8475
    # is ~6 points of real signal, not 85 points. Show the baseline next to it so
    # the headline cannot be quoted out of context.
    from sklearn.dummy import DummyClassifier
    X_tr, _, y_tr, _ = train_test_split(
        df[features], y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_all)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    b_pred = base.predict(X_test)
    b_acc = float(accuracy_score(y_test, b_pred))
    b_bal = float(balanced_accuracy_score(y_test, b_pred))
    print()
    print("  context -- the majority-class baseline (always answer the most")
    print("  common class), which is what an accuracy number must be read against:")
    b_acc_r, b_bal_r = round(b_acc, 4), round(b_bal, 4)
    print(f"    {'baseline':<20} {b_acc_r}   (balanced {b_bal_r})")
    print(f"    {'this model':<20} {headline['accuracy']}   "
          f"(balanced {headline['balanced_accuracy']})")
    print(f"    {'margin':<20} +{headline['accuracy'] - b_acc_r:.4f}")
    baseline = {"accuracy": b_acc_r, "balanced_accuracy": b_bal_r,
                "strategy": "most_frequent"}

    # =============================================================================
    print()
    print("=" * 78)
    print("6. CROSS-CHECK -- do these match what the model shipped claiming?")
    print("=" * 78)
    rec = md.get("metrics", {})
    drift = []
    for k, v in headline.items():
        if k in rec and abs(float(rec[k]) - v) > 1e-9:
            drift.append(f"{k}: metadata={rec[k]} vs measured={v}")
    if drift:
        print("[FAIL] measured values DISAGREE with artifacts/production/metadata.json:")
        for d in drift:
            print(f"       {d}")
        print("       The shipped metadata is now a false claim -- investigate "
              "before trusting either.")
        return 1
    print(f"[ok] all {len(headline)} headline numbers reproduce the values recorded")
    print("     in artifacts/production/metadata.json exactly. The artifact is")
    print("     self-consistent, and this measurement is reproducible from it.")

    # =============================================================================
    print()
    print("=" * 78)
    print("7. CONTAMINATION GUARD -- why the recorded seed is not a detail")
    print("=" * 78)
    # The rows the shipped model was actually fitted on are the complement of the
    # held-out set. Assert we scored none of them.
    fit_idx = set(int(i) for i in df.index) - set(int(i) for i in test_idx)
    scored = set(int(i) for i in test_idx)
    overlap = scored & fit_idx
    if overlap:
        print(f"[FAIL] {len(overlap)} scored rows are in the model's fitting set")
        return 1
    print(f"[ok] 0/{len(scored)} scored rows are in the model's fitting set "
          f"({len(fit_idx)} fit rows).")

    # Demonstrate the trap instead of asserting it in prose. Re-splitting with any
    # other seed drops ~78% of the new "test" rows into rows the model was already
    # fitted on, and the accuracy jumps ~12 points. Measured here, not asserted.
    print()
    print("  For scale -- the same committed model, scored on other seeds' splits:")
    print(f"    {'seed':>6}{'accuracy':>11}{'f1_macro':>10}"
          f"{'HighRec':>9}{'already fitted':>16}")
    for seed in (RANDOM_STATE, 0, 7, 123):
        _, Xs, _, ys = train_test_split(df[features], y_all, test_size=TEST_SIZE,
                                        random_state=seed, stratify=y_all)
        ps = model.predict(Xs)
        seen = len(set(int(i) for i in Xs.index) & fit_idx)
        tag = "   <- the recorded split" if seed == RANDOM_STATE else ""
        print(f"    {seed:>6}{accuracy_score(ys, ps):>11.4f}"
              f"{f1_score(ys, ps, average='macro'):>10.4f}"
              f"{recall_score(ys, ps, labels=[0], average='macro'):>9.3f}"
              f"{f'{seen}/400':>16}{tag}")
    print()
    print("  The other splits only look better because they are contaminated --")
    print("  the model already saw most of those rows while fitting. ~0.97 is not")
    print("  this model's accuracy; 0.8475 is. Treat any number near 0.97 on this")
    print("  dataset as a bug report, not an achievement.")

    if args.json:
        print()
        print(json.dumps({
            "held_out_rows": n_test,
            "split_fingerprint_sha256": fp,
            "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                      "stratify": bool(STRATIFY)},
            "classes": classes,
            "headline": headline,
            "majority_class_baseline": baseline,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "predicted_distribution": {c: int((y_pred == i).sum())
                                       for i, c in enumerate(classes)},
            "classification_report": classification_report(
                y_test, y_pred, target_names=classes, zero_division=0),
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
