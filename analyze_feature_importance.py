"""
Issue #12 -- what is the model actually keying on?

Two importance measures, one honest question:

  * IMPURITY-BASED  (sklearn feature_importances_): computed DURING fit on the
    TRAINING data. It answers "how much does this feature reduce node impurity
    in the fitted trees?" It is biased toward high-cardinality/continuous
    features and splits credit between correlated features, and it never sees a
    single held-out example.
  * PERMUTATION      (sklearn.inspection.permutation_importance): shuffle one
    feature's values on the HELD-OUT set, score the model, measure the drop.
    It answers the more honest question "if this feature lost all its
    information, how much worse would held-out predictions get?"

The distinction matters because a model can lean on a feature that LOOKS
important during training (easy splits, many unique values) while being nearly
irrelevant for real predictions -- or the reverse.

This script is read-only: it loads the deployed bundle, never retrains.

It also investigates:
  * redundancy pairs (BMI/Obesity, Physical_Activity/Physical_Activity_Level):
    correlation + individual vs joint permutation importance (dilution check)
  * dominance: is one feature overwhelmingly more important than the rest?
  * medical plausibility inputs: correlation of each feature with the ordinal
    risk level (Low=0, Medium=1, High=2)

Run:  python analyze_feature_importance.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (joblib load)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, make_scorer, recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: F401  (joblib load)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "artifacts" / "model_bundle.joblib"
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"
FIGURE_DIR = BASE_DIR / "docs" / "figures"

N_REPEATS = 10
ORDINAL = {"Low": 0, "Medium": 1, "High": 2}
REDUNDANT_PAIRS = [("BMI", "Obesity"),
                   ("Physical_Activity", "Physical_Activity_Level")]


def held_out_split():
    """Exact 400-row held-out split train.py used (no training)."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH)
    feats = [c for c in df.columns
             if c not in ("Patient_ID", "Cancer_Type", "Risk_Level", "Overall_Risk_Score")]
    y = LabelEncoder().fit_transform(df["Risk_Level"])
    _, X_te, _, y_te = train_test_split(df[feats], y, test_size=0.2,
                                        random_state=42, stratify=y)
    return feats, X_te, y_te


def main() -> int:
    if not BUNDLE_PATH.exists():
        sys.exit(f"[FAIL] {BUNDLE_PATH} missing -- run `python train.py`")
    bundle = joblib.load(BUNDLE_PATH)
    forest = bundle["model"]
    scaler = bundle["scaler"]
    feats, X_te, y_te = held_out_split()
    X_te_n = scaler.transform(X_te[feats].to_numpy())

    print("=" * 78)
    print("1. IMPURITY-BASED IMPORTANCE (training-time; recorded by train.py)")
    print("=" * 78)
    imp = pd.Series(forest.feature_importances_, index=feats).sort_values(ascending=False)
    for name, v in imp.items():
        print(f"    {name:<28} {v:.4f}")

    print()
    print("=" * 78)
    print(f"2. PERMUTATION IMPORTANCE ON HELD-OUT (n_repeats={N_REPEATS})")
    print("=" * 78)
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
        "high_recall": make_scorer(lambda yt, yp: recall_score(yt, yp, labels=[0], average=None)[0]),
        "balanced_acc": make_scorer(balanced_accuracy_score),
    }
    perm = {}
    for name, scorer in scorers.items():
        r = permutation_importance(forest, X_te_n, y_te, scoring=scorer,
                                   n_repeats=N_REPEATS, random_state=42, n_jobs=1)
        perm[name] = pd.DataFrame(
            {"mean": r.importances_mean, "std": r.importances_std}, index=feats)
        total = r.importances_mean.sum()
        top = perm[name]["mean"].sort_values(ascending=False).head(5)
        print(f"\n  [{name}]  (sum of drops: {total:.4f})")
        for ft in top.index:
            print(f"    {ft:<28} {top[ft]:+.4f} ± {perm[name].loc[ft, 'std']:.4f}")

    print()
    print("=" * 78)
    print("3. DOMINANCE -- does one feature dominate?")
    print("=" * 78)
    for name, tbl in perm.items():
        s = tbl["mean"].sort_values(ascending=False)
        top1, top2 = s.iloc[0], s.iloc[1]
        share = s.iloc[0] / s.abs().sum() if s.abs().sum() > 0 else float("nan")  # noqa: F841
        print(f"  [{name}] top1={s.index[0]} ({top1:+.4f}), "
              f"top2={s.index[1]} ({top2:+.4f}), ratio top1/top2 = {top1 / top2:.2f}")

    print()
    print("=" * 78)
    print("4. REDUNDANT PAIRS -- correlation + individual vs joint permutation")
    print("=" * 78)
    df = pd.read_csv(DATA_PATH)
    df["_risk_ordinal"] = df["Risk_Level"].map(ORDINAL)
    for a, b in REDUNDANT_PAIRS:
        rho = df[[a, b]].corr(method="spearman").iloc[0, 1]
        rho_t = df[[a, "_risk_ordinal"]].corr(method="spearman").iloc[0, 1]
        rho_b = df[[b, "_risk_ordinal"]].corr(method="spearman").iloc[0, 1]
        acc_tbl = perm["accuracy"]
        ind = acc_tbl.loc[[a, b], "mean"]
        # joint: shuffle both columns with the SAME permutation
        Xj = X_te[feats].to_numpy()
        idx = np.arange(len(Xj))
        rng = np.random.RandomState(42)
        col_a, col_b = feats.index(a), feats.index(b)
        rng.shuffle(idx)
        Xp = Xj.copy()
        Xp[idx, col_a], Xp[idx, col_b] = Xj[:, col_a], Xj[:, col_b]
        base = accuracy_score(y_te, forest.predict(scaler.transform(Xp)))
        drop_joint = float(accuracy_score(y_te, forest.predict(scaler.transform(X_te[feats].to_numpy()))) - base)
        print(f"\n  {a} vs {b}:")
        print(f"    spearman(a, b)          = {rho:.3f}")
        print(f"    spearman(a, risk)       = {rho_t:+.3f}   "
              f"spearman(b, risk) = {rho_b:+.3f}")
        print(f"    perm-imp (accuracy)     {a} {ind[a]:+.4f}   "
              f"{b} {ind[b]:+.4f}")
        print(f"    perm-imp JOINT (both)   {drop_joint:+.4f}   "
              f"(dilution: joint > sum of parts = {drop_joint > ind[a] + ind[b]})")

    print()
    print("=" * 78)
    print("5. FEATURE->RISK DIRECTION (medical-plausibility input)")
    print("=" * 78)
    corr = df[feats + ["_risk_ordinal"]].corr(method="spearman")["_risk_ordinal"].drop("_risk_ordinal")
    top = corr.reindex(perm["accuracy"]["mean"].sort_values(ascending=False).index)
    for ft in top.index:
        print(f"    {ft:<28} spearman vs risk level: {top[ft]:+.3f}")

    print()
    print("=" * 78)
    print("6. PLOT -> docs/figures/permutation_importance.png")
    print("=" * 78)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    order = perm["accuracy"]["mean"].sort_values(ascending=False).index

    ax = axes[0]
    vals = imp.reindex(order)
    ax.barh(order, vals, color="#8aa2c0")
    ax.invert_yaxis()
    ax.set_title("Impurity-based importance\n(fit-time, train data -- biased)\n", fontsize=9)
    ax.set_xlabel("mean decrease in impurity", fontsize=8.5)

    ax = axes[1]
    tbl = perm["accuracy"].reindex(order)
    ax.barh(order, tbl["mean"], xerr=tbl["std"], capsize=3,
            color=["#d9534f" if i < 5 else "#6f9e77" for i in range(len(order))])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"Permutation importance, held-out (n={len(y_te)}, "
                 f"repeats={N_REPEATS})\naccuracy drop when a feature is shuffled",
                 fontsize=9)
    ax.set_xlabel("drop in accuracy (higher = more important)", fontsize=8.5)
    for t in (axes[0].get_yticklabels() + axes[1].get_yticklabels()):
        t.set_fontsize(8)
    fig.suptitle("Issue #12: what is the model keying on?  "
                 f"forest {type(forest).__name__}, 17 leak-free features",
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = FIGURE_DIR / "permutation_importance.png"
    fig.savefig(out_png, dpi=150)
    print(f"[ok] saved {out_png.relative_to(BASE_DIR)}")

    # machine-readable table next to the plot
    csv = pd.DataFrame({
        **{f"perm_{k}_mean": tbl["mean"] for k, tbl in perm.items()},
        **{f"perm_{k}_std": tbl["std"] for k, tbl in perm.items()},
        "impurity_based": imp,
    })
    csv.to_csv(FIGURE_DIR / "permutation_importance.csv")
    print(f"[ok] saved { (FIGURE_DIR / 'permutation_importance.csv').relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())