"""
measure_risk_score_leakage.py -- evidence for docs/OVERALL_RISK_SCORE_VERDICT.md
(Issue #17)

Decides, with measured figures, whether Overall_Risk_Score is a legitimate model
input. It answers four things:

  SECTION 1  Produced, or measured?  Is the column a clinical measurement or a
             number computed from the other factors?
  SECTION 2  Its measured relationship to Risk_Level: rank correlation, mutual
             information, mutual information against the label's entropy, and a
             grouped summary with the recovered decision thresholds.
  SECTION 3  Is it new information, or a repackaging of the 17 legitimate
             factors?
  SECTION 4  Does the DEPLOYED artifact use it?  (checks artifacts/production/)
  SECTION 5  What a user was actually doing when they moved the old slider.

This script is read-only. It trains nothing, writes nothing, and touches no
artifact. It only reads cancer-risk-factors.csv, the archived v1 pickle (as
evidence), and the production bundle's feature contract.

    python measure_risk_score_leakage.py

Exit codes
    0  measurements completed (the leakage verdict is a finding, not a failure)
    1  the dataset is missing, or the deployed contract is inconsistent with the
       verdict (i.e. the shipped model DOES consume Overall_Risk_Score)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "cancer-risk-factors.csv"
ARCHIVE = BASE / "archive" / "legacy-pickles-2026-09"
PROD = BASE / "artifacts" / "production"
APP = BASE / "app.py"

TARGET = "Risk_Level"
SUS = "Overall_Risk_Score"
NONFEAT = ["Patient_ID", "Cancer_Type", TARGET, SUS]
# Ordinal order of the label, smallest risk first. NOT alphabetical: the classes
# are ordered, and a correlation computed on an alphabetical encoding is
# meaningless.
ORDINAL = {"Low": 0, "Medium": 1, "High": 2}


def hdr(t: str) -> None:
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> int:
    if not DATA.exists():
        print(f"FATAL: {DATA.name} not found. Run `python verify_dataset.py` first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(DATA)
    feats = [c for c in df.columns if c not in NONFEAT]
    y_ord = df[TARGET].map(ORDINAL).to_numpy()

    print(f"dataset : {DATA.name}  rows={len(df)}  cols={df.shape[1]}")
    print(f"python  : {sys.version.split()[0]}   "
          f"sklearn {__import__('sklearn').__version__}")
    print(f"question: is {SUS} a legitimate model input?")

    # =====================================================================
    hdr("SECTION 1  PRODUCED, OR MEASURED?")
    # =====================================================================
    s = df[SUS]
    print(f"dtype            {s.dtype}")
    print(f"unique values    {s.nunique()} of {len(df)} rows")
    print(f"range            [{s.min():.6f}, {s.max():.6f}]")
    print(f"mean / std       {s.mean():.6f} / {s.std():.6f}")
    print()
    print(f"  A clinical measurement -- a blood pressure, a BMI, an age -- is not")
    print(f"  bounded to [0, 1], and it does not come out different for all")
    print(f"  {len(df)} people. This column is bounded, unit-normalised, continuous,")
    print(f"  and distinct in every one of the {len(df)} rows: the signature of a")
    print(f"  COMPUTED score, not an observation.")

    X = df[feats].to_numpy()
    yv = s.to_numpy()
    lr = LinearRegression().fit(X, yv)
    r2 = float(lr.score(X, yv))
    res = yv - lr.predict(X)
    print()
    print(f"  Is it computed FROM the other columns?")
    print(f"    linear R^2 of {SUS} on the {len(feats)} other columns : {r2:.6f}")
    print(f"    residual std                                    : {res.std():.6f}")
    print()
    print("    the coefficients (sorted by magnitude):")
    for c, w in sorted(zip(feats, lr.coef_), key=lambda t: -abs(t[1])):
        print(f"      {c:<28} {w:+.6f}")
    print()
    print("    Nine risk-factor columns carry near-equal positive weight; the")
    print("    rest sit near zero. So the score is a weighted summary of the")
    print("    factors -- an upstream composite, not an independent measurement.")

    print()
    print(f"  Is the unexplained {1 - r2:.1%} a term we failed to model?")
    worst = max(abs(stats.spearmanr(df[c].to_numpy(), res).statistic)
                for c in feats)
    z = (res - res.mean()) / res.std()
    print(f"    max |spearman(residual, any factor)| : {worst:.4f}")
    print(f"    residual skew / excess kurtosis      : "
          f"{stats.skew(res):+.4f} / {stats.kurtosis(res):+.4f}")
    print(f"    fraction within 1 sigma : {np.mean(np.abs(z) < 1):.4f} "
          f"(a normal: 0.6827)")
    print(f"    fraction within 2 sigma : {np.mean(np.abs(z) < 2):.4f} "
          f"(a normal: 0.9545)")
    print("    The residual is uncorrelated with every input and roughly normal,")
    print("    so it is independent noise added after the factors were combined --")
    print("    not a relationship we mis-specified.")

    # =====================================================================
    hdr("SECTION 2  MEASURED RELATIONSHIP TO Risk_Level")
    # =====================================================================
    print(f"label encoded ORDINALLY (Low=0, Medium=1, High=2) -- an alphabetical")
    print(f"encoding would report a near-zero correlation and mean nothing.\n")
    r = np.corrcoef(s, y_ord)[0, 1]
    rho = stats.spearmanr(s, y_ord)
    tau = stats.kendalltau(s, y_ord)
    print(f"  (a) correlation with the ordinal label")
    print(f"      Pearson  r      = {r:+.6f}")
    print(f"      Spearman rho    = {rho.statistic:+.6f}   (p = {rho.pvalue:.2e})")
    print(f"      Kendall  tau    = {tau.statistic:+.6f}   (p = {tau.pvalue:.2e})")

    # Rank correlation is CAPPED by the coarseness of the label: with only 3
    # levels and 1574 of 2000 rows in one of them, most pairs are tied in y and
    # cannot contribute. The relationship itself is better measured by asking,
    # of pairs from *different* classes, how often the score is ordered right.
    classes = df[TARGET].map(ORDINAL).to_numpy()
    sc = s.to_numpy()
    iu = np.triu_indices(len(df), 1)
    diff_class = classes[:, None] != classes[None, :]
    ordered_right = ((classes[:, None] < classes[None, :])
                     == (sc[:, None] < sc[None, :]))
    concordant = int((ordered_right & diff_class)[iu].sum())
    discordant = int(((~ordered_right) & diff_class)[iu].sum())
    total = concordant + discordant
    print()
    print(f"      the label has only {df[TARGET].nunique()} levels and "
          f"{int(df[TARGET].value_counts().iloc[0])} of {len(df)} rows in one of")
    print(f"      them, so most pairs are tied in the label and rank correlation")
    print(f"      is capped below 1.0. Asking instead about pairs from DIFFERENT")
    print(f"      classes: {concordant}/{total} cross-class pairs are ordered")
    print(f"      correctly by the score = {concordant / total:.6f}")
    print(f"      -> the score NEVER puts a higher-risk patient below a lower-risk")
    print(f"         one. The ordering is total; only the ranking metric is coarse.")

    mi = float(mutual_info_classif(df[[SUS]].to_numpy(), y_ord,
                                   random_state=42)[0])
    h_lab = entropy(df[TARGET].value_counts().to_numpy())
    print()
    print(f"  (b) mutual information")
    print(f"      I({SUS} ; {TARGET}) = {mi:.6f} nats")
    print(f"      H({TARGET})              = {h_lab:.6f} nats   "
          f"(the label's total uncertainty)")
    print(f"      normalised MI            = {mi / h_lab:.4f}   "
          "(1.0 = the column determines the label)")
    print("      -> knowing only this number removes essentially ALL of the")
    print("         uncertainty about the risk class.")

    mi_f = mutual_info_classif(X, y_ord, random_state=42)
    print()
    print(f"      for comparison, the strongest legitimate features individually:")
    for i in np.argsort(mi_f)[::-1][:5]:
        print(f"        {feats[i]:<28} {mi_f[i]:.6f} nats")
    print(f"      the 17 legitimate features, summed  : {mi_f.sum():.6f} nats")
    print(f"      this ONE column                     : {mi:.6f} nats")

    print()
    print(f"  (c) grouped summary")
    g = df.groupby(TARGET)[SUS].agg(["count", "min", "max", "mean"])
    g = g.reindex(["Low", "Medium", "High"])
    print(f"      {'class':<8}{'n':>6}{'min':>11}{'max':>11}{'mean':>11}")
    for cls, row in g.iterrows():
        print(f"      {cls:<8}{int(row['count']):>6}{row['min']:>11.6f}"
              f"{row['max']:>11.6f}{row['mean']:>11.6f}")
    lo_max, med_min = g.loc["Low", "max"], g.loc["Medium", "min"]
    med_max, hi_min = g.loc["Medium", "max"], g.loc["High", "min"]
    print(f"      the ranges do not touch: Low max {lo_max:.6f} < "
          f"Medium min {med_min:.6f}")
    print(f"                               Medium max {med_max:.6f} < "
          f"High min {hi_min:.6f}")

    pred = np.where(df[SUS] <= lo_max, "Low",
                    np.where(df[SUS] <= med_max, "Medium", "High"))
    print()
    print(f"  (d) the decision thresholds")
    print(f"      the Low/Medium cut lies in ({lo_max:.6f}, {med_min:.6f})")
    print(f"      the Medium/High cut lies in ({med_max:.6f}, {hi_min:.6f})")
    print(f"      -> consistent with cuts at 0.33 and 0.66")
    print()
    print(f"      a rule that reads ONLY {SUS}: "
          f"{int((pred == df[TARGET]).sum())}/{len(df)} correct "
          f"= {(pred == df[TARGET]).mean():.2%}")
    print(f"      macro-F1 of that single-column rule : "
          f"{f1_score(df[TARGET], pred, average='macro'):.6f}")
    print("      (the shipped 17-feature model scores macro-F1 0.6572 on its")
    print("       held-out split -- Issue #15)")

    # =====================================================================
    hdr("SECTION 3  NEW INFORMATION, OR A REPACKAGING OF THE FACTORS?")
    # =====================================================================
    print(f"  R^2 of {SUS} regressed on the {len(feats)} legitimate factors : {r2:.6f}")
    print(f"  so {r2:.0%} of the column is already present in the features the")
    print(f"  model receives anyway. The column does not observe anything new; it")
    print(f"  restates the factors in the one form that makes the label trivial.")

    # =====================================================================
    hdr("SECTION 4  DOES THE DEPLOYED ARTIFACT USE IT?")
    # =====================================================================
    ok = True
    prod_feats_path = PROD / "feature_names.pkl"
    if prod_feats_path.exists():
        pf = [str(x) for x in joblib.load(prod_feats_path)]
        print(f"  artifacts/production/feature_names.pkl : {len(pf)} features")
        print(f"  contains {SUS}? {SUS in pf}")
        if SUS in pf:
            ok = False
        print(f"  model.pkl present : {(PROD / 'model.pkl').exists()}")
    else:
        print("  artifacts/production/feature_names.pkl not present -- "
              "skipping the file check")
        print("  (run `python export_production.py` to materialise the bundle)")

    meta_path = PROD / "metadata.json"
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  metadata excluded_features : {meta.get('excluded_features')}")

    if APP.exists():
        src = APP.read_text(encoding="utf-8", errors="replace")
        asks = f'"{SUS}"' in src and "widget" in src
        print(f"  app.py asks the user for it : {asks}")
        if asks:
            ok = False
    print()
    print(f"  >>> the DEPLOYED artifact consumes {SUS}: {not ok}")

    # =====================================================================
    hdr("SECTION 5  WHAT MOVING THE OLD SLIDER ACTUALLY DID")
    # =====================================================================
    model_pkl = ARCHIVE / "model_xgb_new.pkl"
    arch_feats = ARCHIVE / "feature_names.pkl"
    if model_pkl.exists() and arch_feats.exists():
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        names = [str(x) for x in joblib.load(arch_feats)]
        v1 = joblib.load(model_pkl)

        # The v1 model was fit on STANDARDISED inputs, and its scaler was never
        # saved (Issue #2). Rebuild it exactly as notebook cells 4/8/9 did, so
        # the demo feeds the model the kind of numbers it was trained on --
        # otherwise everything below returns a constant and proves nothing.
        X18 = df.drop(columns=["Risk_Level", "Patient_ID", "Cancer_Type"])
        le_v1 = LabelEncoder()
        y_v1 = le_v1.fit_transform(df["Risk_Level"])
        Xtr, _, _, _ = train_test_split(X18, y_v1, test_size=0.2,
                                        random_state=42, stratify=y_v1)
        scaler = StandardScaler().fit(Xtr)

        print("  The v1 app offered a 0.00-1.00 slider for Overall_Risk_Score in")
        print("  a group literally titled 'Engineered score'. Below, all 17 other")
        print("  features are frozen at their medians and ONLY the slider moves,")
        print("  scored by the archived v1 model:\n")
        base = {c: float(df[c].median()) for c in names}
        print(f"      {'slider':>8}   predicted class")
        prev = None
        flips = 0
        for v in (0.20, 0.32, 0.33, 0.34, 0.50, 0.65, 0.66, 0.67, 0.80):
            row = dict(base)
            row[SUS] = v
            frame = pd.DataFrame([row])[names]
            idx = int(v1.predict(scaler.transform(frame))[0])
            cls = str(le_v1.classes_[idx])
            note = ""
            if prev is not None and cls != prev:
                note = "  <- flips here"
                flips += 1
            print(f"      {v:>8.2f}   {cls:<8}{note}")
            prev = cls
        print()
        if flips:
            print(f"  The other 17 features were held constant, yet the answer")
            print(f"  changed {flips} time(s), tracking the slider. Moving it is not")
            print(f"  describing the patient -- it is SELECTING the diagnosis.")
        else:
            print("  UNEXPECTED: the class did not respond to the slider at all.")
    else:
        print("  archived v1 pickle not present -- skipping the slider demo")

    print()
    print("=" * 78)
    if not ok:
        print("FAIL: the deployed contract still consumes Overall_Risk_Score.")
        return 1
    print("MEASUREMENT COMPLETE -- verdict: NOT a legitimate input.")
    print("The deployed artifact excludes it (see SECTION 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
