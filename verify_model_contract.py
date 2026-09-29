"""
Issue #14 -- assert the saved model and feature_names.pkl agree, loudly.

Answers from the ARTIFACTS themselves (never the notebook or the filename):

  * how many features the saved model expects   ->  model.n_features_in_
  * whether it records their names              ->  feature_names_in_ (pipeline, scaler, forest)
  * where feature_names.pkl came from           ->  train.py derive_features() (CSV column order
                                                    minus 4 non-features) copied verbatim by
                                                    export_production.py
  * what happens when an upload reorders columns->  the app's df[FEATURE_NAMES] reorder restores
                                                    canonical order (proved by prediction equality),
                                                    and the scaler inside model.pkl defends the order
                                                    if it recorded feature_names_in_

Exit codes
    0  every assertion passed; the input contract holds end to end
    1  at least one assertion failed (loud FAIL lines, non-zero exit)

Run:  python verify_model_contract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from train import (BASE_DIR, DATA_PATH, LEAKY_FEATURE, NON_FEATURE_COLUMNS,
                   OUT_DIR, _final_estimator)

PROD_DIR = BASE_DIR / "artifacts" / "production"
MODEL_PKL = PROD_DIR / "model.pkl"
LABEL_PKL = PROD_DIR / "label_encoder.pkl"
FEATURES_PKL = PROD_DIR / "feature_names.pkl"
METADATA_JSON = PROD_DIR / "metadata.json"
BUNDLE_PATH = OUT_DIR / "model_bundle.joblib"

EXPECTED_COUNT = 17  # known-good; each check re-derives it from an artifact too

results: list[tuple[str, bool, str, bool]] = []  # (name, passed, detail, warn_only)


def red(title: str) -> str:
    return f"[FAIL] {title}"


def green(title: str) -> str:
    return f"[ok]   {title}"


def check(name: str, passed: bool, detail: str, warn_only: bool = False):
    results.append((name, passed, detail, warn_only))
    tag = "WARN" if (warn_only and not passed) else ("ok" if passed else "FAIL")
    print(f"[{tag}] {name:<44} {detail}")


def ok_count() -> int:
    return sum(1 for _, p, _, _ in results if p)


def warn_count() -> int:
    return sum(1 for _, p, _, w in results if w and not p)


def fail_count() -> int:
    return sum(1 for _, p, _, w in results if not p and not w)


def passed_to_text() -> str:
    return "PASS" if fail_count() == 0 else "FAIL"


def fail_text() -> str:
    return f", {fail_count()} FAIL" if fail_count() else ""


def warn_text() -> str:
    return f", {warn_count()} WARN" if warn_count() else ""


def main() -> int:
    for f in (MODEL_PKL, FEATURES_PKL, METADATA_JSON, BUNDLE_PATH):
        if not f.exists():
            print(red(f"missing {f} -- run `python train.py` + `python export_production.py` first"))
            return 1

    pipeline = joblib.load(MODEL_PKL)
    features = list(joblib.load(FEATURES_PKL))
    md = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    bundle = joblib.load(BUNDLE_PATH)

    # ------------------------------------------------------------------ counts
    n_pipe = int(getattr(pipeline, "n_features_in_", -1))
    check("1. model.pkl n_features_in_", n_pipe == EXPECTED_COUNT,
          f"pipeline expects {n_pipe} (known-good {EXPECTED_COUNT})")

    forest = _final_estimator(pipeline)
    n_forest = int(getattr(forest, "n_features_in_", -1))
    check("2. final estimator n_features_in_", n_forest == EXPECTED_COUNT,
          f"{type(forest).__name__} expects {n_forest}")

    scaler = pipeline.named_steps.get("scaler")
    n_scaler = int(getattr(scaler, "n_features_in_", -1)) if scaler is not None else -1
    check("3. scaler step n_features_in_", n_scaler == EXPECTED_COUNT,
          f"scaler expects {n_scaler}")

    # ------------------------------------------------------- recorded names? --
    fnames_pipe = getattr(pipeline, "feature_names_in_", None)
    check("4. pipeline.feature_names_in_", fnames_pipe is not None,
          f"pipeline records names: {fnames_pipe is not None}", warn_only=True)

    fnames_scaler = getattr(scaler, "feature_names_in_", None)
    check("5. scaler.feature_names_in_", fnames_scaler is not None,
          f"scaler records {len(fnames_scaler) if fnames_scaler is not None else 'NO'} names",
          warn_only=True)

    fnames_forest = getattr(forest, "feature_names_in_", None)
    check("6. forest.feature_names_in_", fnames_forest is not None,
          f"forest records {len(fnames_forest) if fnames_forest is not None else 'NO'} names "
          "(fit on numpy -> expected None)", warn_only=True)

    if fnames_scaler is not None:
        ordered = list(fnames_scaler) == features
        check("7. scaler names == feature_names.pkl (order-sensitive)",
              ordered,
              "identical ordered list" if ordered else
              f"MISMATCH: scaler={list(fnames_scaler)} vs pkl={features}")

    # --------------------------------------- sources of feature_names.pkl agree
    agreed = True
    s_names = getattr(scaler, "feature_names_in_", None)
    s_list = list(s_names) if s_names is not None else []
    sources = {
        "feature_names.pkl": features,
        "bundle.feature_names": list(bundle["feature_names"]),
        "scaler.feature_names_in_": s_list,
    }
    base = features
    lines = []
    for name, s in sources.items():
        ok = s is not None and list(s) == base
        agreed = agreed and ok
        lines.append(f"{name}={'matches' if ok else 'MISMATCH ' + str(s)}")
    lines.append("metadata.json carries no feature list (by design: pkl+scaler+checksum hold it)")
    check("8. all recorded feature lists agree", agreed, "; ".join(lines[-2:]))

    import hashlib
    def sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    rec = md.get("files", {}).get("feature_names.pkl", {})
    rec_sha = rec.get("sha256")
    check("8b. feature_names.pkl is the exported file",
          rec_sha is not None and sha256_of(FEATURES_PKL) == rec_sha,
          f"disk sha256 {sha256_of(FEATURES_PKL)[:16]}... matches the recorded export"
          if rec_sha is not None and sha256_of(FEATURES_PKL) == rec_sha else
          f"disk sha256 {sha256_of(FEATURES_PKL)[:16]}... vs recorded {rec_sha}")

    df = pd.read_csv(DATA_PATH)
    derived = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != LEAKY_FEATURE]
    check("9. feature_names.pkl == CSV order minus 4 non-features",
          derived == features,
          "order is the dataset's physical column order (train.py derive_features)")

    # ------------------------------------------- internal consistency of list
    check("10. count + no duplicates + leak-free",
          len(features) == EXPECTED_COUNT and len(set(features)) == len(features)
          and LEAKY_FEATURE not in features,
          f"{len(features)} unique features, {LEAKY_FEATURE} absent")

    # ------------------------------------------------- order-defense behavior
    # A DataFrame in the WRONG column order thrown straight at the artifact.
    perm = features[::-1][:5] + features[5:]
    df_wrong = pd.DataFrame(np.random.RandomState(0).rand(10, 17), columns=perm)
    try:
        pred_wrong = pipeline.predict(df_wrong)
        defended = False
        detail = f"pipeline silently predicted {len(pred_wrong)} rows on WRONG-ORDER columns"
    except ValueError as exc:
        defended = True
        detail = f"pipeline RAISED on wrong-order columns: {str(exc)[:90]}..."
    names_known = getattr(pipeline, "feature_names_in_", None) is not None
    # If the artifact records names it MUST refuse a wrong order; if it records
    # none, order protection rests only on app.py's df[FEATURE_NAMES] reorder.
    check("11. wrong-order DataFrame straight at model.pkl",
          defended if names_known else True,
          detail + ("" if names_known else " -- names NOT recorded here; "
                    "protection is the app reorder (see 12/13)"),
          warn_only=not names_known)

    # ---------------------------------------------------- app-path equivalence
    # A user CSV with the RIGHT columns in a DIFFERENT order goes through
    # app.py preprocess_input: missing/extra column checks then df[FEATURE_NAMES].
    row = df.iloc[:100].copy()
    canonical = row[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    shuffled_cols = features[-4:] + features[:-4]          # same 17, reordered
    upload = row[shuffled_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    reordered = upload[features].copy()                    # what preprocess_input does
    p_canon = pipeline.predict(canonical.to_numpy())
    p_upload_raw = pipeline.predict(upload.to_numpy())     # no app reorder (wrong)
    p_app = pipeline.predict(reordered.to_numpy())         # after app reorder

    check("12. app reorder restores canonical predictions",
          bool(np.array_equal(p_app, p_canon)),
          f"permuted-order upload -> app reorder -> {bool(np.array_equal(p_app, p_canon))} "
          f"identical to canonical (yes: app fixes the order)")

    wrong_ratio = float((p_upload_raw != p_canon).mean())
    check("13. raw wrong-order matrix WOULD differ (risk made visible)",
          wrong_ratio > 0,
          f"{wrong_ratio:.0%} of labels would flip without the app's reorder")

    # ----------------------------------------------- zero-width sanity checks
    probe = pd.DataFrame([{f: 0.0 for f in features}])
    acc_roundtrip = bool(np.array_equal(
        pipeline.predict(probe.to_numpy()),
        pipeline.named_steps["model"].predict(
            scaler.transform(probe[features].to_numpy()))))
    check("14. round trip through documented inference path",
          acc_roundtrip, "zero row -> parallel paths agree")

    # ------------------------------------------------------- value-range table
    print("\n" + "=" * 76)
    print("VALUE RANGES (from the full validated dataset, Issue #14)")
    print("=" * 76)
    print(f"{'feature':<26}{'dtype':<9}{'min':<8}{'max':<9}{'scale':<15}how established")
    for f in features:
        s = df[f]
        lo, hi = float(s.min()), float(s.max())
        vals = set(np.unique(s))
        if vals <= {0, 1}:
            scale = "binary 0/1"
        elif f in ("Age", "BMI"):
            scale = "continuous"
        else:
            scale = "ordinal 0-10"
        print(f"{f:<26}{str(s.dtype):<9}{lo:<8.6g}{hi:<9.6g}{scale:<15}"
              f"dataset stats (dataset sha256-gated)")
    print(f"\n{passed_to_text()}: model input contract -- {ok_count()} checks pass, "
          f"{len(results)} total{fail_text()}{warn_text()}")
    return 0 if fail_count() == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotFittedError:
        print(red("model.pkl pipeline is not fitted"))
        sys.exit(1)