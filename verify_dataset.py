"""
Read-only reproducibility verifier for the Cancer Risk Prediction dataset.

This script ANSWERS questions about the existing artifacts. It does not
train, retrain, tune, or modify anything -- it only reads files and prints
evidence. Safe to run at any time; it writes nothing.

    python verify_dataset.py          # human-readable report
    python verify_dataset.py --json   # machine-readable, for CI

Exit codes
    0  all checks passed
    1  at least one check failed (dataset missing, hash mismatch, contract broken)

To see the full written analysis, read docs/DATASET_PROVENANCE.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

# The artifacts were pickled under older scikit-learn versions; that mismatch is
# reported as a finding rather than printed as a wall of warnings on every run.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import pandas as pd
import sklearn

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Recorded facts about the dataset that produced model_xgb_new.pkl.
# These are the values to compare against, not recomputed expectations.
# ---------------------------------------------------------------------------

DATASET_FILE = "cancer-risk-factors.csv"
DATASET_SHA256 = "01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5"
DATASET_BYTES = 141851
DATASET_ROWS = 2000
DATASET_COLS = 21
N_FEATURES = 18
TARGET = "Risk_Level"
TARGET_CLASSES = ["High", "Low", "Medium"]
# LabelEncoder sorts its classes, so the encoding is deterministic:
TARGET_ENCODING = {"High": 0, "Low": 1, "Medium": 2}
NON_FEATURE_COLUMNS = ["Patient_ID", "Cancer_Type", "Risk_Level"]
# Columns dropped before the split. Their ORDER in the CSV defines the
# feature order, which is the model's real input contract.
EXPECTED_FEATURE_ORDER = [
    "Age", "Gender", "Smoking", "Alcohol_Use", "Obesity", "Family_History",
    "Diet_Red_Meat", "Diet_Salted_Processed", "Fruit_Veg_Intake",
    "Physical_Activity", "Air_Pollution", "Occupational_Hazards",
    "BRCA_Mutation", "H_Pylori_Infection", "Calcium_Intake",
    "Overall_Risk_Score", "BMI", "Physical_Activity_Level",
]

MODEL_FILE = "model_xgb_new.pkl"
ENCODER_FILE = "label_encoder.pkl"
FEATURES_FILE = "feature_names.pkl"
# The fitted StandardScaler is NOT part of the artifact set -- a known gap.
SCALER_FILE = "scaler.pkl"

results: list[dict] = []


def check(name: str, passed: bool, detail: str = "", warn_only: bool = False) -> bool:
    results.append({"check": name, "status": "PASS" if passed else ("WARN" if warn_only else "FAIL"),
                    "detail": detail})
    return passed


def sha256_raw(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_normalized(path: Path) -> str:
    """CRLF -> LF normalized, so the hash survives git line-ending conversion."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk.replace(b"\r\n", b"\n"))
    return h.hexdigest()


def embedded_sklearn_version(path: Path) -> str | None:
    """
    Recover the sklearn version an artifact was pickled with.

    sklearn writes `_sklearn_version` into the pickle but strips it in
    `__setstate__`, so it cannot be read off the loaded object.
    """
    if not path.exists():
        return None
    found = {m.decode() for m in re.findall(
        rb"_sklearn_version.{0,3}([0-9]+\.[0-9]+(?:\.[0-9]+)?)", path.read_bytes(), re.S)}
    return ", ".join(sorted(found)) or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    # -- 1. dataset identity --------------------------------------------------
    data_path = BASE_DIR / DATASET_FILE
    if not data_path.exists():
        check("dataset present", False, f"{DATASET_FILE} not found in {BASE_DIR}")
        return report(args.json)

    check("dataset present", True, f"{DATASET_FILE} ({data_path.stat().st_size} bytes)")

    # Determine line-ending-only difference up front: git's autocrlf rewrites
    # LF -> CRLF on checkout, which changes the byte count without changing a
    # single data value. That must not be reported as corruption.
    raw_hash = sha256_raw(data_path)
    eol_only = raw_hash != DATASET_SHA256 and sha256_normalized(data_path) == DATASET_SHA256

    check("dataset size",
          data_path.stat().st_size == DATASET_BYTES or eol_only,
          f"{data_path.stat().st_size} bytes (expected {DATASET_BYTES})"
          + (" -- differs only by CRLF line endings" if eol_only else ""),
          warn_only=eol_only)

    if raw_hash == DATASET_SHA256:
        check("dataset sha256", True, f"{raw_hash} (byte-exact)")
    elif eol_only:
        check("dataset sha256", True,
              f"{raw_hash} -- differs only by CRLF line endings; content matches "
              f"{DATASET_SHA256}", warn_only=True)
    else:
        check("dataset sha256", False,
              f"expected {DATASET_SHA256}, got {raw_hash}. The data has CHANGED; this model "
              f"was trained on the recorded version.")

    df = pd.read_csv(data_path)

    # -- 2. shape -------------------------------------------------------------
    check("row count", len(df) == DATASET_ROWS, f"{len(df)} rows (expected {DATASET_ROWS})")
    check("column count", df.shape[1] == DATASET_COLS,
          f"{df.shape[1]} columns (expected {DATASET_COLS})")

    # -- 3. data quality ------------------------------------------------------
    missing = int(df.isna().sum().sum())
    check("no missing values", missing == 0, f"{missing} missing cells")
    check("no duplicate rows", int(df.duplicated().sum()) == 0,
          f"{int(df.duplicated().sum())} fully duplicated rows")
    check("unique patient ids", int(df["Patient_ID"].duplicated().sum()) == 0,
          f"{df['Patient_ID'].nunique()} unique of {len(df)}")

    # -- 4. target ------------------------------------------------------------
    classes = sorted(df[TARGET].unique().tolist())
    check("target classes", classes == TARGET_CLASSES, f"{classes} (expected {TARGET_CLASSES})")
    counts = df[TARGET].value_counts().to_dict()
    check("target is imbalanced", True,
          "; ".join(f"{k}={v} ({v / len(df) * 100:.1f}%)" for k, v in counts.items()))

    # -- 5. feature order (the real input contract) ---------------------------
    derived = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    check("derived feature order", derived == EXPECTED_FEATURE_ORDER,
          f"{len(derived)} features in CSV column order after dropping {NON_FEATURE_COLUMNS}")
    check("Overall_Risk_Score present", "Overall_Risk_Score" in derived,
          "this feature is an input AND dominates the model (~70% importance). The "
          "notebook author later dropped it as target leakage (cells 18/25/53), but "
          "the shipped model still consumes it.", warn_only=True)

    # -- 6. model artifact contract -------------------------------------------
    model_path = BASE_DIR / MODEL_FILE
    if not model_path.exists():
        check("model artifact present", False, f"{MODEL_FILE} not found")
        return report(args.json)

    model = joblib.load(model_path)
    check("model estimator type", type(model).__name__ == "RandomForestClassifier",
          f"{type(model).__name__} -- note the filename says 'xgb' but it is a RandomForest",
          warn_only=type(model).__name__ != "RandomForestClassifier")
    check("model expects 18 features", getattr(model, "n_features_in_", None) == N_FEATURES,
          f"n_features_in_={getattr(model, 'n_features_in_', None)}")
    check("model records feature names", getattr(model, "feature_names_in_", None) is not None,
          "feature_names_in_ is None: the model was fit on a numpy array, so feature "
          "ORDER is not recorded in the model and depends entirely on feature_names.pkl",
          warn_only=getattr(model, "feature_names_in_", None) is None)

    encoder_path, features_path = BASE_DIR / ENCODER_FILE, BASE_DIR / FEATURES_FILE
    if encoder_path.exists():
        le = joblib.load(encoder_path)
        mapping = {c: int(i) for i, c in enumerate(le.classes_)}
        check("label encoding", mapping == TARGET_ENCODING,
              f"{mapping} (expected {TARGET_ENCODING})")
    else:
        check("label encoder present", False, f"{ENCODER_FILE} not found")

    if features_path.exists():
        check("feature_names.pkl order", list(joblib.load(features_path)) == EXPECTED_FEATURE_ORDER,
              f"{len(joblib.load(features_path))} features, order matches CSV")
    else:
        check("feature_names.pkl present", False, f"{FEATURES_FILE} not found")

    check("model classes_ match encoder", list(getattr(model, "classes_", [])) == [0, 1, 2],
          f"classes_={list(getattr(model, 'classes_', []))} matches encoded labels 0/1/2")

    # -- 7. known gaps (reported, never fixed here) --------------------------
    check("scaler persisted", (BASE_DIR / SCALER_FILE).exists(),
          f"{SCALER_FILE} is absent. The model was fit on StandardScaler output but the "
          f"scaler was never saved, so app.py feeds raw values and the model collapses "
          f"to a single class. This is Issue #2, not fixed here.", warn_only=True)

    # -- 8. environment provenance -------------------------------------------
    origins = {
        MODEL_FILE: embedded_sklearn_version(model_path),
        ENCODER_FILE: embedded_sklearn_version(encoder_path),
        FEATURES_FILE: embedded_sklearn_version(features_path),
    }
    versions = {k: v for k, v in origins.items() if v}
    check("artifacts share one environment", len(set(versions.values())) <= 1,
          f"pickled with differing sklearn versions: {versions}; running sklearn "
          f"{sklearn.__version__}", warn_only=len(set(versions.values())) > 1)

    return report(args.json)


def report(as_json: bool) -> int:
    failed = [r for r in results if r["status"] == "FAIL"]
    warned = [r for r in results if r["status"] == "WARN"]

    if as_json:
        print(json.dumps({
            "results": results,
            "summary": {"passed": len(results) - len(failed) - len(warned),
                        "warned": len(warned), "failed": len(failed)},
        }, indent=2))
    else:
        width = max(len(r["check"]) for r in results)
        for r in results:
            mark = {"PASS": "ok  ", "WARN": "WARN", "FAIL": "FAIL"}[r["status"]]
            print(f"[{mark}] {r['check']:<{width}}  {r['detail']}")
        print()
        print(f"{len(results) - len(failed) - len(warned)} passed, "
              f"{len(warned)} warnings, {len(failed)} failed")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
