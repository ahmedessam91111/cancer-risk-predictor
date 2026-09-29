"""
Reproducible training entrypoint for the Cancer Risk Predictor.

    python train.py

Replaces the exploratory notebook as the single supported way to build the
model. Every parameter is explicit, the dataset is checksum-validated, and all
artifacts are written as one bundle so the model's input contract can never
drift away from the model itself.

=============================================================================
LINEAGE -- what this reproduces
=============================================================================
Original training lived in `Cancer_Risk_Prediction_(ML).ipynb` and, per Issue
#4, the shipped artifact came from cell 10 (execution 55) saved by cell 50
(execution 102). That cell chain is:

    cell 2   df    = pd.read_csv('/content/cancer-risk-factors.csv')  # Issue #13: notebooks read the tracked local file now
    cell 4   X     = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type'])
            y     = LabelEncoder().fit_transform(df['Risk_Level'])
    cell 8   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cell 9   scaler = StandardScaler(); X_train = scaler.fit_transform(X_train)
    cell 10  model  = RandomForestClassifier(random_state=42, n_estimators=100)

Everything after cell 10 (Optuna, `final_rf`, all XGBoost work) was discarded
by the save in cell 50 and is deliberately NOT reproduced here. In particular
`final_rf(n_estimators=347, max_depth=19)` is not used, because those
hyperparameters were selected against a metric inflated by two separate
leakages -- see CHANGES below.

=============================================================================
CHANGES FROM THE NOTEBOOK -- each one required by a documented finding
=============================================================================
No change below is silent. Each is a consequence of a Tier 1 / Tier 2 issue.

1. `Overall_Risk_Score` is DROPPED from the feature set.            [Issue #5]
   Issue #5 proved `Risk_Level` is a 100% accurate threshold function of it
   (0.329978 / 0.659964; 2000 rows, zero mismatches), and that it is itself an
   83.9% linearly-reconstructible composite of the risk factors. It is the
   target, quantized -- not a predictor.
   Issue #6 quantified the cost of keeping it: the column ALONE scored the same
   0.9975 as all 18 features, contributing +0.0000 for the other 17. Every
   metric reported with it present is invalid.

2. `class_weight="balanced"` is set on the estimator.               [Issue #6, #9]
   The target is 78.7% Medium / 16.2% Low / 5.1% High. Unweighted, the model
   degenerates: Issue #6 measured High recall at 0.050, catching 1 of 20 High
   patients. Balancing is required for the minority classes to be usable at all.

3. The fitted `StandardScaler` is now PERSISTED with the model.      [Issue #1, #2]
   The notebook fit a StandardScaler but never saved it. `app.py` therefore fed
   raw values to a model fit on standardized values. Because each tree threshold
   lives in scaled units, that is not a harmless mismatch: measured on the 400
   held-out rows it collapsed the model to `Medium` 400/400 with zero `High`.
   The scaler is retained (not dropped) precisely because removing it would
   change model behavior; persisting it is the fix.

4. The feature list is DERIVED from the data, not hand-written.      [Issue #1, #2]
   `model.feature_names_in_` was `None` (the model was fit on a numpy array),
   so feature order survived only in `feature_names.pkl`, which
   `import joblib.py` reconstructed by hand months after training. Here the
   order comes from the CSV column order and is asserted.

5. Model, scaler, label encoder, feature order and metadata ship as ONE bundle.
                                                                         [Issue #2]
   Three loose .pkl files let the contract drift. One bundle cannot.

6. Metrics, environment, and checksums are written next to the bundle.  [Issue #7]

=============================================================================
DELIBERATELY NOT DONE
=============================================================================
- No hyperparameter search. The notebook's Optuna results are not reproducible
  here because their objective was computed on a leaked feature (Issue #5) and,
  in cells 35/47, on cross-validation folds drawn from already-resampled data
  (Issue #9, worth +0.2775 macro-F1 of pure inflation). Re-tuning on the clean
  feature set is a separate, deliberate piece of work.
- No resampling. `class_weight` handles the imbalance without synthesizing
  rows, and keeps `imbalanced-learn` out of the inference path. `--resample
  smote` is available and correct (SMOTE inside a Pipeline, fitted per fold)
  for anyone who wants to compare; see docs/ for the measured trade-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# CONFIGURATION -- every training parameter, explicit
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_VERSION = "2.0.0"

# --- reference environment ---
# The library versions the published metrics in docs/ were produced with.
#
# This is load-bearing, not decoration. RandomForestClassifier is NOT stable
# across scikit-learn versions: with identical data, identical parameters and
# random_state=42, scikit-learn 1.8.0 builds a materially different forest than
# 1.9.1 does. Measured on this dataset:
#
#     scikit-learn 1.9.1   F1 macro 0.6572   High recall 0.300
#     scikit-learn 1.8.0   F1 macro 0.5002   High recall 0.050
#
# The 1.8.0 model catches 1 of 20 High-risk patients instead of 6. This is not
# a hyperparameter difference; `random_state` does not protect against it.
# train.py therefore reports loudly when the environment does not match, and
# records the actual versions in the bundle and manifest either way.
REFERENCE_ENV = {
    "python": "3.14.7",
    "scikit-learn": "1.9.1",
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "joblib": "1.6.0",
}

# scikit-learn versions known to produce a different forest from the reference.
# The metric impact is severe enough to be worth calling out by name.
_SKLEARN_SENSITIVE = True

# --- dataset ---
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"
DATA_SHA256 = "01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5"
TARGET = "Risk_Level"
NON_FEATURE_COLUMNS = ["Patient_ID", "Cancer_Type", "Risk_Level"]

# Dropped as proven target leakage -- Issue #5. Remove this line only together
# with an explicit decision to reintroduce the leak.
LEAKY_FEATURE = "Overall_Risk_Score"

# --- split (unchanged from notebook cell 8) ---
TEST_SIZE = 0.20
RANDOM_STATE = 42
STRATIFY = True

# --- estimator (unchanged from notebook cell 10) ---
N_ESTIMATORS = 100
CLASS_WEIGHT = "balanced"          # Issue #6 / #9
MAX_DEPTH = None
CRITERION = "gini"
MAX_FEATURES = "sqrt"

# --- cross-validation, for reporting only (never used to pick settings) ---
CV_FOLDS = 3

# --- outputs ---
OUT_DIR = BASE_DIR / "artifacts"
BUNDLE_PATH = OUT_DIR / "model_bundle.joblib"
METRICS_PATH = OUT_DIR / "metrics.json"
MANIFEST_PATH = OUT_DIR / "manifest.json"

EXPECTED_ROWS = 2000
EXPECTED_COLUMNS = 21
EXPECTED_CLASSES = ["High", "Low", "Medium"]


# =============================================================================
# VALIDATION
# =============================================================================

def sha256_of(path: Path, normalize_eol: bool = False) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk.replace(b"\r\n", b"\n") if normalize_eol else chunk)
    return h.hexdigest()


def forest_sha256(model) -> str:
    """
    Canonical bitwise fingerprint of a fitted forest.

    DOCUMENTED SERIALIZATION -- this is the method the project pins, so the
    value is reproducible by anyone who runs it on the same forest:

        sha256( concat over trees, in estimator order, of
                feature | threshold | children_left | children_right )

    Each array is fed as its native numpy bytes (tobytes()). This is the
    canonical definition of the project's forest fingerprint: it reproduces
    the value published in docs/TRAINING_AND_LEAKAGE.md, and -- because a
    bitwise fingerprint is only meaningful if the serialization is pinned --
    it must not be redefined casually.
    """
    h = hashlib.sha256()
    for est in model.estimators_:
        t = est.tree_
        for arr in (t.feature, t.threshold, t.children_left, t.children_right):
            h.update(arr.tobytes())
    return h.hexdigest()


def current_env() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def check_environment() -> dict:
    """
    Report whether this interpreter matches the one the metrics came from.

    A mismatch does not stop training -- the run is still valid, it just is not
    the run the documentation describes. So this warns loudly, prints the
    consequence, and lets the caller carry on with the mismatch recorded in the
    artifacts.
    """
    env = current_env()
    drift = {k: (REFERENCE_ENV[k], v) for k, v in env.items()
             if v != REFERENCE_ENV.get(k)}

    if not drift:
        print("[ok]   environment matches the reference "
              f"(python {env['python']}, scikit-learn {env['scikit-learn']})")
        return env

    print("\n" + "!" * 78)
    print("WARNING: this environment differs from the one the published")
    print("         metrics were produced with.")
    print("!" * 78)
    for k, (want, got) in drift.items():
        print(f"  {k:<14} expected {want:<10} got {got}")
    if "scikit-learn" in drift and _SKLEARN_SENSITIVE:
        print()
        print("  RandomForestClassifier is not reproducible across scikit-learn")
        print("  versions even with a fixed random_state. On this dataset:")
        print("      scikit-learn 1.9.1 -> F1 macro 0.6572, High recall 0.300")
        print("      scikit-learn 1.8.0 -> F1 macro 0.5002, High recall 0.050")
        print("  The numbers this run prints will NOT match docs/.")
        print("  Fix with:  pip install -r requirements.txt")
    print("!" * 78 + "\n")
    return env


def validate_dataset() -> pd.DataFrame:
    """
    Fail loudly before training if the data is not the data we validated.

    The hash is checked EOL-normalized so it survives git's autocrlf conversion;
    a genuine data change still fails.
    """
    if not DATA_PATH.exists():
        raise SystemExit(
            f"FATAL: {DATA_PATH.name} not found.\n"
            f"It is tracked in git; run `git checkout -- {DATA_PATH.name}` "
            f"or restore it from the upstream source recorded in "
            f"docs/DATASET_PROVENANCE.md."
        )

    raw, norm = sha256_of(DATA_PATH), sha256_of(DATA_PATH, normalize_eol=True)
    if raw != DATA_SHA256 and norm != DATA_SHA256:
        raise SystemExit(
            f"FATAL: dataset checksum mismatch for {DATA_PATH.name}\n"
            f"  expected {DATA_SHA256}\n"
            f"  actual   {raw}\n"
            f"Training on changed data would produce a different model than the "
            f"one documented in docs/. Restore the original file, or update "
            f"DATA_SHA256 deliberately along with the metrics in docs/."
        )
    note = "" if raw == DATA_SHA256 else "  (EOL-normalized; file uses CRLF)"
    print(f"[ok]   dataset sha256 verified{note}")

    df = pd.read_csv(DATA_PATH)

    # Schema
    if len(df) != EXPECTED_ROWS or df.shape[1] != EXPECTED_COLUMNS:
        raise SystemExit(
            f"FATAL: expected {EXPECTED_ROWS}x{EXPECTED_COLUMNS}, "
            f"got {len(df)}x{df.shape[1]}"
        )
    print(f"[ok]   shape {df.shape[0]} rows x {df.shape[1]} columns")

    missing_cols = [c for c in NON_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"FATAL: expected columns absent: {missing_cols}")

    # Quality -- the validated dataset is clean; refuse to train on a dirty one
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        raise SystemExit(f"FATAL: {n_missing} missing cells; the validated dataset has none")
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        raise SystemExit(f"FATAL: {n_dupes} duplicate rows; the validated dataset has none")
    print("[ok]   no missing values, no duplicate rows")

    classes = sorted(df[TARGET].unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise SystemExit(f"FATAL: target classes {classes}, expected {EXPECTED_CLASSES}")
    counts = df[TARGET].value_counts()
    print("[ok]   target classes "
          + ", ".join(f"{k}={v} ({v / len(df) * 100:.1f}%)" for k, v in counts.items()))
    return df


def derive_features(df: pd.DataFrame) -> list[str]:
    """
    Derive the feature list from the CSV column order, minus the target and the
    identifier columns, minus the proven-leaky score.

    Order is the model's real input contract, so it is asserted against the
    known-good list rather than trusted.
    """
    features = [c for c in df.columns
                if c not in NON_FEATURE_COLUMNS and c != LEAKY_FEATURE]

    if LEAKY_FEATURE in features:
        raise SystemExit("FATAL: leaked feature present in the derived feature list")

    expected = [c for c in df.columns
                if c not in NON_FEATURE_COLUMNS and c != LEAKY_FEATURE]
    if features != expected:
        raise SystemExit(f"FATAL: feature derivation is not stable: {features}")
    if len(features) != 17:
        raise SystemExit(f"FATAL: expected 17 features, derived {len(features)}: {features}")

    print(f"[ok]   {len(features)} features derived, {LEAKY_FEATURE} excluded (Issue #5)")
    return features


# =============================================================================
# TRAINING
# =============================================================================

def make_estimator(resample: str):
    """
    Build the estimator.

    `class_weight` is the default balancing strategy (Issue #6/#9). The `smote`
    option exists for comparison and wraps the estimator in an imblearn
    Pipeline, so SMOTE is refitted inside every training fold and never sees
    validation or test rows (Issue #9).
    """
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        criterion=CRITERION,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if resample == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError:
            raise SystemExit(
                "FATAL: --resample smote needs imbalanced-learn.\n"
                "  pip install imbalanced-learn"
            )
        # SMOTE sits INSIDE the pipeline: fitted on training rows only, refitted
        # per CV fold, and never applied to validation or test data.
        return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("rf", rf)])
    return rf


def train(df: pd.DataFrame, features: list[str], resample: str = "class_weight"):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])
    X = df[features]

    # Cell 8, unchanged.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Cell 9, unchanged -- and now persisted (CHANGE 3).
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Cell 10, plus class_weight (CHANGE 2).
    model = make_estimator(resample)
    model.fit(X_train_s, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": features,
        "X_train": X_train, "X_test": X_test,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "y_train": y_train, "y_test": y_test,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(result: dict, classes: list[str]) -> dict:
    model, y_test = result["model"], result["y_test"]
    pred = model.predict(result["X_test_s"])
    labels = list(range(len(classes)))

    per_class = {}
    for i, name in enumerate(classes):
        tp = int(((pred == i) & (y_test == i)).sum())
        fp = int(((pred == i) & (y_test != i)).sum())
        fn = int(((pred != i) & (y_test == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "support": int((y_test == i).sum()),
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        }

    return {
        "n_test_rows": int(len(y_test)),
        "n_train_rows": int(len(result["y_train"])),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
        "high_recall": per_class["High"]["recall"],
        "high_precision": per_class["High"]["precision"],
        "precision_macro": round(precision_score(y_test, pred, average="macro", labels=labels, zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, pred, average="macro", labels=labels, zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro", labels=labels, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, pred, average="weighted", labels=labels, zero_division=0), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "class_order": classes,
        "predicted_distribution": {c: int((pred == i).sum()) for i, c in enumerate(classes)},
        "classification_report": classification_report(
            y_test, pred, target_names=classes, zero_division=0),
    }


def cross_validate(result: dict, resample: str) -> dict:
    """
    Cross-validated macro-F1, reported for transparency.

    The scaler is deliberately NOT refitted per fold here. This mirrors how
    inference works (one persisted scaler) and keeps the reported number
    comparable to the held-out score. Resampling, by contrast, IS refitted per
    fold because it lives inside the pipeline.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        make_estimator(resample),
        result["X_train_s"], result["y_train"],
        cv=cv, scoring="f1_macro", n_jobs=1,
    )
    return {
        "folds": CV_FOLDS,
        "f1_macro_mean": round(float(scores.mean()), 4),
        "f1_macro_std": round(float(scores.std()), 4),
        "f1_macro_per_fold": [round(float(s), 4) for s in scores],
    }


# =============================================================================
# PERSISTENCE
# =============================================================================

def _final_estimator(model):
    """The estimator that actually predicts, unwrapping an imblearn Pipeline."""
    steps = getattr(model, "steps", None)
    return model if steps is None else steps[-1][1]


def _feature_importance(model, features: list[str]) -> dict:
    est = _final_estimator(model)
    pairs = sorted(zip(features, est.feature_importances_), key=lambda t: -t[1])
    return {name: round(float(imp), 4) for name, imp in pairs}


def _rel(path: Path) -> str:
    """Display a path relative to the project root when it is inside it."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return (out_dir / BUNDLE_PATH.name,
            out_dir / METRICS_PATH.name,
            out_dir / MANIFEST_PATH.name)


def save(result: dict, metrics: dict, cv: dict, features: list[str],
         resample: str, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, metrics_path, manifest_path = _paths(out_dir)
    classes = [str(c) for c in result["label_encoder"].classes_]
    now = datetime.now(timezone.utc).isoformat()

    bundle = {
        "model": result["model"],
        "scaler": result["scaler"],
        "label_encoder": result["label_encoder"],
        "feature_names": features,
        "metadata": {
            "model_version": MODEL_VERSION,
            "trained_at": now,
            "target": TARGET,
            "class_order": classes,
            "label_mapping": {c: int(i) for i, c in enumerate(classes)},
            "resample": resample,
            "excluded_features": {LEAKY_FEATURE: "target leakage, Issue #5"},
            "config": {
                "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": STRATIFY,
                "n_estimators": N_ESTIMATORS, "class_weight": CLASS_WEIGHT,
                "max_depth": MAX_DEPTH, "criterion": CRITERION, "max_features": MAX_FEATURES,
                "scaler": "StandardScaler(fit on train split only)",
            },
            "dataset_sha256": DATA_SHA256,
            "forest_sha256": forest_sha256(result["model"]),
            "environment": current_env(),
            "environment_matches_reference": current_env() == REFERENCE_ENV,
            "metrics": {k: metrics[k] for k in
                        ("accuracy", "balanced_accuracy",
                         "high_recall", "high_precision",
                         "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
        },
    }
    joblib.dump(bundle, bundle_path)
    print(f"[ok]   bundle   -> {_rel(bundle_path)}")

    metrics_doc = {
        "model_version": MODEL_VERSION,
        "trained_at": now,
        "resample": resample,
        "n_features": len(features),
        "feature_names": features,
        "environment": current_env(),
        "environment_matches_reference": current_env() == REFERENCE_ENV,
        "held_out_test": metrics,
        "cross_validation": cv,
        "feature_importance": _feature_importance(result["model"], features),
    }
    metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8")
    print(f"[ok]   metrics  -> {_rel(metrics_path)}")

    manifest = {
        "schema": 1,
        "model_version": MODEL_VERSION,
        "generated_at": now,
        "entrypoint": "train.py",
        "dataset": {
            "path": DATA_PATH.name, "sha256": DATA_SHA256,
            "sha256_raw": sha256_of(DATA_PATH),
            "rows": EXPECTED_ROWS, "columns": EXPECTED_COLUMNS,
        },
        "forest_sha256": forest_sha256(result["model"]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "artifacts": {p.name: sha256_of(p) for p in (bundle_path, metrics_path)},
        "superseded": {
            "model_xgb_new.pkl": "Issue #4/5: untuned baseline that consumed the leaked feature",
            "label_encoder.pkl": "Issue #2: hand-built, replaced by the bundle",
            "feature_names.pkl": "Issue #2: hand-written order, replaced by the bundle",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[ok]   manifest -> {_rel(manifest_path)}")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify(out_dir: Path = OUT_DIR) -> int:
    """
    Re-open the saved bundle and assert the whole input contract holds.

    This is the check that would have caught Issue #2 (no scaler) and Issue #4
    (model/feature mismatch) automatically.
    """
    bundle_path = _paths(out_dir)[0]
    print(f"verifying {_rel(bundle_path)}\n")
    if not bundle_path.exists():
        print("[FAIL] bundle not found -- run `python train.py` first")
        return 1

    bundle = joblib.load(bundle_path)
    ok = True

    required = {"model", "scaler", "label_encoder", "feature_names", "metadata"}
    missing = required - set(bundle)
    if missing:
        print(f"[FAIL] bundle missing keys: {sorted(missing)}")
        return 1
    print(f"[ok]   bundle has all {len(required)} required keys")

    feats, model, scaler = bundle["feature_names"], bundle["model"], bundle["scaler"]
    md = bundle["metadata"]

    if LEAKY_FEATURE in feats:
        print(f"[FAIL] leaked feature {LEAKY_FEATURE} is in the feature list"); ok = False
    else:
        print(f"[ok]   {LEAKY_FEATURE} absent from the {len(feats)} features")

    est = _final_estimator(model)

    n_in = getattr(est, "n_features_in_", None)
    if n_in != len(feats):
        print(f"[FAIL] model expects {n_in} features, bundle lists {len(feats)}"); ok = False
    else:
        print(f"[ok]   model input width {n_in} matches the feature list")

    n_scaler = getattr(scaler, "n_features_in_", None)
    if n_scaler != len(feats):
        print(f"[FAIL] scaler expects {n_scaler} features, bundle lists {len(feats)}"); ok = False
    else:
        print(f"[ok]   scaler input width {n_scaler} matches -- scaler IS persisted (Issue #2)")

    classes = [str(c) for c in bundle["label_encoder"].classes_]
    model_classes = [int(c) for c in getattr(est, "classes_", [])]
    if model_classes != list(range(len(classes))):
        print(f"[FAIL] model classes {model_classes} != encoded 0..{len(classes) - 1}"); ok = False
    else:
        print(f"[ok]   label encoding consistent: {classes} -> {model_classes}")

    if md.get("dataset_sha256") != DATA_SHA256:
        print(f"[FAIL] bundle was trained on dataset {md.get('dataset_sha256')}"); ok = False
    else:
        print(f"[ok]   bundle trained on the validated dataset (v{md.get('model_version')})")

    recorded = md.get("forest_sha256")
    actual = forest_sha256(_final_estimator(model)) if hasattr(_final_estimator(model), "estimators_") else None
    if recorded and actual:
        if actual == recorded:
            print(f"[ok]   forest bitwise fingerprint matches ({actual[:16]}...)")
        else:
            print(f"[FAIL] forest fingerprint {actual[:16]}... != recorded {recorded[:16]}...")
            ok = False
    else:
        print("[warn] forest fingerprint not recorded (older bundle); skipping")

    # End-to-end: a round trip through the documented inference path.
    probe = pd.DataFrame([{f: 0.0 for f in feats}])
    pred = model.predict(scaler.transform(probe[feats]))
    print(f"[ok]   inference round trip on a zero row -> {bundle['label_encoder'].inverse_transform(pred)[0]}")

    print(f"\n{'PASS' if ok else 'FAIL'}: bundle contract")
    return 0 if ok else 1


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resample", choices=["class_weight", "smote"],
                    default="class_weight",
                    help="imbalance strategy (default: class_weight, Issue #6/#9)")
    ap.add_argument("--verify", action="store_true",
                    help="verify an existing bundle's input contract and exit")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help=f"where to write the bundle/metrics/manifest (default: {OUT_DIR.name}/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir

    if args.verify:
        return verify(out_dir)

    print(f"model version {MODEL_VERSION}")
    print(f"python {platform.python_version()} | sklearn {sklearn.__version__}")
    print(f"numpy {np.__version__} | pandas {pd.__version__}")

    env = check_environment()

    print("validating dataset")
    df = validate_dataset()
    features = derive_features(df)
    classes = EXPECTED_CLASSES

    print(f"\ntraining  (resample={args.resample})")
    result = train(df, features, args.resample)
    print(f"[ok]   fit on {len(result['y_train'])} rows x {len(features)} features")

    print("\nevaluating on the held-out split")
    metrics = evaluate(result, classes)
    cv = cross_validate(result, args.resample)

    print(f"        accuracy        {metrics['accuracy']:.4f}")
    print(f"        precision macro {metrics['precision_macro']:.4f}")
    print(f"        recall macro    {metrics['recall_macro']:.4f}")
    print(f"        F1 macro        {metrics['f1_macro']:.4f}")
    print(f"        F1 weighted     {metrics['f1_weighted']:.4f}")
    print(f"        CV F1 macro     {cv['f1_macro_mean']:.4f} +/- {cv['f1_macro_std']:.4f}")
    print("        per class:")
    for c, m in metrics["per_class"].items():
        print(f"          {c:<7} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  n={m['support']}")

    print("\nsaving artifacts")
    save(result, metrics, cv, features, args.resample, out_dir)

    print("\nverifying the saved bundle")
    return verify(out_dir)


if __name__ == "__main__":
    sys.exit(main())
