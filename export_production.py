"""
Export the verified model bundle into the production artifact layout.

The app runs from these four files -- nothing else -- and needs NO retraining:

    artifacts/production/
        model.pkl            full inference pipeline: StandardScaler -> model
        label_encoder.pkl    LabelEncoder (classes High / Low / Medium)
        feature_names.pkl    the 17 feature names, in pipeline input order
        metadata.json        model version, config, metrics, fingerprints, checksums

DELIBERATE DESIGN, documented (this is the Issue #2 lesson):
  * model.pkl is a Pipeline that embeds the fitted scaler. There is therefore no
    way to feed the forest unscaled input -- the exact failure mode Issue #2
    documented ("model expects 18 features... hand-written feature order... no
    scaler"). The layout still contains exactly the four files requested.
  * feature_names.pkl and label_encoder.pkl live in artifacts/production/, NOT
    at the repo root next to the deprecated files with the same names. Never
    load them by bare relative name; always resolve from this directory.

CALIBRATION (Issue #11 decision "A -- hybrid", user-approved):
  By default the exported model.pkl wraps the forest in a CalibratedModel:
    * predict()      -> raw forest argmax. Class labels are byte-for-byte what
                        train.py produced, so EVERY published metric (accuracy,
                        High recall/precision, macro-F1...) stays exactly valid.
    * predict_proba()-> per-class isotonic recalibration fitted on 5-fold
                        OOF probabilities of the SAME forest (clone, identical
                        hyperparameters, fixed scaler), rows renormalised to 1.
  The calibrators are fitted on the training split ONLY; the untouched 400-row
  held-out set is used to report before/after Brier and ECE in metadata.json.
  Use --no-calibrate to export the plain uncalibrated pipeline instead.

The export does NOT retrain. It loads artifacts/model_bundle.joblib (produced
and verified by train.py), re-wraps its scaler + forest, and proves by
construction that the production pipeline's predictions on the held-out split
are bit-identical to the bundle's predict path.

Run:  python export_production.py [--verify-only] [--no-calibrate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from train import (BASE_DIR, DATA_PATH, DATA_SHA256, EXPECTED_CLASSES,
                   LEAKY_FEATURE, MODEL_VERSION, OUT_DIR, RANDOM_STATE,
                   TEST_SIZE, TARGET, _final_estimator, current_env,
                   forest_sha256, sha256_of)
from calibrated_model import CalibratedModel
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: F401  (kept for joblib load)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROD_DIR = BASE_DIR / "artifacts" / "production"
MODEL_PKL = PROD_DIR / "model.pkl"
LABEL_PKL = PROD_DIR / "label_encoder.pkl"
FEATURES_PKL = PROD_DIR / "feature_names.pkl"
METADATA_JSON = PROD_DIR / "metadata.json"


def _held_out_split():
    """Recreate the exact train/test split train.py used (no training)."""
    df = pd.read_csv(DATA_PATH)
    feats = [c for c in df.columns if c not in ("Patient_ID", "Cancer_Type",
                                                TARGET, LEAKY_FEATURE)]
    y = LabelEncoder().fit_transform(df[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], y, test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE, stratify=y)
    return feats, X_tr, X_te, y_tr, y_te


def scaler_sha256(scaler) -> str:
    """
    Process-stable fingerprint of the fitted scaler (array bytes, not pickle
    bytes). Values are normalized to float64 so the byte layout cannot drift
    with dtype. Pickled FILE bytes are not stable across processes
    (PYTHONHASHSEED affects pickle output), which is why fingerprints of the
    model CONTENT are defined on arrays, not on file bytes.
    """
    h = hashlib.sha256()
    for arr in (scaler.mean_, scaler.var_, scaler.scale_):
        h.update(np.asarray(arr, dtype=np.float64).tobytes())
    h.update(np.int64(scaler.n_features_in_).tobytes())
    return h.hexdigest()


def calibration_sha256(calibrators) -> str:
    """Process-stable array fingerprint of the fitted isotonic calibrators."""
    h = hashlib.sha256()
    for iso in calibrators:
        for arr in (iso.X_thresholds_, iso.y_thresholds_):
            h.update(np.asarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def _fit_calibrators(forest, scaler, X_tr, y_tr) -> tuple[list, np.ndarray]:
    """
    Out-of-fold per-class isotonic calibrators for the SAME forest.

    Protocol (mirrors analyze_probability_calibration.py, Issue #11 evidence):
      5-fold stratified CV over the training split; each fold refits a clone of
      the forest with identical hyperparameters on the fold-train subset and
      predicts the fold-val; OOF probabilities feed one IsotonicRegression per
      class (target: 1{y == c}), with out_of_bounds='clip'.

    Only the training split is touched here; the held-out set is reserved for
    the honest before/after measurement in _verify_held_out().
    """
    X_tr_n = scaler.transform(X_tr.to_numpy())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros((len(X_tr), 3))
    for tr_i, va_i in skf.split(X_tr_n, y_tr):
        est = clone(forest)
        est.fit(X_tr_n[tr_i], y_tr[tr_i])
        pv = est.predict_proba(X_tr_n[va_i])
        order = {int(c): i for i, c in enumerate(est.classes_)}
        oof[va_i] = pv[:, [order[c] for c in range(3)]]
    calibrators = [
        IsotonicRegression(out_of_bounds="clip")
        .fit(oof[:, c], (y_tr == c).astype(float))
        for c in range(3)
    ]
    return calibrators, oof


def _raw_proba(model, scaler, X):
    """Forest vote-fractions in class order 0..2 (High, Low, Medium)."""
    p = model.predict_proba(scaler.transform(X))
    order = {int(c): i for i, c in enumerate(model.classes_)}
    return p[:, [order[c] for c in range(3)]]


def _ece_macro(proba, y_oh, n_bins: int = 10) -> float:
    """Mean over classes of ECE = sum_bins n/N * |predicted - observed|."""
    eces = []
    for c in range(3):
        p = proba[:, c]
        yy = y_oh[:, c]
        uniq = np.unique(np.round(p, 12))
        if len(uniq) > n_bins:
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        else:
            span = float(p.max() - p.min()) + 1e-9
            edges = np.unique(np.linspace(float(p.min()) - 0.001 * span,
                                          float(p.max()) + 0.001 * span, n_bins + 1))
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        e = 0.0
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            e += float(m.sum()) / len(p) * abs(float(p[m].mean()) - float(yy[m].mean()))
        eces.append(e)
    return float(np.mean(eces))


def export(out_dir: Path = PROD_DIR, calibrate: bool = True) -> tuple[dict, dict]:
    """Write the four production files. Returns (metadata, verify_report)."""
    if not (OUT_DIR / "model_bundle.joblib").exists():
        sys.exit("[FAIL] artifacts/model_bundle.joblib missing -- run `python train.py` first")

    bundle = joblib.load(OUT_DIR / "model_bundle.joblib")
    md = bundle["metadata"]

    # ---- contract asserts (mirrors train.py verify) -------------------------
    if LEAKY_FEATURE in bundle["feature_names"]:
        sys.exit(f"[FAIL] leaked feature {LEAKY_FEATURE} in bundle feature list")
    if md.get("dataset_sha256") != DATA_SHA256:
        sys.exit("[FAIL] bundle was trained on a different dataset check")

    forest = _final_estimator(bundle["model"])
    scaler = bundle["scaler"]
    features = list(bundle["feature_names"])

    feats, X_tr, X_te, y_tr, y_te = _held_out_split()

    calibration = None
    if calibrate:
        calibrators, _ = _fit_calibrators(forest, scaler, X_tr, y_tr)
        estimator = CalibratedModel(forest, calibrators)
        calibration = {
            "method": "per-class IsotonicRegression(out_of_bounds='clip') "
                      "(Issue #11 decision A: hybrid)",
            "fit_on": "5-fold stratified OOF probabilities of the SAME forest "
                      "(clone, identical hyperparameters, fixed scaler, "
                      "seed 0) -- training split only",
            "renormalised": "rows scaled to sum to 1 after per-class transforms",
            "labels_from": "raw forest argmax (Issue #10 metrics unchanged)",
            "sha256": calibration_sha256(calibrators),
        }
        print(f"[ok]   calibrated probabilities ON "
              f"(calibration {calibration['sha256'][:16]}...)")
    else:
        estimator = forest

    pipeline = Pipeline([("scaler", scaler), ("model", estimator)])

    reconstituted_sha = forest_sha256(forest)
    recorded_sha = md.get("forest_sha256")
    if recorded_sha is None:
        print("[warn] bundle predates forest_sha256 recording; relying on metric identity")
    elif reconstituted_sha != recorded_sha:
        sys.exit(f"[FAIL] forest fingerprint mismatch: {reconstituted_sha[:16]}... "
                 f"vs recorded {recorded_sha[:16]}...")
    print(f"[ok]   forest fingerprint matches the bundle's record ({recorded_sha[:16]}...)")

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema": 2,
        "artifact_set": "production",
        "model_version": md.get("model_version"),
        "trained_at": md.get("trained_at"),
        "exported_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entrypoint": "export_production.py",
        "input_bundle": {
            "path": "artifacts/model_bundle.joblib",
            "sha256": sha256_of(OUT_DIR / "model_bundle.joblib"),
        },
        "model_pkl": (
            "sklearn Pipeline [StandardScaler -> CalibratedModel]; scaler "
            "embedded; predict() = raw forest argmax (metrics unchanged), "
            "predict_proba() = OOF-calibrated probabilities (Issue #11 hybrid)"
            if calibrate else
            "sklearn Pipeline [StandardScaler -> RandomForestClassifier]; "
            "scaler embedded so no unscaled input can reach the forest"
        ),
        "target": TARGET,
        "class_order": EXPECTED_CLASSES,
        "label_mapping": md.get("label_mapping"),
        "resample": md.get("resample"),
        "excluded_features": md.get("excluded_features"),
        "config": md.get("config"),
        "dataset": {"path": DATA_PATH.name, "sha256": DATA_SHA256},
        "forest_sha256": recorded_sha,
        "scaler_fingerprint": scaler_sha256(scaler),
        "calibration": calibration,
        "environment": current_env(),
        "environment_matches_reference": current_env() == md.get("environment"),
        "metrics": md.get("metrics"),
        "files": {},
    }

    joblib.dump(pipeline, MODEL_PKL)
    joblib.dump(bundle["label_encoder"], LABEL_PKL)
    joblib.dump(features, FEATURES_PKL)
    # Checksums of the three pickles live inside metadata.json; metadata.json's
    # OWN checksum must live outside the file (a file can never contain its own
    # exact hash), so it is recorded in artifacts/manifest.json below.
    for name, payload in (("model.pkl", MODEL_PKL), ("label_encoder.pkl", LABEL_PKL),
                          ("feature_names.pkl", FEATURES_PKL)):
        metadata["files"][name] = {
            "sha256": sha256_of(payload),
            "bytes": payload.stat().st_size,
        }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["production"] = {
            "exported_at": metadata["exported_at"],
            "input_bundle_sha256": metadata["input_bundle"]["sha256"][:16],
            "calibration_sha256": (calibration or {}).get("sha256"),
            "metadata.json": {"sha256": sha256_of(METADATA_JSON)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[ok]   manifest records artifacts/production/metadata.json sha256")

    for f in (MODEL_PKL, LABEL_PKL, FEATURES_PKL, METADATA_JSON):
        print(f"[ok]   {f.relative_to(BASE_DIR)}  ({f.stat().st_size:,} bytes)")

    report = _verify_held_out(pipeline, bundle, features, calibrated=bool(calibration))
    if calibration is not None:
        metadata["calibration_holdout"] = report["calibration_stats"]
        METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["production"]["metadata.json"] = {"sha256": sha256_of(METADATA_JSON)}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for line in report["lines"]:
        print(line)
    return metadata, report


def _verify_held_out(pipeline, bundle, features, calibrated: bool = False) -> dict:
    """Score the held-out split through BOTH paths and require bit-identical."""
    _, _, X_te, _, y_te = _held_out_split()
    scaler, model = bundle["scaler"], bundle["model"]
    X_te_n = X_te[features].to_numpy()

    pred_prod = pipeline.predict(X_te_n)
    pred_bundle = model.predict(scaler.transform(X_te_n))

    identical = bool(np.array_equal(pred_prod, pred_bundle))
    acc = float((pred_prod == y_te).mean())
    le = bundle["label_encoder"]
    names = le.inverse_transform(pred_prod)
    dist = Counter(names)

    lines = [
        f"[{'ok' if identical else 'FAIL'}] production pipeline predictions "
        f"bit-identical to the bundle's predict path",
        f"[ok]   held-out accuracy through the production pipeline: {acc:.4f} "
        f"(published: {bundle['metadata']['metrics']['accuracy']:.4f})",
        f"[ok]   label encoder maps back to {sorted(dist)}",
    ]
    ok = identical and abs(acc - float(bundle["metadata"]["metrics"]["accuracy"])) < 1e-9
    lines.append(("PASS" if ok else "FAIL") + ": production export contract")
    lines.append(f"        predicted distribution: {dict(dist)}")

    stats = None
    if calibrated:
        # Labels are the raw forest argmax (byte-identical, proven above);
        # only the probabilities are recalibrated. Measured here on the
        # untouched 400 held-out rows.
        raw = _raw_proba(model, scaler, X_te_n)
        cal = pipeline.predict_proba(X_te_n)
        y_oh = (y_te[:, None] == np.arange(3)).astype(float)
        maxdiff = float(np.abs(raw - cal).max())
        changed = maxdiff > 1e-12
        brier = lambda p: float(np.mean(((p - y_oh) ** 2).sum(axis=1)))  # noqa: E731
        brier_raw, brier_cal = brier(raw), brier(cal)
        ece_raw, ece_cal = _ece_macro(raw, y_oh), _ece_macro(cal, y_oh)
        lines.append(f"[{'ok' if changed else 'FAIL'}] probabilities actually change "
                     f"under calibration (max abs diff {maxdiff:.4f})")
        lines.append(f"[ok]   Brier: raw {brier_raw:.4f} -> calibrated {brier_cal:.4f} "
                     f"({(brier_raw - brier_cal) / brier_raw * 100:+.1f}%)")
        lines.append(f"[ok]   ECE macro: raw {ece_raw:.4f} -> calibrated {ece_cal:.4f}")
        stats = {"brier_raw": brier_raw, "brier_calibrated": brier_cal,
                 "ece_macro_raw": ece_raw, "ece_macro_calibrated": ece_cal,
                 "labels_bit_identical": identical, "max_proba_diff": maxdiff}
        ok = ok and changed

    return {"ok": ok, "lines": lines, "accuracy": acc,
            "predicted_distribution": dict(dist), "calibration_stats": stats}


def verify_only(out_dir: Path = PROD_DIR) -> int:
    ok = True
    for f in (MODEL_PKL, LABEL_PKL, FEATURES_PKL, METADATA_JSON):
        exists = f.exists()
        print(f"[{'ok' if exists else 'FAIL'}] {f.relative_to(BASE_DIR)}")
        ok = ok and exists
    if not ok:
        return 1
    md = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    for name, rec in md.get("files", {}).items():
        path = out_dir / name
        want, got = rec["sha256"], sha256_of(path)
        state = "ok" if want == got else "FAIL"
        print(f"[{state}] {name} sha256 {got[:16]}...")
        ok = ok and (want == got)
    # metadata.json's own checksum is recorded externally, in the manifest.
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rec = manifest.get("production", {}).get("metadata.json", {})
        want = rec.get("sha256")
        if want:
            got = sha256_of(METADATA_JSON)
            state = "ok" if want == got else "FAIL"
            print(f"[{state}] metadata.json sha256 {got[:16]}... (via manifest.json)")
            ok = ok and (want == got)
        else:
            print("[warn] manifest has no production metadata.json checksum; re-run export_production.py")
            ok = False
    # Content identity is verified on ARRAY bytes (process-stable), because
    # pickle file bytes drift across processes with PYTHONHASHSEED. The file
    # sha256 checks above prove files are unchanged since this export; these
    # checks prove the CONTENT is the canonical model.
    bundle_path = OUT_DIR / "model_bundle.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        md_content = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
        prod_model = joblib.load(MODEL_PKL)
        forest = _final_estimator(prod_model)
        f_now, f_rec = forest_sha256(forest), md_content.get("forest_sha256")
        s_now = scaler_sha256(bundle["scaler"])
        s_rec = md_content.get("scaler_fingerprint")
        state = "ok" if f_now == f_rec and s_now == s_rec else "FAIL"
        print(f"[{state}] model content fingerprint matches the bundle "
              f"(forest {f_now[:16]}..., scaler {s_now[:16]}...)")
        ok = ok and (f_now == f_rec) and (s_now == s_rec)
        cal_rec = md_content.get("calibration") or {}
        cal_sha = cal_rec.get("sha256")
        if cal_sha:
            wrapped = _final_estimator(prod_model)
            if hasattr(wrapped, "calibrators"):
                cal_now = calibration_sha256(wrapped.calibrators)
                state = "ok" if cal_now == cal_sha else "FAIL"
                print(f"[{state}] calibrated-probabilities fingerprint matches "
                      f"metadata (calibration {cal_now[:16]}...)")
                ok = ok and (cal_now == cal_sha)
            else:
                print("[FAIL] model.pkl is not the calibrated hybrid but "
                      "metadata declares calibration")
                ok = False
    else:
        print("[warn] cannot check content fingerprints: bundle missing")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true",
                    help="check the existing production artifacts and exit")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="export the plain uncalibrated forest pipeline instead "
                         "of the Issue #11 calibrated hybrid (default)")
    args = ap.parse_args()
    if args.verify_only:
        return verify_only()
    export(calibrate=not args.no_calibrate)
    return 0


if __name__ == "__main__":
    sys.exit(main())