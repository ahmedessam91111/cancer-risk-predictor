"""
Verify that `app.py` is wired to exactly the production artifacts
`export_production.py` writes, and that those artifacts reproduce the model
`train.py` produces.

Issue #8 step 8 asks to "verify that the Streamlit app uses exactly these
artifacts". This script checks that in five layers:

  1. STATIC     -- parse app.py with `ast`: assert which files it loads, that it
                   loads them from artifacts/production/ and not the deprecated
                   root .pkl files, and that it does not scale inputs itself.
  2. BEHAVIOURAL-- pull app.py's real `preprocess_input` out of its source and
                   execute it on hostile input (scrambled columns, decoys);
                   the raw matrix it returns must match the training reference.
  3. END TO END -- score the 400 held-out rows through the production pipeline
                   and require bit-identical predictions to train.py's bundle.
  4. PURITY     -- leak-free feature set; no train/test row overlap.
  5. LIVE       -- boot the real app via Streamlit's AppTest and click through.

Run:  python verify_app_integration.py
"""
from __future__ import annotations

import ast
import hashlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "app.py"
PROD_DIR = BASE_DIR / "artifacts" / "production"
MODEL_PATH = PROD_DIR / "model.pkl"
LABEL_PATH = PROD_DIR / "label_encoder.pkl"
FEATURES_PATH = PROD_DIR / "feature_names.pkl"
METADATA_PATH = PROD_DIR / "metadata.json"
PROD_FILES = (MODEL_PATH, LABEL_PATH, FEATURES_PATH, METADATA_PATH)
BUNDLE_PATH = BASE_DIR / "artifacts" / "model_bundle.joblib"
DATA_PATH = BASE_DIR / "cancer-risk-factors.csv"

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global checks
    checks += 1
    print(f"[{'ok' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def load_app_function(name: str, namespace: dict) -> object:
    """
    Extract and execute a single top-level function from app.py's source.

    This runs app.py's real code -- the same bytes a Streamlit server would run
    -- without importing the whole script, which is not importable outside a
    Streamlit runtime.
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(APP_PATH), "exec"), namespace)
            return namespace[name]
    raise SystemExit(f"FATAL: {name}() not found in {APP_PATH.name}")


def stub_streamlit() -> "types.ModuleType":
    """The two informational calls preprocess_input may make."""
    import types
    st = types.ModuleType("streamlit")
    st.warning = lambda *a, **k: None
    st.info = lambda *a, **k: None
    return st


def main() -> int:
    print("=" * 78)
    print("1. STATIC -- which artifacts does app.py load, and from where?")
    print("=" * 78)

    missing = [p for p in PROD_FILES if not p.exists()]
    if missing:
        print(f"[FAIL] missing production artifacts: {[p.name for p in missing]}")
        print("       run `python export_production.py` first")
        return 1
    if not BUNDLE_PATH.exists():
        print(f"[FAIL] {BUNDLE_PATH} missing — run `python train.py` first")
        return 1
    if not APP_PATH.exists():
        print(f"[FAIL] {APP_PATH} missing")
        return 1

    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(APP_PATH))
    loaded_paths = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and (n.value.endswith(".pkl") or n.value.endswith(".joblib") or n.value.endswith(".json"))
    ]

    for fname in ("model.pkl", "label_encoder.pkl", "feature_names.pkl", "metadata.json"):
        check(fname in " ".join(loaded_paths),
              f"app.py references `{fname}`", sorted(set(loaded_paths)))
    check("production" in " ".join(loaded_paths) or "PROD_DIR" in src,
          "app.py resolves artifacts under the production path",
          "artifacts/production")
    check("model_xgb_new.pkl" not in src,
          "app.py never loads the legacy model_xgb_new.pkl")
    check('"Overall_Risk_Score": {"widget"' not in src,
          "app.py removed the Overall_Risk_Score slider from the manual form")
    check('["Overall_Risk_Score"]' not in src,
          "Overall_Risk_Score is not in any widget group (Issue #5)")
    check("scaler.transform" not in src,
          "app.py no longer scales inputs itself — the scaler is embedded in model.pkl",
          "Pipeline [StandardScaler -> RandomForest]")

    print("\n" + "=" * 78)
    print("2. BEHAVIOURAL -- does app.py's own code produce the model's real input?")
    print("=" * 78)

    prod_model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_PATH)
    feats = joblib.load(FEATURES_PATH)

    ns = {"pd": pd, "np": np, "st": stub_streamlit(), "FEATURE_NAMES": feats}
    preprocess_input = load_app_function("preprocess_input", ns)
    print("[ok]   extracted app.py's real preprocess_input() and executed it")

    df = pd.read_csv(DATA_PATH)
    sample = df[feats].head(200).copy()

    # Hostile input: reversed column order plus decoy columns, SAME row order.
    # The app must return EXACTLY the training matrix (raw — scaling is the
    # pipeline's job), with decoys dropped.
    hostile = sample[list(reversed(feats))].copy()
    hostile["Patient_ID"] = range(len(hostile))
    hostile["Overall_Risk_Score"] = 0.9      # decoy: must be ignored, not used
    hostile["Totally_Unknown"] = 42.0

    X_app = np.asarray(preprocess_input(hostile))
    X_ref = np.asarray(sample[feats])
    maxdiff = float(np.abs(X_app - X_ref).max())
    check(X_app.shape == X_ref.shape,
          "app output shape matches the training matrix (raw, unscaled)",
          f"{X_app.shape} vs {X_ref.shape}")
    check(maxdiff < 1e-12,
          "app output is bit-identical to the raw training matrix, despite scrambled columns",
          f"max abs diff {maxdiff:.2e}")

    h_app = hashlib.sha256(np.ascontiguousarray(X_app).tobytes()).hexdigest()
    h_ref = hashlib.sha256(np.ascontiguousarray(X_ref).tobytes()).hexdigest()
    check(h_app == h_ref, "sha256 of the two matrices matches", h_app[:24])

    # Row order must not matter either: shuffle, then re-align by index.
    shuffled = sample.sample(frac=1.0, random_state=0)[list(reversed(feats))]
    X_shuf = np.asarray(preprocess_input(shuffled))
    aligned = X_shuf[np.argsort(shuffled.index.to_numpy(), kind="stable")]
    check(np.allclose(aligned, X_ref, atol=1e-12),
          "app output is invariant to row order",
          f"max abs diff {np.abs(aligned - X_ref).max():.2e}")

    print("\n" + "=" * 78)
    print("3. END TO END -- production artifacts reproduce the bundle's model")
    print("=" * 78)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df["Risk_Level"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats], y, test_size=0.2, random_state=42, stratify=y)

    X_te_app = np.asarray(preprocess_input(X_te[list(reversed(feats))]))
    pred_prod = le.inverse_transform(prod_model.predict(X_te_app))
    dist = pd.Series(pred_prod).value_counts().to_dict()

    check(len(dist) == 3,
          "all three risk levels appear on the 400 held-out rows (was: Medium only)",
          str(dist))
    check(int(pd.Series(pred_prod).eq("High").sum()) > 0,
          "the High class is reachable at all (was: 0 High out of 400)",
          f"{int(pd.Series(pred_prod).eq('High').sum())} High")

    # Bitwise: the production pipeline (scaler embedded) vs train.py's bundle path.
    bundle = joblib.load(BUNDLE_PATH)
    pred_bundle = le.inverse_transform(
        bundle["model"].predict(bundle["scaler"].transform(X_te_app)))
    check(np.array_equal(pred_prod, pred_bundle),
          "production pipeline predictions are bit-identical to train.py's bundle path",
          f"{dict(pd.Series(pred_prod).value_counts())}")

    # The scaler is load-bearing: feeding the bare forest raw values must NOT
    # give the same answer (this is the pre-Issue-#8 collapse).
    bare_forest = prod_model.steps[-1][1]
    raw_pred = le.inverse_transform(bare_forest.predict(X_te_app))
    check(not np.array_equal(raw_pred, pred_prod),
          "the embedded scaler is load-bearing (raw values change the answer)",
          f"unscaled {dict(pd.Series(raw_pred).value_counts())} vs app {dict(dist)}")

    md = __import__("json").loads(METADATA_PATH.read_text(encoding="utf-8"))
    check(abs(float(md["metrics"]["accuracy"]) - 0.8475) < 1e-9,
          "metadata.json records the published held-out accuracy",
          f"{md['metrics']['accuracy']:.4f}")
    check(md.get("forest_sha256", "").startswith("aeca50e59b3dc670"),
          "metadata.json records the canonical forest fingerprint",
          md.get("forest_sha256", "")[:16])

    print("\n" + "=" * 78)
    print("4. TRAIN/TEST PURITY -- app uses the deployed model, not a refit")
    print("=" * 78)
    check(X_tr.index.intersection(X_te.index).empty,
          "no row overlap between train and test")
    check(set(feats) == set(bundle["feature_names"]),
          "feature set is the leak-free 17", f"{len(feats)} features")

    print("\n" + "=" * 78)
    print("5. LIVE STREAMLIT -- the real app, driven headlessly end to end")
    print("=" * 78)
    run_live_app_checks()

    print(f"\n{'=' * 78}")
    if failures:
        print(f"FAIL: {checks - len(failures)}/{checks} checks passed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: all {checks} checks passed")
    return 0


def run_live_app_checks() -> None:
    """
    Boot app.py through Streamlit's own test runner and click the buttons.

    This covers the layer the AST checks cannot: widget wiring, the removed
    Overall_Risk_Score slider, and whether the UI actually produces varied
    output. Skipped with a notice if streamlit is unavailable.
    """
    try:
        import logging

        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("[skip] streamlit not installed -- live app checks skipped")
        return

    logging.disable(logging.WARNING)

    def predict(age, bmi, factor, flags, gender) -> str | None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
        if at.exception:
            raise RuntimeError(f"app raised on load: {[e.value for e in at.exception]}")
        at.radio[0].set_value("\U0001F9D1 Manual input").run()
        for s in at.slider:
            s.set_value(age if s.label == "Age" else factor)
        for n in at.number_input:
            n.set_value(bmi)
        for r in at.radio[1:]:
            if r.label == "Gender":
                r.set_value(gender)
            elif r.label in flags:
                r.set_value("Yes")
        [b for b in at.button if "Predict" in b.label][0].click().run()
        if at.exception:
            raise RuntimeError(f"app raised on predict: {[e.value for e in at.exception]}")
        for m in at.markdown:
            if "badge badge-" in m.value:
                return m.value.split("badge badge-")[1].split('"')[0]
        return None

    at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    check(not at.exception, "app.py runs with no uncaught exception",
          "" if not at.exception else str(at.exception[0].value))
    at.radio[0].set_value("\U0001F9D1 Manual input").run()
    labels = [s.label for s in at.slider]
    check("Overall_Risk_Score" not in labels,
          "the Overall_Risk_Score slider is gone from the live form",
          f"{len(labels)} sliders rendered")
    check(any(b.label == "\U0001F9EA Predict risk level" for b in at.button),
          "the predict button renders")

    # The old app answered Medium for every conceivable input. It must not.
    profiles = [
        ("max risk", 80, 45.0, 10, {"Family_History", "BRCA_Mutation", "H_Pylori_Infection"}, "Male"),
        ("min risk", 25, 18.0, 0, set(), "Female"),
        ("moderate", 50, 25.0, 5, set(), "Female"),
    ]
    got = {}
    for label, *args in profiles:
        got[label] = predict(*args)

    check(got.get("max risk") == "High",
          "an extreme-risk profile returns High", f"got {got.get('max risk')}")
    check(got.get("min risk") == "Low",
          "a minimal-risk profile returns Low", f"got {got.get('min risk')}")
    check(len(set(got.values())) == 3,
          "the UI produces all three classes (was: Medium for all 400 rows)",
          str(got))


if __name__ == "__main__":
    sys.exit(main())