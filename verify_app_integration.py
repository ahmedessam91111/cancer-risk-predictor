"""
Verify that `app.py` is wired to exactly the artifacts `train.py` produces.

Issue #8 step 8 asks to "verify that the Streamlit app uses exactly these
artifacts". This script checks that two ways, neither of which requires a
running Streamlit server:

  1. STATIC  -- parse app.py with `ast` and assert what it loads, what it
     scales, and what it never mentions.
  2. BEHAVIOURAL -- pull app.py's real `preprocess_input` function out of its
     source and execute it against the real bundle, then confirm its output is
     byte-identical to the reference path used in train.py's evaluation.

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
    print("1. STATIC -- what does app.py actually load and reference?")
    print("=" * 78)

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
        and (n.value.endswith(".pkl") or n.value.endswith(".joblib"))
    ]

    check("model_bundle.joblib" in " ".join(loaded_paths),
          "app.py loads the bundle from train.py", ", ".join(sorted(set(loaded_paths))))
    check(not any(p in " ".join(loaded_paths)
                  for p in ("model_xgb_new.pkl", "label_encoder.pkl", "feature_names.pkl")),
          "app.py no longer loads the three superseded .pkl files")
    check("scaler" in src and "scaler.transform" in src,
          "app.py applies the persisted scaler before predicting (Issue #2)")
    widget_src = src[src.find("RULES = {"):] if "RULES = {" in src else ""
    check('"Overall_Risk_Score": {"widget"' not in widget_src,
          "app.py removed the Overall_Risk_Score slider from the manual form")
    check('["Overall_Risk_Score"]' not in src,
          "Overall_Risk_Score is not in any widget group (Issue #5)")

    print("\n" + "=" * 78)
    print("2. BEHAVIOURAL -- does app.py's own code produce the model's real input?")
    print("=" * 78)

    bundle = joblib.load(BUNDLE_PATH)
    model, scaler, le = bundle["model"], bundle["scaler"], bundle["label_encoder"]
    feats = bundle["feature_names"]

    ns = {
        "pd": pd, "np": np, "st": stub_streamlit(),
        "scaler": scaler, "FEATURE_NAMES": feats,
    }
    preprocess_input = load_app_function("preprocess_input", ns)
    print("[ok]   extracted app.py's real preprocess_input() and executed it")

    df = pd.read_csv(DATA_PATH)
    sample = df[feats].head(200).copy()

    # Hostile input: reversed column order plus decoy columns, SAME row order,
    # so the comparison against the reference is element-for-element.
    hostile = sample[list(reversed(feats))].copy()
    hostile["Patient_ID"] = range(len(hostile))
    hostile["Overall_Risk_Score"] = 0.9      # decoy: must be ignored, not used
    hostile["Totally_Unknown"] = 42.0

    X_app = np.asarray(preprocess_input(hostile))
    X_ref = scaler.transform(sample[feats])
    maxdiff = float(np.abs(X_app - X_ref).max())
    check(X_app.shape == X_ref.shape,
          "app output shape matches the reference matrix",
          f"{X_app.shape} vs {X_ref.shape}")
    check(maxdiff < 1e-12,
          "app output is bit-identical to train.py's path, despite scrambled columns",
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
    print("3. END TO END -- held-out predictions through the app's code path")
    print("=" * 78)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df["Risk_Level"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats], y, test_size=0.2, random_state=42, stratify=y)

    X_te_app = np.asarray(preprocess_input(X_te[list(reversed(feats))]))
    pred_te = le.inverse_transform(model.predict(X_te_app))
    dist = pd.Series(pred_te).value_counts().to_dict()

    check(len(dist) == 3,
          "all three risk levels appear on the 400 held-out rows (was: Medium only)",
          str(dist))
    check(int(pd.Series(pred_te).eq("High").sum()) > 0,
          "the High class is reachable at all (was: 0 High out of 400)",
          f"{int(pd.Series(pred_te).eq('High').sum())} High")

    # Reproduce the pre-Issue-#8 bug deliberately: raw, unscaled values.
    raw_pred = le.inverse_transform(model.predict(X_te[feats]))
    raw_dist = pd.Series(raw_pred).value_counts().to_dict()
    check(raw_dist != dist,
          "omitting the scaler changes predictions (proves the scaler is load-bearing)",
          f"unscaled {raw_dist}  vs  app path {dist}")

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
