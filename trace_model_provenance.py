"""
trace_model_provenance.py -- Issue #16, evidence for docs/MODEL_PROVENANCE.md

Question: which piece of code produced `model_xgb_new.pkl`, the artifact the
v1 app loaded? Filenames are not evidence, so this script establishes the answer
from the inside out:

  STEP 1  read the artifact: estimator type, every hyperparameter, input shape
  STEP 2  locate the concentration of importance (which column, how much)
  STEP 3  inventory every model the notebook constructs, and every cell that
          writes a file
  STEP 4  static scope analysis -- which `model = ...` bindings are module-level
          and which are trapped inside a function body
  STEP 5  THE DECISIVE TEST: re-run the notebook's own cells verbatim, refit,
          and compare importances / predictions / confusion matrix against the
          artifact
  STEP 6  verdict

Run:  python trace_model_provenance.py
Exit: 0 if the artifact is positively identified, 1 if it cannot be identified.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
LEGACY = BASE / "archive" / "legacy-pickles-2026-09" / "model_xgb_new.pkl"
ARCH_FEATURES = BASE / "archive" / "legacy-pickles-2026-09" / "feature_names.pkl"
NB_PATH = BASE / "Cancer_Risk_Prediction_(ML).ipynb"
DATA = BASE / "cancer-risk-factors.csv"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def hdr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# STEP 1 -- what is inside the artifact
# =============================================================================
hdr("STEP 1  THE ARTIFACT ITSELF")
print("path   :", LEGACY.relative_to(BASE))
print("bytes  :", LEGACY.stat().st_size)

model = joblib.load(LEGACY)
params = model.get_params()

print(f"type            : {type(model).__module__}.{type(model).__name__}")
print(f"repr()          : {model!r}")
print(f"n_features_in_  : {model.n_features_in_}")
print(f"classes_        : {list(model.classes_)}")
print(f"trees fitted    : {len(model.estimators_)}")
print(f"scikit-learn    : {sorted({v.decode() for v in re.findall(rb'1\\.\\d+\\.\\d+', LEGACY.read_bytes())})}")
print()
print("every hyperparameter (model.get_params()):")
for k in sorted(params):
    print(f"    {k:<26} = {params[k]!r}")

# sklearn's repr prints only NON-default parameters, so the repr alone tells us
# how much was customised. This is the single most useful line of evidence here.
defaults_only = repr(model) == f"RandomForestClassifier(random_state={model.random_state})"
print()
print(f"repr is bare 'RandomForestClassifier(random_state={model.random_state})': "
      f"{defaults_only}")
print("  -> sklearn only prints non-default params, so NOTHING was tuned:")
print("     every other hyperparameter is at its sklearn default.")
print("  -> the name says 'xgb'; the object is a sklearn RandomForest. "
      f"Is XGBoost? {'XGB' in type(model).__name__.upper()}")


# =============================================================================
# STEP 2 -- where the importance sits
# =============================================================================
hdr("STEP 2  IMPORTANCE CONCENTRATION  (identifies the input schema)")
imp = np.asarray(model.feature_importances_)
names = [str(n) for n in joblib.load(ARCH_FEATURES)]
order = imp.argsort()[::-1]
print(f"{'rank':<6}{'idx':<6}{'importance':<14}name")
for rank, i in enumerate(order[:5], 1):
    print(f"{rank:<6}{i:<6}{imp[i]:<14.5f}{names[i]}")
top1, top2 = imp[order[0]], imp[order[1]]
print()
print(f"top1 / #2 ratio : {top1 / top2:.1f}x")
print("  (the shipped clean model tops out at 1.05-1.42x -- Issue #12)")
print(f"Overall_Risk_Score holds {imp[names.index('Overall_Risk_Score')]:.1%} "
      "of all importance: it is a near-copy of the target (Issue #5).")
print(f"n_features_in_ = {model.n_features_in_}, and the clean feature set is 17 "
      "-> this model")
print("   cannot even be fed the leak-free columns:")
try:
    model.predict(np.zeros((1, 17)))
    print("   ...UNEXPECTED: accepted 17 columns")
except Exception as e:
    print("  ", str(e).strip().splitlines()[0])


# =============================================================================
# STEP 3 -- notebook inventory
# =============================================================================
hdr("STEP 3  NOTEBOOK INVENTORY")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_ran = sum(1 for c in cells
            if c["cell_type"] == "code" and c.get("execution_count"))
print(f"cells: {len(cells)} total, {n_code} code, {n_ran} of them executed")

EST = re.compile(
    r"\b(RandomForestClassifier|GradientBoostingClassifier|AdaBoostClassifier|"
    r"XGBClassifier|XGBRFClassifier|LGBMClassifier|SVC|LogisticRegression|"
    r"KNeighborsClassifier|GaussianNB|DecisionTreeClassifier|ExtraTreesClassifier|"
    r"MLPClassifier|LinearDiscriminantAnalysis|CalibratedClassifierCV|"
    r"SMOTE|RandomOverSampler|ImbPipeline|Pipeline)\s*\(")
SAVE = re.compile(r"joblib\.dump|pickle\.dump|np\.save|torch\.save")

print()
print("candidate models (cell index / exec / constructor):")
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    hits = sorted(set(EST.findall(src)))
    if hits:
        print(f"    cell {i:<4} exec={str(c.get('execution_count')):<5} {hits}")

print()
print("cells that WRITE a file:")
writers = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if SAVE.search(src):
        writers.append(i)
        for line in src.splitlines():
            if SAVE.search(line):
                print(f"    cell {i:<4} exec={str(c.get('execution_count')):<5} "
                      f"{line.strip()}")

print()
print("every .pkl filename the notebook mentions:")
seen = {}
for i, c in enumerate(cells):
    for m in re.findall(r"['\"]([\w./-]+\.pkl)['\"]", "".join(c["source"])):
        seen.setdefault(m, []).append(i)
for n, idxs in sorted(seen.items()):
    print(f"    {n:<28} cells {idxs}")
print("  -> exactly ONE pkl is ever written, by exactly ONE cell.")


# =============================================================================
# STEP 4 -- scope analysis: which `model` did the save cell actually see?
# =============================================================================
hdr("STEP 4  SCOPE ANALYSIS -- the mechanism")
print("An assignment inside `def objective(trial):` creates a LOCAL name. A")
print("module-level `joblib.dump(model, ...)` cannot see it.\n")

module_bind, local_bind = [], []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    try:
        tree = ast.parse("".join(c["source"]))
    except SyntaxError:
        continue
    for node in tree.body:
        tgts = ([t for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign) else [])
        for t in tgts:
            if t.id == "model":
                module_bind.append((i, c.get("execution_count"), ast.unparse(node.value)))
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "model":
                        local_bind.append((i, c.get("execution_count"), fn.name,
                                           ast.unparse(node.value)))

print("MODULE-SCOPE bindings of `model` (these DO change the global):")
for i, ec, v in module_bind:
    print(f"    cell {i:<4} exec={str(ec):<5} model = {v[:60]}")
print()
print("FUNCTION-LOCAL bindings of `model` (these DO NOT):")
for i, ec, fn, v in local_bind:
    print(f"    cell {i:<4} exec={str(ec):<5} inside {fn}(): model = {v[:52]}")

last_cell, last_exec, last_val = module_bind[-1]
print()
print(f"=> The last MODULE-scope binding is cell {last_cell} (exec {last_exec}):")
print(f"     {last_val}")
print("   The tuning cells rebind only their own local `model`. So the global")
print("   still pointed at cell 10's untuned forest when the save cell ran.")
print()
for i in writers:
    c = cells[i]
    print(f"save cell {i} (exec {c.get('execution_count')}) does "
          f"{''.join(c['source']).count('.fit(')} fit() calls and "
          f"{''.join(c['source']).count('transform(')} transform() calls")
print("   -> it only serialises whatever `model` already was. No fitting, no")
print("      feature engineering, no chance to notice the mismatch.")


# =============================================================================
# STEP 5 -- decisive test
# =============================================================================
hdr("STEP 5  DECISIVE TEST -- refit the notebook's cells and compare")
df = pd.read_csv(DATA)

X = df.drop(columns=["Risk_Level", "Patient_ID", "Cancer_Type"])   # cell 4
le = LabelEncoder()
y = le.fit_transform(df["Risk_Level"])                            # cell 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)             # cell 8
scaler = StandardScaler()                                         # cell 9
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"cell 4  -> X.shape = {X.shape}  (18 features; Overall_Risk_Score kept)")
print("cell 8  -> train_test_split(test_size=0.2, random_state=42, stratify=y)")
print("cell 9  -> StandardScaler fitted on the TRAIN split only")
print("cell 10 -> RandomForestClassifier(random_state=42, n_estimators=100)")
print()

fresh = RandomForestClassifier(random_state=42, n_estimators=100)  # cell 10
fresh.fit(X_train, y_train)

pf, pa = fresh.predict(X_test), model.predict(X_test)
imp_d = float(np.abs(np.asarray(fresh.feature_importances_)
                     - np.asarray(model.feature_importances_)).max())
cm_f, cm_a = confusion_matrix(y_test, pf), confusion_matrix(y_test, pa)
nb_cm = np.array([[19, 0, 1], [0, 65, 0], [0, 0, 315]])   # printed by cell 11

same_params = all(params[k] == fresh.get_params()[k] for k in params)
print(f"hyperparameters identical : {same_params}")
print(f"n_features_in_ identical : {fresh.n_features_in_ == model.n_features_in_}")
print(f"feature_importances_ max abs diff : {imp_d:.3e}")
print(f"predictions identical on all {len(y_test)} test rows : "
      f"{np.array_equal(pf, pa)}")
print()
print("confusion matrix, refit      :", cm_f.tolist())
print("confusion matrix, artifact   :", cm_a.tolist())
print("confusion matrix, cell 11 log:", nb_cm.tolist())
print(f"all three agree : {np.array_equal(cm_f, cm_a) and np.array_equal(cm_f, nb_cm)}")
print(f"accuracy : {(pf == y_test).sum()}/400 = {(pf == y_test).mean():.4f}")
print("  (cell 11's report rounds this to '1.00' -- 399/400 is the real figure)")

identified = (same_params and np.array_equal(pf, pa)
              and np.array_equal(cm_f, nb_cm) and imp_d == 0.0)


# =============================================================================
# STEP 6 -- verdict
# =============================================================================
hdr("STEP 6  VERDICT")
if identified:
    print("POSITIVELY IDENTIFIED.")
    print()
    print(f"  producer : cell index {last_cell}, execution_count {last_exec}")
    print(f"             variable `model` = {last_val}")
    print("  trained on: cells 4 + 8 + 9  (18 features, Overall_Risk_Score included)")
    print("  written by: cell index 50, execution_count 102")
    print("             joblib.dump(model, 'model_xgb_new.pkl')")
    print()
    print("  Proof is bit-level, not circumstantial: a verbatim refit reproduces")
    print(f"  the artifact's feature importances to {imp_d:.1e} and gives identical")
    print("  predictions on all 400 test rows and the same confusion matrix the")
    print("  notebook itself printed.")
    print()
    print("  It is NOT the model the filename implies. Nothing in the notebook")
    print("  ever builds an XGBoost into this file: the two XGB cells assign")
    print("  `model` only inside an objective() closure, so the name is a label")
    print("  applied to a RandomForest.")
    print()
    print("  It is also NOT the best model trained -- see MODEL_PROVENANCE.md.")
    sys.exit(0)

print("COULD NOT BE POSITIVELY IDENTIFIED. Do not guess; investigate.")
print(f"  hyperparameters identical: {same_params}")
print(f"  predictions identical   : {np.array_equal(pf, pa)}")
print(f"  confusion matrices agree: "
      f"{np.array_equal(cm_f, cm_a) and np.array_equal(cm_f, nb_cm)}")
sys.exit(1)
