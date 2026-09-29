"""
CalibratedModel -- Issue #11 "hybrid" classifier wrapper.

Own small module (not export_production.py) so joblib stores the class by an
importable module reference. If it lived in a `python xxx.py`-style script its
module would be `__main__` and every later joblib.load() would fail.

Behaviour (user-approved decision A for Issue #11):

  * predict()      -> raw forest argmax. Class labels are byte-for-byte what
                      train.py produced, so EVERY published metric (accuracy,
                      High recall/precision, macro-F1, ...) stays exactly valid.
  * predict_proba()-> per-class isotonic recalibration (fitted out-of-fold on
                      the training split by export_production._fit_calibrators),
                      renormalised so each row sums to 1.

The calibrators are fitted elsewhere; this class only composes and applies.
It subclasses sklearn's BaseEstimator and is "fitted by construction" so
Pipeline's check_is_fitted() accepts it.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class CalibratedModel(BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, forest, calibrators):
        self.forest = forest
        self.calibrators = list(calibrators)

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):            # noqa: N803 - sklearn signature
        # Already fitted: the wrapper composes a fitted forest + fitted
        # calibrators. Present so sklearn's check_is_fitted accepts it.
        return self

    # ---- sklearn-facing surface -------------------------------------------------
    @property
    def classes_(self):
        return self.forest.classes_

    @property
    def estimators_(self):
        return self.forest.estimators_

    @property
    def feature_importances_(self):
        return self.forest.feature_importances_

    @property
    def n_features_in_(self):
        return self.forest.n_features_in_

    @property
    def feature_names_in_(self):
        return self.forest.feature_names_in_

    # ---- prediction --------------------------------------------------------------
    def predict(self, X):
        # Raw forest argmax -- Issue #10 metrics must not move.
        return self.forest.predict(X)

    def predict_proba(self, X):
        raw = self.forest.predict_proba(X)
        order = {int(c): i for i, c in enumerate(self.forest.classes_)}
        cols = [raw[:, order[c]] for c in range(len(self.classes_))]
        out = np.column_stack(
            [cal.predict(cols[c]) for c, cal in enumerate(self.calibrators)])
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0                      # guard against all-zero rows
        return out / s

    def __repr__(self):  # pragma: no cover - readability only
        return f"CalibratedModel(forest={type(self.forest).__name__}, n_calibrators={len(self.calibrators)})"