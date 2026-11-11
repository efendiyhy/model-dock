"""Compatibility module for models pickled with ``ncs_models.Gabungan``.

joblib/pickle only stores an object's *state*; the class itself must be
importable under the same module path when the file is loaded. Models trained
in a notebook that did ``from ncs_models import Gabungan`` therefore fail to
load with ``No module named 'ncs_models'`` unless this module exists.

``Gabungan`` ("combined") is a soft-voting ensemble. Its pickled state is:

* ``anggota``  – member specs: ``(feature_list, estimator_or_None, flag, weight)``
* ``model_``   – fitted members, same tuple shape with a fitted estimator
* ``classes_`` – final class labels
* ``prior_``   – class prior, used as a fallback when there are no members
* ``seed``     – random seed used during training

Prediction = weighted average of each member's ``predict_proba`` (computed on
that member's own feature subset, aligned to ``classes_``), then argmax.

If you have the original ``ncs_models.py`` used for training, you can replace
this file with it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Gabungan:
    """Weighted soft-voting ensemble over per-feature-subset estimators."""

    # ------------------------------------------------------------------ helpers
    def _members(self) -> list[tuple[list[str], Any, Any, float]]:
        members = getattr(self, "model_", None) or []
        out = []
        for m in members:
            feats, est = list(m[0]), m[1]
            weight = float(m[3]) if len(m) > 3 and m[3] is not None else 1.0
            flag = m[2] if len(m) > 2 else None
            if est is not None:
                out.append((feats, est, flag, weight))
        return out

    @property
    def feature_names_in_(self) -> np.ndarray:
        names: list[str] = []
        seen: set[str] = set()
        for feats, *_ in self._members():
            for f in feats:
                if f not in seen:
                    seen.add(f)
                    names.append(f)
        if not names:
            for spec in getattr(self, "anggota", None) or []:
                for f in spec[0]:
                    if f not in seen:
                        seen.add(f)
                        names.append(f)
        return np.asarray(names, dtype=object)

    @property
    def n_features_in_(self) -> int:
        return len(self.feature_names_in_)

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(self.feature_names_in_)
        if arr.shape[1] != len(cols):
            raise ValueError(f"Gabungan membutuhkan {len(cols)} fitur, diberikan {arr.shape[1]}.")
        return pd.DataFrame(arr, columns=cols).apply(pd.to_numeric, errors="coerce")

    # --------------------------------------------------------------- inference
    def predict_proba(self, X: Any) -> np.ndarray:
        frame = self._as_frame(X)
        classes = list(np.asarray(self.classes_))
        index = {c: i for i, c in enumerate(classes)}
        total = np.zeros((len(frame), len(classes)), dtype=float)
        weight_sum = 0.0
        for feats, est, _flag, weight in self._members():
            proba = np.asarray(est.predict_proba(frame[feats]), dtype=float)
            est_classes = list(getattr(est, "classes_", classes))
            aligned = np.zeros_like(total)
            for j, c in enumerate(est_classes):
                if c in index:
                    aligned[:, index[c]] = proba[:, j]
            total += weight * aligned
            weight_sum += weight
        if weight_sum <= 0:
            prior = np.asarray(getattr(self, "prior_", np.full(len(classes), 1 / len(classes))), dtype=float)
            return np.tile(prior, (len(frame), 1))
        total /= weight_sum
        row_sums = total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return total / row_sums

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]
