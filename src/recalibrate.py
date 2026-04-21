"""FRAX recalibration hierarchy adapted from prevent-benchmark / oncotype-benchmark.

Binary-outcome variant:
    Stage-1: per-stratum logistic refit of `y ~ logit(p_raw)`. Strata = (race × sex).
    Stage-2: per-substratum (T-score-band) logistic refit of `y ~ logit(p_stage1)`.
    Stage-3: per-substratum isotonic smoothing of Stage-2 predictions.

Strata with fewer than `min_per_stratum` observations or events fall back to the
pooled recalibrator to avoid overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Stage1Recalibrator:
    per_stratum: dict[Any, tuple[float, float]] = field(default_factory=dict)
    fallback: tuple[float, float] = (0.0, 1.0)

    def transform(self, p: np.ndarray, strata: pd.Series) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        strata = pd.Series(strata).reset_index(drop=True)
        logit_p = _safe_logit(p)
        out = np.empty_like(p)
        for i, s in enumerate(strata):
            b0, b1 = self.per_stratum.get(s, self.fallback)
            out[i] = _safe_sigmoid(b0 + b1 * logit_p[i])
        return np.clip(out, 1e-6, 1 - 1e-6)


def fit_stage1(y: np.ndarray, p: np.ndarray, strata: pd.Series,
               min_per_stratum: int = 50, min_events: int = 5) -> Stage1Recalibrator:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    strata = pd.Series(strata).reset_index(drop=True)
    logit_p = _safe_logit(p)

    rec = Stage1Recalibrator()
    rec.fallback = _fit_logistic(logit_p, y)

    for s in strata.unique():
        mask = (strata == s).values
        if mask.sum() < min_per_stratum or y[mask].sum() < min_events:
            continue
        params = _fit_logistic(logit_p[mask], y[mask])
        if params is not None:
            rec.per_stratum[s] = params
    return rec


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        m.fit(x.reshape(-1, 1), y.astype(int))
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return (0.0, 1.0)


@dataclass
class Stage2Recalibrator:
    stage1: Stage1Recalibrator
    per_substratum: dict[Any, tuple[float, float]] = field(default_factory=dict)

    def transform(self, p: np.ndarray, strata: pd.Series, substrata: pd.Series) -> np.ndarray:
        p1 = self.stage1.transform(p, strata)
        substrata = pd.Series(substrata).reset_index(drop=True)
        logit_p1 = _safe_logit(p1)
        out = p1.copy()
        for sub, (b0, b1) in self.per_substratum.items():
            mask = (substrata == sub).values
            if mask.sum() == 0:
                continue
            out[mask] = _safe_sigmoid(b0 + b1 * logit_p1[mask])
        return np.clip(out, 1e-6, 1 - 1e-6)


def fit_stage2(stage1: Stage1Recalibrator, y: np.ndarray, p: np.ndarray,
               strata: pd.Series, substrata: pd.Series,
               min_per_substratum: int = 100, min_events: int = 10) -> Stage2Recalibrator:
    y = np.asarray(y, dtype=float)
    p1 = stage1.transform(p, strata)
    logit_p1 = _safe_logit(p1)
    substrata = pd.Series(substrata).reset_index(drop=True)

    rec = Stage2Recalibrator(stage1=stage1)
    for sub in substrata.unique():
        mask = (substrata == sub).values
        if mask.sum() < min_per_substratum or y[mask].sum() < min_events:
            continue
        params = _fit_logistic(logit_p1[mask], y[mask])
        if params is not None:
            rec.per_substratum[sub] = params
    return rec


@dataclass
class Stage3Recalibrator:
    stage2: Stage2Recalibrator
    per_substratum_iso: dict[Any, IsotonicRegression] = field(default_factory=dict)

    def transform(self, p: np.ndarray, strata: pd.Series, substrata: pd.Series) -> np.ndarray:
        p2 = self.stage2.transform(p, strata, substrata)
        substrata = pd.Series(substrata).reset_index(drop=True)
        out = p2.copy()
        for sub, iso in self.per_substratum_iso.items():
            mask = (substrata == sub).values
            if mask.sum() == 0:
                continue
            out[mask] = iso.transform(p2[mask])
        return np.clip(out, 1e-6, 1 - 1e-6)


def fit_stage3(stage2: Stage2Recalibrator, y: np.ndarray, p: np.ndarray,
               strata: pd.Series, substrata: pd.Series,
               min_per_substratum: int = 100, min_events: int = 10) -> Stage3Recalibrator:
    y = np.asarray(y, dtype=float)
    p2 = stage2.transform(p, strata, substrata)
    substrata = pd.Series(substrata).reset_index(drop=True)
    rec = Stage3Recalibrator(stage2=stage2)
    for sub in substrata.unique():
        mask = (substrata == sub).values
        if mask.sum() < min_per_substratum or y[mask].sum() < min_events:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        try:
            iso.fit(p2[mask], y[mask])
            rec.per_substratum_iso[sub] = iso
        except Exception:
            continue
    return rec
