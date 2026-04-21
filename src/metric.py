"""Binary-outcome (prevalent fracture) metric panel.

AUROC + calibration slope + Observed/Expected + Net Benefit at a decision threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 10 or y.sum() < 3 or (len(y) - y.sum()) < 3:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def oe_ratio(y: np.ndarray, p: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"oe": float("nan"), "observed": float("nan"), "expected": float("nan")}
    observed = float(y.mean())
    expected = float(p.mean())
    oe = observed / expected if expected > 0 else float("nan")
    return {"oe": oe, "observed": observed, "expected": expected}


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    N = len(y)
    if N == 0:
        return {"nb": float("nan"), "threshold": threshold, "n_positive": 0}
    positive = p >= threshold
    n_pos = int(positive.sum())
    if n_pos == 0:
        return {"nb": 0.0, "threshold": threshold, "n_positive": 0}
    tp = int(((y == 1) & positive).sum())
    fp = int(((y == 0) & positive).sum())
    w = threshold / (1 - threshold)
    nb = (tp / N) - (fp / N) * w
    return {"nb": float(nb), "threshold": threshold, "n_positive": n_pos}


def sensitivity_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n_pos_true = (y == 1).sum()
    if n_pos_true == 0:
        return float("nan")
    tp = int(((y == 1) & (p >= threshold)).sum())
    return tp / n_pos_true


def specificity_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n_neg_true = (y == 0).sum()
    if n_neg_true == 0:
        return float("nan")
    tn = int(((y == 0) & (p < threshold)).sum())
    return tn / n_neg_true


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of logit(p) against y in a logistic refit."""
    mask = np.isfinite(y) & np.isfinite(p) & (p > 0) & (p < 1)
    y, p = y[mask], p[mask]
    if len(y) < 10 or y.sum() < 3:
        return float("nan")
    from sklearn.linear_model import LogisticRegression

    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        m.fit(logit_p, y)
        return float(m.coef_[0, 0])
    except Exception:
        return float("nan")
