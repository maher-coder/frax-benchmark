"""Cross-validation schemes for the FRAX benchmark.

Primary scheme: LOCO across the four cohort labels (NHANES III + NHANES_D/E/F).
This doubles as an era-LOCO (1988-1994 vs 2005-2010) AND a DXA-vendor-LOCO
(Hologic QDR-1000 vs Discovery).

Secondary: stratified k-fold within each cohort, stratified by self_hip_fx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def loco_splits(cohort: pd.Series) -> Iterator[Split]:
    c = pd.Series(cohort).astype(str)
    for held in sorted(c.unique()):
        test_mask = c.values == held
        yield Split(
            name=f"LOCO:{held}",
            train_idx=np.where(~test_mask)[0],
            test_idx=np.where(test_mask)[0],
        )


def multi_seed_per_cohort_kfold(
    cohort: pd.Series, event: pd.Series, n_splits: int = 5,
    seeds: tuple[int, ...] = (42, 7, 13),
) -> Iterator[Split]:
    cohorts = pd.Series(cohort).astype(str).reset_index(drop=True)
    events = pd.Series(event).fillna(0).astype(int).reset_index(drop=True)
    for seed in seeds:
        for c in sorted(cohorts.unique()):
            idx_c = np.where(cohorts.values == c)[0]
            y_c = events.values[idx_c]
            if y_c.sum() < n_splits or (len(y_c) - y_c.sum()) < n_splits:
                continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_i, (_, te) in enumerate(skf.split(idx_c, y_c)):
                yield Split(
                    name=f"{c}:seed{seed}:fold{fold_i}",
                    train_idx=np.setdiff1d(np.arange(len(cohort)), idx_c[te]),
                    test_idx=idx_c[te],
                )


def bootstrap_ci(metric_fn, *args, n_boot: int = 200, alpha: float = 0.05, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    point = metric_fn(*args)
    n = len(args[0])
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_args = [a[idx] if hasattr(a, "__getitem__") else a for a in args]
        try:
            v = metric_fn(*boot_args)
            if np.isfinite(v):
                estimates.append(v)
        except Exception:
            continue
    if not estimates:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))
