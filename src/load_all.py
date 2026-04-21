"""Unified loader that pools NHANES III + NHANES continuous D/E/F into a single
FRAX-ready DataFrame.

Schema (harmonised across both source types):
    seqn, cohort, age, sex, race, weight_kg, height_cm, bmi,
    fn_bmd_g_cm2, fn_t_score, self_hip_fx, self_wrist_fx, self_spine_fx,
    prior_fx_any, parent_hip_fx, current_smoker, glucocorticoid,
    rheumatoid_arthritis, secondary_osteoporosis, alcohol_3u
"""
from __future__ import annotations

import pandas as pd

from .cohorts.nhanes3 import load_harmonised as _load_nh3
from .cohorts.nhanes_continuous import load_all_cycles as _load_nhc


SHARED_COLS = [
    "seqn", "cohort", "age", "sex", "race", "weight_kg", "height_cm", "bmi",
    "fn_bmd_g_cm2", "fn_t_score",
    "self_hip_fx", "self_wrist_fx", "self_spine_fx", "prior_fx_any", "parent_hip_fx",
    "current_smoker", "glucocorticoid", "rheumatoid_arthritis",
    "secondary_osteoporosis", "alcohol_3u",
]


def load_pooled(adults_with_dxa: bool = False, age_range: tuple[int, int] = (40, 90)) -> pd.DataFrame:
    nh3 = _load_nh3().reindex(columns=SHARED_COLS).copy()
    nh3["seqn"] = nh3["seqn"].astype(str)
    nhc = _load_nhc()[SHARED_COLS].copy()
    df = pd.concat([nhc, nh3], ignore_index=True)

    if adults_with_dxa:
        age_ok = df["age"].between(*age_range)
        dxa_ok = df["fn_bmd_g_cm2"].notna()
        body_ok = df["weight_kg"].notna() & df["height_cm"].notna()
        df = df[age_ok & dxa_ok & body_ok].reset_index(drop=True)
    return df
