"""NHANES continuous (2005-2010) cohort reader: D (2005-06) + E (2007-08) + F (2009-10).

DXA was collected only on a subsample 8-69 years old across these cycles. Each cycle
ships:
  - DEMO_{D,E,F}.xpt  — demographics (age, sex, race, survey weights, PSU, stratum)
  - BMX_{D,E,F}.xpt   — body measures (height, weight, BMI)
  - DXXFEM_{D,E,F}.xpt — femoral DXA (DXXOFBMD = total femur, DXXNKBMD = femur neck)
  - OSQ_{D,E,F}.xpt   — osteoporosis questionnaire (OSQ060 prior fracture, OSQ130
    glucocorticoid use ≥ 3 months, OSQ200 parent hip fracture, OSQ010A/B/C site-
    specific fractures)

We do NOT have 10-year incident fracture follow-up on continuous NHANES (would
require restricted CMS linkage). Our outcome is prevalent self-reported fracture
plus NDI-2019 all-cause mortality when available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "nhanes_continuous"

CYCLES = {
    "D": ("2005-2006", "2005_2006"),
    "E": ("2007-2008", "2007_2008"),
    "F": ("2009-2010", "2009_2010"),
}


def _read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    # Force column-name strings and strip whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_cycle(suffix: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one NHANES cycle ('D', 'E', or 'F') joined demographics + BMX + DXA + OSQ + SMQ."""
    data_dir = Path(data_dir) if data_dir else DATA
    demo = _read_xpt(data_dir / f"DEMO_{suffix}.xpt")
    bmx = _read_xpt(data_dir / f"BMX_{suffix}.xpt")
    dxa = _read_xpt(data_dir / f"DXXFEM_{suffix}.xpt")
    osq = _read_xpt(data_dir / f"OSQ_{suffix}.xpt")
    smq_path = data_dir / f"SMQ_{suffix}.xpt"
    smq = _read_xpt(smq_path) if smq_path.exists() else pd.DataFrame({"SEQN": []})

    df = demo.merge(bmx, on="SEQN", how="left")
    df = df.merge(dxa, on="SEQN", how="left")
    df = df.merge(osq, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")
    df["cycle"] = suffix
    return df


def _binary_yn(s: pd.Series) -> pd.Series:
    """Map NHANES yes/no codes: 1=yes, 2=no, 7/9=don't know/refused → NaN."""
    return s.map({1.0: 1.0, 2.0: 0.0, 1: 1.0, 2: 0.0}).astype("float64")


def harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a FRAX-ready frame with the same schema as nhanes3.load_harmonised()."""
    out = pd.DataFrame(index=df.index)
    out["seqn"] = df["SEQN"].astype("Int64").astype(str)
    out["cohort"] = "NHANES_" + df["cycle"].astype(str)

    # Demographics
    out["age"] = pd.to_numeric(df.get("RIDAGEYR"), errors="coerce")
    out["sex"] = df.get("RIAGENDR").map({1.0: "male", 2.0: "female", 1: "male", 2: "female"})
    race_col = df.get("RIDRETH1", pd.Series(index=df.index))
    out["race"] = race_col.map({
        1.0: "mexican_american", 2.0: "other_hispanic",
        3.0: "nh_white", 4.0: "nh_black", 5.0: "other",
        1: "mexican_american", 2: "other_hispanic",
        3: "nh_white", 4: "nh_black", 5: "other",
    })

    # Body measures
    out["weight_kg"] = pd.to_numeric(df.get("BMXWT"), errors="coerce")
    out["height_cm"] = pd.to_numeric(df.get("BMXHT"), errors="coerce")
    out["bmi"] = pd.to_numeric(df.get("BMXBMI"), errors="coerce")

    # DXA: femur-neck BMD in DXXNKBMD (g/cm²)
    out["fn_bmd_g_cm2"] = pd.to_numeric(df.get("DXXNKBMD"), errors="coerce")
    out["total_hip_bmd_g_cm2"] = pd.to_numeric(df.get("DXXOFBMD"), errors="coerce")

    # FRAX clinical inputs from OSQ
    out["self_hip_fx"] = _binary_yn(df.get("OSQ010A", pd.Series(index=df.index, dtype=float)))
    out["self_wrist_fx"] = _binary_yn(df.get("OSQ010B", pd.Series(index=df.index, dtype=float)))
    out["self_spine_fx"] = _binary_yn(df.get("OSQ010C", pd.Series(index=df.index, dtype=float)))
    prior_any = df.get("OSQ060")  # ever broken/fractured any bone (alternate prior-fx indicator)
    out["prior_fx_any"] = _binary_yn(prior_any) if prior_any is not None else (
        ((out["self_hip_fx"] == 1) | (out["self_wrist_fx"] == 1) | (out["self_spine_fx"] == 1)).astype("float64")
    )
    out["parent_hip_fx"] = _binary_yn(df.get("OSQ200", pd.Series(index=df.index, dtype=float)))
    out["glucocorticoid"] = _binary_yn(df.get("OSQ130", pd.Series(index=df.index, dtype=float)))

    # Smoking & alcohol — take from demographics / other files
    # SMQ040: do you now smoke cigarettes? 1=every day, 2=some days, 3=not at all.
    if "SMQ040" in df.columns:
        sm = pd.to_numeric(df["SMQ040"], errors="coerce")
        out["current_smoker"] = sm.isin([1.0, 2.0]).astype("float64")
    else:
        out["current_smoker"] = np.nan
    # ALQ120Q frequency of drinks per week — not always available; default 0
    out["alcohol_3u"] = 0.0

    out["rheumatoid_arthritis"] = 0.0  # Not routinely captured
    out["secondary_osteoporosis"] = 0.0

    # T-score vs NHANES III ref (Looker 1998)
    FN_REF_MEAN = 0.858
    FN_REF_SD = 0.120
    out["fn_t_score"] = (out["fn_bmd_g_cm2"] - FN_REF_MEAN) / FN_REF_SD

    # Survey weights + design (for design-aware analyses)
    out["stratum"] = pd.to_numeric(df.get("SDMVSTRA"), errors="coerce")
    out["psu"] = pd.to_numeric(df.get("SDMVPSU"), errors="coerce")
    out["exam_weight"] = pd.to_numeric(df.get("WTMEC2YR"), errors="coerce")

    return out


def load_all_cycles(data_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    for suffix in CYCLES:
        raw = load_cycle(suffix, data_dir)
        frames.append(harmonise(raw))
    out = pd.concat(frames, ignore_index=True)
    return out
