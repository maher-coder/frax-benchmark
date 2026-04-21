"""NHANES III (1988-1994) cohort reader.

Parses the fixed-format `adult.dat`, `exam.dat`, and the NDI-2019 Linked
Mortality File using column specifications extracted from the bundled `.sas`
codebooks. Returns a harmonised DataFrame with the FRAX inputs + fracture
endpoints + mortality censoring.

NHANES III files + columns we care about
----------------------------------------

adult.dat (318 cols)
    HSAGEIR   — age at interview (years)
    HSSEX     — sex 1=male 2=female
    BMPWT     — weight (kg)
    BMPHT     — height (cm)
    HAC7      — mother had hip fracture (FRAX input: parent_hip_fracture)
    HAG5A     — self-reported hip fracture (outcome: prevalent_hip_fx)
    HAG5B     — self-reported wrist fracture
    HAG5C     — self-reported spine fracture
    HAF10A..F — glucocorticoid use (FRAX input: glucocorticoids)
    HAR1 / HAR4 — smoking status (FRAX input: current_smoker)
    HAT27      — alcohol consumption frequency (FRAX input: alcohol_3u)
    SDPPSU6    — primary sampling unit (design)
    SDPSTRA6   — stratum (design)
    WTPFEX6    — exam weight

exam.dat (thousands of cols, only DXA needed)
    BDPFNBMD  — femoral-neck BMD (g/cm²)       ← the FRAX input
    BDPTRBMD, BDPINBMD, BDPWTBMD, BDPTOBMD     — alternative BMD sites

NHANES_III_MORT_2019_PUBLIC.dat (NDI linked mortality)
    Fixed-format per the layout file on CDC data-linkage page.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "nhanes3"


# FRAX inputs we extract from NHANES III, with codebook field + our harmonised name.
# FRAX availability in NHANES III — what we have vs not:
#   age, sex, BMI, femoral-neck BMD, prior fx self-report (hip/wrist/spine),
#   parent hip fx (mother only, HAC7), current smoking (HAR1 + HAR3 derivation),
#   alcohol (HAN6HS + HAN6JS heavy-drinking proxy)
# NOT available in NHANES III:
#   glucocorticoid use — not asked in NHANES III 1988-1994
#   rheumatoid arthritis — not directly asked (HAM series has some arthritis but not RA-specific)
#   secondary osteoporosis — composite concept, not directly captured
# We set the missing inputs to 0 (FRAX default when unknown) and document the bias.

ADULT_FIELDS: dict[str, str] = {
    "SEQN": "seqn",
    "HSAGEIR": "age",
    "HSSEX": "sex_raw",
    "HAC7": "parent_hip_fx_raw",
    "HAG5A": "self_hip_fx_raw",
    "HAG5B": "self_wrist_fx_raw",
    "HAG5C": "self_spine_fx_raw",
    "HAR1": "ever_smoked_raw",
    "HAR3": "currently_smokes_raw",
    "HAN6HS": "beer_month",
    "HAN6JS": "liquor_month",
    "SDPSTRA6": "stratum",
    "SDPPSU6": "psu",
    "WTPFEX6": "exam_weight",
    "WTPFQX6": "interview_weight",
}

EXAM_FIELDS: dict[str, str] = {
    "SEQN": "seqn",
    "BMPWT": "weight_kg",
    "BMPHT": "height_cm",
    "DMARACER": "race_raw",
    "DMAETHNR": "ethn_raw",
    "BDPFNBMD": "fn_bmd_g_cm2",
    "BDPTRBMD": "tr_bmd_g_cm2",
    "BDPINBMD": "in_bmd_g_cm2",
    "BDPWTBMD": "wt_bmd_g_cm2",
    "BDPTOBMD": "total_hip_bmd_g_cm2",
}


def _parse_sas_codebook(path: Path) -> dict[str, tuple[int, int]]:
    """Extract variable → (start_col, end_col) map from a NHANES .sas file.

    Looks for the INPUT block with entries like `VAR     12-15` or `VAR    1-5`.
    Returns 1-indexed (start, end) inclusive.
    """
    spec: dict[str, tuple[int, int]] = {}
    text = path.read_text()
    m = re.search(r"INPUT\b(.*?)(;|$)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return spec
    block = m.group(1)
    for line in block.splitlines():
        m2 = re.match(r"\s*([A-Z0-9_]+)\s+(\d+)(?:\s*-\s*(\d+))?", line)
        if m2:
            var, start, end = m2.groups()
            s = int(start)
            e = int(end) if end else s
            spec[var] = (s, e)
    return spec


def _read_fixed(path: Path, spec: dict[str, tuple[int, int]], keep: list[str]) -> pd.DataFrame:
    """Read fixed-format .dat using spec. Only load `keep` columns for speed."""
    keep_spec = {v: spec[v] for v in keep if v in spec}
    if not keep_spec:
        raise RuntimeError(f"No spec entries for {keep}. Available: {sorted(spec)[:20]}")
    colspecs = [(s - 1, e) for (s, e) in keep_spec.values()]
    names = list(keep_spec.keys())
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, header=None)
    return df


def _clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip().replace({"": np.nan, ".": np.nan}), errors="coerce")


def load_nhanes3(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA
    adult_spec = _parse_sas_codebook(data_dir / "adult_codebook.sas")
    exam_spec = _parse_sas_codebook(data_dir / "exam_codebook.sas")

    adult = _read_fixed(data_dir / "adult.dat", adult_spec, list(ADULT_FIELDS))
    adult = adult.rename(columns=ADULT_FIELDS)

    exam = _read_fixed(data_dir / "exam.dat", exam_spec, list(EXAM_FIELDS))
    exam = exam.rename(columns=EXAM_FIELDS)

    df = adult.merge(exam, on="seqn", how="left")

    # Numeric conversions
    for c in ("age", "weight_kg", "height_cm", "fn_bmd_g_cm2", "tr_bmd_g_cm2", "in_bmd_g_cm2",
              "wt_bmd_g_cm2", "total_hip_bmd_g_cm2", "exam_weight", "interview_weight",
              "stratum", "psu", "beer_month", "liquor_month"):
        if c in df.columns:
            df[c] = _clean_numeric(df[c])

    # Apply plausible-range filters (NHANES III uses 888/8888/88888 sentinels for missing).
    PLAUSIBLE = {
        "age": (0, 90),
        "weight_kg": (20, 300),
        "height_cm": (100, 220),
        "fn_bmd_g_cm2": (0.1, 2.0),
        "tr_bmd_g_cm2": (0.1, 2.0),
        "in_bmd_g_cm2": (0.1, 2.0),
        "wt_bmd_g_cm2": (0.1, 2.0),
        "total_hip_bmd_g_cm2": (0.1, 2.0),
        "beer_month": (0, 300),
        "liquor_month": (0, 300),
    }
    for c, (lo, hi) in PLAUSIBLE.items():
        if c in df.columns:
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

    # Harmonised categorical/binary inputs
    df["sex"] = df["sex_raw"].map({"1": "male", "2": "female"})
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    df["parent_hip_fx"] = df["parent_hip_fx_raw"].map({"1": 1, "2": 0}).astype("float64")
    df["self_hip_fx"] = df["self_hip_fx_raw"].map({"1": 1, "2": 0}).astype("float64")
    df["self_wrist_fx"] = df["self_wrist_fx_raw"].map({"1": 1, "2": 0}).astype("float64")
    df["self_spine_fx"] = df["self_spine_fx_raw"].map({"1": 1, "2": 0}).astype("float64")
    df["prior_fx_any"] = ((df["self_hip_fx"] == 1) | (df["self_wrist_fx"] == 1) | (df["self_spine_fx"] == 1)).astype("float64")

    # Glucocorticoid use — not asked in NHANES III. Set to FRAX default 0.
    df["glucocorticoid"] = 0.0

    # Rheumatoid arthritis + secondary osteoporosis — not captured. FRAX default 0.
    df["rheumatoid_arthritis"] = 0.0
    df["secondary_osteoporosis"] = 0.0

    # Current smoker: HAR3 asks "do you smoke now"; HAR1 = "ever smoked 100+".
    # If HAR1==2 (never smoked 100+), participant was not asked HAR3 → set 0.
    ever = df["ever_smoked_raw"].map({"1": 1, "2": 0}).astype("float64")
    now = df["currently_smokes_raw"].map({"1": 1, "2": 0}).astype("float64")
    df["current_smoker"] = np.where(ever == 0, 0.0, now)

    # Alcohol ≥ 3 units/day proxy: (beer+liquor)/month ≥ 90 ⇒ 3/day.
    # Published NHANES III conversion: 1 beer = 1 unit, 1 liquor drink ≈ 1 unit.
    df["alcohol_3u"] = ((df["beer_month"].fillna(0) + df["liquor_month"].fillna(0)) >= 90).astype("float64")

    # Race: NHANES III race-ethn mapping
    df["race"] = df["race_raw"].map({
        "1": "nh_white", "2": "nh_black", "3": "mexican_american",
        "4": "other", "5": "other",
    })

    # Derived T-score (female 20-29 reference — placeholder; will replace with exact NHANES III ref)
    # NHANES III female 20-29 femoral neck: mean=0.858 g/cm², sd=0.120 (McCloskey & Kanis 2011)
    FN_REF_MEAN = 0.858
    FN_REF_SD = 0.120
    df["fn_t_score"] = (df["fn_bmd_g_cm2"] - FN_REF_MEAN) / FN_REF_SD

    df["cohort"] = "NHANES3"
    return df


def load_linked_mortality(data_dir: Path | None = None) -> pd.DataFrame:
    """NDI-2019 Linked Mortality File for NHANES III.

    Layout per CDC NHANES-III public-use data-linkage 2019 documentation:
      SEQN: cols 1-5
      ELIGSTAT: col 15 (1=eligible, 2=under age 18, 3=ineligible)
      MORTSTAT: col 16 (0=alive, 1=deceased)
      UCOD_113: cols 17-19 (underlying cause of death, ICD-10 recodes)
      DIABETES: col 47, HYPERTEN: col 48
      PERMTH_INT: cols 44-46 (person-months from interview)
      PERMTH_EXM: cols 41-43 (person-months from exam)
    """
    data_dir = Path(data_dir) if data_dir else DATA
    path = data_dir / "NHANES_III_MORT_2019_PUBLIC.dat"
    colspecs = [
        (0, 5), (14, 15), (15, 16), (16, 19),
        (40, 43), (43, 46), (46, 47), (47, 48),
    ]
    names = ["seqn", "eligstat", "mortstat", "ucod_113",
             "permth_exm", "permth_int", "diabetes", "hyperten"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, header=None)
    for c in ("mortstat", "permth_exm", "permth_int", "diabetes", "hyperten"):
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    df["seqn"] = df["seqn"].str.strip()
    return df


def load_harmonised(data_dir: Path | None = None) -> pd.DataFrame:
    """Merge NHANES III clinical + DXA + NDI mortality for a single FRAX-ready table."""
    clin = load_nhanes3(data_dir)
    mort = load_linked_mortality(data_dir)
    df = clin.merge(mort, on="seqn", how="left")

    # Build hip-fracture outcome flag from cause-of-death.
    # UCOD_113 codes 089 = "Falls (V-W)" and 090 = "Other external causes" include hip fx;
    # more precisely, ICD-10 S72.* (fracture of femur) is not in UCOD_113's 113-cause recode.
    # We conservatively flag mortality events within a follow-up window as a sensitivity axis,
    # and use self-reported prevalent hip fracture from HAG5A as the primary outcome.
    df["death_event"] = (df["mortstat"] == 1).astype("Int64")
    df["followup_months"] = df["permth_exm"].fillna(df["permth_int"])

    df["cohort"] = "NHANES3"
    return df
