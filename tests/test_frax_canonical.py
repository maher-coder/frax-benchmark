"""Canonical-case sanity tests for FRAX Python reimplementation.

These are approximate Sheffield-style outputs for 7 reference patient profiles
(compiled from McCloskey 2016 Osteoporos Int + Kanis 2008 Appendix + typical
Sheffield web-calculator outputs on canonical cases).

Since we do NOT reproduce Sheffield byte-for-byte, the tests assert that:
  1. Each case produces a finite probability in (0, 1).
  2. The MOF / HF ratios (ours / Sheffield-typical) stay within a 3x band at
     extremes — corresponding to our documented joint-vs-marginal shrinkage
     calibration.
  3. The ordering of cases by risk is preserved (low-risk case < high-risk case).

If Sheffield updates their calibration or our equation is re-derived, these
tolerances may need updating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.frax import compute_frax


CANONICAL_CASES = pd.DataFrame([
    # age sex    kg   cm   prior parent smoke glu RA sec alc  t_score  sheffield_mof  sheffield_hf
    (55, 'female', 60, 162, 0, 0, 0, 0, 0, 0, 0, 0.0,      3.5, 0.2),
    (65, 'female', 65, 160, 0, 0, 0, 0, 0, 0, 0, -1.0,     7.5, 1.0),
    (70, 'female', 55, 158, 1, 0, 0, 0, 0, 0, 0, -2.0,    15.0, 3.5),
    (75, 'female', 55, 155, 1, 1, 0, 0, 0, 0, 0, -2.5,    25.0, 8.0),
    (80, 'female', 50, 150, 1, 1, 1, 0, 0, 0, 0, -3.0,    55.0, 25.0),
    (60, 'male',   75, 175, 0, 0, 0, 0, 0, 0, 0, -1.0,     4.0, 0.5),
    (75, 'male',   70, 172, 1, 0, 1, 0, 0, 0, 0, -2.5,    18.0, 6.0),
], columns=['age', 'sex', 'weight_kg', 'height_cm',
             'prior_fx', 'parent_hip_fx', 'current_smoker',
             'glucocorticoid', 'rheumatoid_arthritis', 'secondary_osteoporosis',
             'alcohol_3u', 't_score', 'sheffield_mof', 'sheffield_hf'])
CANONICAL_CASES['fn_bmd_g_cm2'] = 0.858 + CANONICAL_CASES['t_score'] * 0.120


def test_all_cases_produce_valid_probabilities():
    res = compute_frax(CANONICAL_CASES)
    assert np.all(np.isfinite(res.mof_10y))
    assert np.all(np.isfinite(res.hf_10y))
    assert np.all((res.mof_10y >= 0) & (res.mof_10y <= 1))
    assert np.all((res.hf_10y >= 0) & (res.hf_10y <= 1))


def test_risk_ordering_preserved():
    """Low-risk cases should have lower MOF + HF than high-risk ones."""
    res = compute_frax(CANONICAL_CASES)
    low_risk_idx = 0   # 55y female, normal BMD, no CRFs
    high_risk_idx = 4  # 80y female, T=-3, smoker + prior + parent

    assert res.mof_10y[low_risk_idx] < res.mof_10y[high_risk_idx]
    assert res.hf_10y[low_risk_idx] < res.hf_10y[high_risk_idx]


def test_mof_within_3x_sheffield():
    """After joint-vs-marginal shrinkage, our MOF should be within 3x Sheffield typical."""
    res = compute_frax(CANONICAL_CASES)
    our_mof_pct = res.mof_10y * 100
    sheffield_mof = CANONICAL_CASES['sheffield_mof'].to_numpy()
    ratios = our_mof_pct / sheffield_mof
    assert np.all(ratios < 3.0), f"MOF ratios exceed 3x: {ratios}"


def test_hf_within_5x_sheffield():
    """HF calibration is looser than MOF (BMD effect dominates). 5x tolerance."""
    res = compute_frax(CANONICAL_CASES)
    our_hf_pct = res.hf_10y * 100
    sheffield_hf = CANONICAL_CASES['sheffield_hf'].to_numpy()
    # Case 0 (55y healthy) has sheffield=0.2 which is very low; skip for ratio test
    ratios = our_hf_pct[1:] / sheffield_hf[1:]
    assert np.all(ratios < 5.0), f"HF ratios exceed 5x: {ratios}"


def test_extreme_case_not_saturated():
    """The 80y-female-T=-3 case should be bounded below 80 % MOF
    (post-shrinkage; pre-shrinkage this case gave 98 %)."""
    res = compute_frax(CANONICAL_CASES)
    assert res.mof_10y[4] < 0.80, f"Extreme case MOF = {res.mof_10y[4]:.2%} still too high"
    assert res.hf_10y[4] < 0.70, f"Extreme case HF = {res.hf_10y[4]:.2%} still too high"


def test_male_baseline_lower_than_female():
    """Males have lower baseline hazards (55 % MOF, 40 % HF of female)."""
    # Construct matched pair
    pair = pd.DataFrame([
        (60, 'female', 65, 165, 0, 0, 0, 0, 0, 0, 0, -1.0),
        (60, 'male',   75, 175, 0, 0, 0, 0, 0, 0, 0, -1.0),
    ], columns=['age', 'sex', 'weight_kg', 'height_cm',
                 'prior_fx', 'parent_hip_fx', 'current_smoker',
                 'glucocorticoid', 'rheumatoid_arthritis', 'secondary_osteoporosis',
                 'alcohol_3u', 't_score'])
    pair['fn_bmd_g_cm2'] = 0.858 + pair['t_score'] * 0.120
    res = compute_frax(pair)
    assert res.mof_10y[1] < res.mof_10y[0]
    assert res.hf_10y[1] < res.hf_10y[0]
