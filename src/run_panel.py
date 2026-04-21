"""Main panel runner: 4-way LOCO × (raw FRAX | Stage-1 | Stage-2 | Stage-3) with
bootstrap CIs + subgroup audit + Net Benefit at USPSTF thresholds.

Outputs results/cv_log.csv.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.cv import bootstrap_ci, loco_splits
from src.frax import compute_frax
from src.load_all import load_pooled
from src.metric import auc_safe, net_benefit, oe_ratio, sensitivity_at_threshold
from src.recalibrate import fit_stage1, fit_stage2, fit_stage3

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "results" / "cv_log.csv"

USPSTF_HF_THR = 0.03
USPSTF_MOF_THR = 0.20


def _boot_auc(y, p):
    return bootstrap_ci(auc_safe, np.asarray(y), np.asarray(p), n_boot=200)


def _substratum(t_score: pd.Series) -> pd.Series:
    """Osteoporotic / osteopenic / normal T-score buckets (WHO criteria)."""
    bins = [-np.inf, -2.5, -1.0, np.inf]
    return pd.cut(t_score, bins=bins, labels=["osteoporotic", "osteopenic", "normal"]).astype(str)


def run():
    print("Loading pooled cohort ...")
    df = load_pooled(adults_with_dxa=True)
    df = df[df["self_hip_fx"].notna()].reset_index(drop=True)
    print(f"n = {len(df)}  hip fx prev = {int(df['self_hip_fx'].sum())}")

    res = compute_frax(df)
    df["hf_10y"] = res.hf_10y
    df["mof_10y"] = res.mof_10y

    stratum = (df["race"].fillna("unknown") + "_" + df["sex"].fillna("unknown"))
    substratum = _substratum(df["fn_t_score"])

    y = df["self_hip_fx"].values.astype(float)
    p_raw = df["hf_10y"].values.astype(float)

    rows: list[dict] = []
    for split in loco_splits(df["cohort"]):
        tr, te = split.train_idx, split.test_idx
        y_tr, y_te = y[tr], y[te]
        p_tr, p_te = p_raw[tr], p_raw[te]
        s_tr = stratum.iloc[tr].reset_index(drop=True)
        s_te = stratum.iloc[te].reset_index(drop=True)
        sub_tr = substratum.iloc[tr].reset_index(drop=True)
        sub_te = substratum.iloc[te].reset_index(drop=True)

        if y_tr.sum() < 10 or y_te.sum() < 3:
            continue

        stage1 = fit_stage1(y_tr, p_tr, s_tr)
        stage2 = fit_stage2(stage1, y_tr, p_tr, s_tr, sub_tr)
        stage3 = fit_stage3(stage2, y_tr, p_tr, s_tr, sub_tr)

        variants: dict[str, np.ndarray] = {
            "raw_hf": p_te,
            "stage1": stage1.transform(p_te, s_te),
            "stage2": stage2.transform(p_te, s_te, sub_te),
            "stage3": stage3.transform(p_te, s_te, sub_te),
        }

        for name, p_pred in variants.items():
            auc, lo, hi = _boot_auc(y_te, p_pred)
            oe = oe_ratio(y_te, p_pred)
            sens_3 = sensitivity_at_threshold(y_te, p_pred, USPSTF_HF_THR)
            nb_3 = net_benefit(y_te, p_pred, USPSTF_HF_THR)
            rows.append({
                "split": split.name,
                "model": name,
                "n_test": len(te),
                "n_events_test": int(y_te.sum()),
                "auc": round(auc, 4) if np.isfinite(auc) else np.nan,
                "auc_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
                "auc_hi": round(hi, 4) if np.isfinite(hi) else np.nan,
                "oe": round(oe["oe"], 3) if np.isfinite(oe["oe"]) else np.nan,
                "observed_rate": round(oe["observed"], 4) if np.isfinite(oe["observed"]) else np.nan,
                "expected_rate": round(oe["expected"], 4) if np.isfinite(oe["expected"]) else np.nan,
                "sens_at_0.03": round(sens_3, 3) if np.isfinite(sens_3) else np.nan,
                "nb_at_0.03": round(nb_3["nb"], 5) if np.isfinite(nb_3["nb"]) else np.nan,
            })
            print(f"  {split.name:18s}  {name:8s}  AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"O/E={oe['oe']:.2f}  sens@3%={sens_3:.2f}  NB@3%={nb_3['nb']:+.4f}")

        # Per-race audit: train recal on pooled, score on (race × female) subgroup of test
        for race in ["nh_black", "nh_white", "mexican_american"]:
            race_mask = (df["race"].iloc[te].values == race) & (df["sex"].iloc[te].values == "female")
            if race_mask.sum() < 30 or y_te[race_mask].sum() < 3:
                continue
            for name, p_pred in variants.items():
                auc, lo, hi = _boot_auc(y_te[race_mask], p_pred[race_mask])
                rows.append({
                    "split": f"{split.name}:race={race}+female",
                    "model": name,
                    "n_test": int(race_mask.sum()),
                    "n_events_test": int(y_te[race_mask].sum()),
                    "auc": round(auc, 4) if np.isfinite(auc) else np.nan,
                    "auc_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
                    "auc_hi": round(hi, 4) if np.isfinite(hi) else np.nan,
                })

    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LOG, index=False)
    print(f"\nWrote {len(rows)} rows to {LOG}")

    print("\n=== LOCO summary (AUC) ===")
    full = pd.DataFrame(rows)
    pivot = full[~full["split"].str.contains("race=")].pivot_table(index="split", columns="model", values="auc").round(3)
    print(pivot)
    print("\n=== Black-female race audit (AUC) ===")
    pivot2 = full[full["split"].str.contains("race=nh_black")].pivot_table(index="split", columns="model", values="auc").round(3)
    print(pivot2)


if __name__ == "__main__":
    run()
