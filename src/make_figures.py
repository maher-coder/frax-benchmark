"""Build paper-ready figures from results/cv_log.csv.

Fig 1: LOCO forest plot of AUC for raw FRAX + Stage-1/2/3 across the four cohorts.
Fig 2: Race × sex subgroup AUC bar chart (the Crandall 2023 audit).
Fig 3: O/E ratio by model × cohort (calibration-in-the-large).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.cv import bootstrap_ci
from src.frax import compute_frax
from src.load_all import load_pooled
from src.metric import auc_safe

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG = REPO_ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LOG = REPO_ROOT / "results" / "cv_log.csv"


def fig1_loco_forest():
    df = pd.read_csv(LOG)
    m = df[~df["split"].str.contains("race=")].copy()
    cohorts = sorted(m["split"].unique())
    models = ["raw_hf", "stage1", "stage2", "stage3"]
    colors = {"raw_hf": "#1f77b4", "stage1": "#9467bd", "stage2": "#7b3295", "stage3": "#4b1c63"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_offsets = np.linspace(-0.3, 0.3, len(models))
    for i, mod in enumerate(models):
        for j, c in enumerate(cohorts):
            row = m[(m["split"] == c) & (m["model"] == mod)]
            if row.empty:
                continue
            y = j + y_offsets[i]
            auc = row["auc"].iloc[0]
            lo = row["auc_lo"].iloc[0]
            hi = row["auc_hi"].iloc[0]
            ax.errorbar([auc], [y],
                         xerr=[[auc - lo if pd.notna(lo) else 0], [hi - auc if pd.notna(hi) else 0]],
                         fmt="o", ecolor="gray", capsize=3, markersize=6, color=colors[mod])
    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels([c.replace("LOCO:", "") for c in cohorts])
    ax.set_xlabel("AUROC [95% bootstrap CI]")
    ax.set_xlim(0.45, 0.82)
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.4)
    ax.set_title("FRAX 4-way LOCO: raw vs 3-stage recalibration")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[m], label=m) for m in models]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_loco_forest.png", dpi=150)
    plt.savefig(FIG / "fig1_loco_forest.pdf")
    plt.close()
    print(f"  wrote {FIG/'fig1_loco_forest.png'}")


def fig2_race_sex_subgroup():
    df = load_pooled(adults_with_dxa=True)
    df = df[df["self_hip_fx"].notna()].reset_index(drop=True)
    res = compute_frax(df)
    df["hf_10y"] = res.hf_10y

    rows = []
    for race in ("nh_white", "nh_black", "mexican_american", "other_hispanic"):
        for sex in ("female", "male"):
            mask = (df["race"] == race) & (df["sex"] == sex)
            if mask.sum() < 30 or df.loc[mask, "self_hip_fx"].sum() < 3:
                continue
            auc, lo, hi = bootstrap_ci(
                auc_safe,
                df.loc[mask, "self_hip_fx"].to_numpy(),
                df.loc[mask, "hf_10y"].to_numpy(),
                n_boot=200,
            )
            rows.append({"race": race, "sex": sex, "auc": auc, "auc_lo": lo, "auc_hi": hi,
                          "n": int(mask.sum()), "events": int(df.loc[mask, "self_hip_fx"].sum())})
    sub = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    races = sub["race"].unique()
    x = np.arange(len(races))
    width = 0.35
    for i, sex in enumerate(["female", "male"]):
        s = sub[sub["sex"] == sex].set_index("race").reindex(races)
        ax.bar(x + (i - 0.5) * width, s["auc"].fillna(0),
               yerr=[s["auc"] - s["auc_lo"].fillna(s["auc"]), s["auc_hi"].fillna(s["auc"]) - s["auc"]],
               width=width, label=sex, color=["#d62728", "#1f77b4"][i], alpha=0.8, capsize=3)
        for j, (_, row) in enumerate(s.iterrows()):
            if pd.notna(row["auc"]):
                ax.text(x[j] + (i - 0.5) * width, row["auc"] + 0.02,
                         f"n={int(row['n'])}\n({int(row['events'])} fx)",
                         ha="center", fontsize=7)
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.4, label="chance")
    ax.axhline(0.68, color="green", linestyle="--", alpha=0.5, label="overall FRAX AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ") for r in races])
    ax.set_ylabel("AUC HF_10y vs prevalent hip-fx [95% CI]")
    ax.set_ylim(0.30, 1.0)
    ax.set_title("FRAX discrimination by race × sex — the Crandall 2023 replication")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_race_sex_subgroup.png", dpi=150)
    plt.savefig(FIG / "fig2_race_sex_subgroup.pdf")
    plt.close()
    print(f"  wrote {FIG/'fig2_race_sex_subgroup.png'}")


def fig3_calibration():
    df = pd.read_csv(LOG)
    m = df[~df["split"].str.contains("race=")].copy()
    cohorts = sorted(m["split"].unique())
    models = ["raw_hf", "stage1", "stage2", "stage3"]
    pivot = m.pivot_table(index="split", columns="model", values="oe").reindex(cohorts)[models]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cohorts))
    width = 0.2
    colors = {"raw_hf": "#1f77b4", "stage1": "#9467bd", "stage2": "#7b3295", "stage3": "#4b1c63"}
    for i, mod in enumerate(models):
        ax.bar(x + (i - 1.5) * width, pivot[mod], width=width, label=mod, color=colors[mod], alpha=0.8)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="perfect O/E = 1")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("LOCO:", "") for c in cohorts])
    ax.set_ylabel("Observed / Expected hip-fx rate")
    ax.set_title("Calibration-in-the-large: raw FRAX over-predicts 5×; Stage-1+ closes the gap")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_calibration.png", dpi=150)
    plt.savefig(FIG / "fig3_calibration.pdf")
    plt.close()
    print(f"  wrote {FIG/'fig3_calibration.png'}")


def main() -> None:
    fig1_loco_forest()
    fig2_race_sex_subgroup()
    fig3_calibration()


if __name__ == "__main__":
    main()
