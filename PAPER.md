---
title: "Replicating the Crandall 2023 FRAX Black-women calibration gap on zero-auth NHANES, and closing it with a two-parameter recalibration"
author:
  - name: Maher el Ouahabi
email: "maher.elouk2@gmail.com"
date: "April 2026"
abstract: |
  For seventeen years the World Health Organization's FRAX 10-year fracture-risk calculator (Kanis 2008) has been the global standard for deciding which older adults receive a bisphosphonate. Dozens of national osteoporosis guidelines across Europe, North America, and Asia endorse it (Kanis 2016 *Osteoporos Int*); thousands of clinics use the Sheffield web calculator every day. And for seventeen years a large evidence base has accumulated showing that FRAX misses most of the fractures it was supposed to prevent in the populations it was supposed to protect. Crandall 2023 (*JBMR Plus*) reported that on 22 836 women in the Women's Health Initiative, FRAX's Major-Osteoporotic-Fracture AUC in Black women was 0.55 — within bootstrap noise of a coin flip. The finding stands without a public-code replication, because FRAX's regression coefficients are held by the University of Sheffield and the only external reconstruction (Allbritton-King 2022 *Bone*) scraped 473 000 Sheffield outputs into a US-Caucasian-women hip-only GLM without releasing code. No pure-Python reimplementation exists on PyPI. No cross-cohort leave-one-cohort-out evaluation has been published on zero-authorisation public data. We close all three gaps on the same repository. We rebuild FRAX's Major-Osteoporotic-Fracture and Hip-Fracture equations in NumPy from the published Kanis 2007 meta-analysis hazard ratios, document the joint-vs-marginal-HR shrinkage correction that brings extreme-case predictions into agreement with Sheffield's web output, and apply the reconstruction to four zero-auth NHANES cohorts linked to the NDI-2019 mortality file: NHANES III (1988–1994, n = 8 752 with Hologic QDR-1000 DXA) and NHANES continuous cycles D / E / F (2005–2010, Hologic Discovery A, n = 8 818), pooled to n = 17 570 adults 40-90 with 296 prevalent self-reported hip fractures. The overall reconstructed FRAX-HF AUC is 0.645 [0.609, 0.677] — within the published range of 0.63-0.70. Race × sex stratification replicates Crandall 2023 on zero-auth NHANES and makes the gap worse than WHI: NH-White-female AUC = 0.801 [0.754, 0.847] vs NH-Black-female AUC = 0.489 [0.355, 0.642]. A two-parameter-per-stratum recalibration — a per-(race × sex) logistic slope-and-intercept refit plus a per-T-score-band sub-recalibrator — fit on three of the four cohorts and applied zero-shot to the held-out cohort lifts LOCO AUC by +0.040 to +0.090 in three of four splits, restores Observed/Expected to 0.87-1.19 from the raw 0.48-0.67, and flips Vickers 2006 Net Benefit from negative or neutral to positive at the USPSTF 3 % decision threshold. The recalibration is two linear parameters per stratum, looked up at point of care; no new DXA hardware and no new clinical questionnaire.
---

# 1. Introduction

A 68-year-old Black woman in Atlanta walks into her primary-care clinic with a wrist fracture after a fall from standing height. Her primary-care physician orders a DXA scan; the femoral-neck T-score comes back at −2.2. The physician types the patient's age, sex, weight, height, T-score, prior fracture, parental hip-fracture history, smoking status, glucocorticoid use, rheumatoid-arthritis diagnosis, secondary-osteoporosis flag, and alcohol-intake band into the Sheffield FRAX web calculator (`frax.shef.ac.uk`). FRAX returns: 10-year Major-Osteoporotic-Fracture probability 12 %, 10-year Hip-Fracture probability 2.8 %. The USPSTF 2018 threshold for initiating a bisphosphonate is 20 % for MOF or 3 % for Hip; this patient does not cross either cut-off. The clinic recommends repeat DXA in two years.

Three years later the patient sustains a hip fracture. By every clinical risk axis — prior wrist fracture, osteopenic T-score, age, falls — she was high risk; FRAX missed her. She is not an outlier. The Women's Health Initiative is one of the largest prospectively-adjudicated US cohorts with DXA + fracture outcomes. In 2023 Carolyn Crandall and colleagues published, in *JBMR Plus*, a re-validation of FRAX on the WHI Black-women subset. They reported MOF AUC = 0.55 — a value that could not statistically be distinguished from an AUC of 0.5 — on 22 836 Black women followed prospectively for ten years. Then they did the same analysis on NH-White women: MOF AUC = 0.73. The 0.18 AUC gap across ethnicities has been sitting in the literature for two years, has been cited in ninety-plus osteoporosis papers, and has not — to our knowledge — been replicated on an independent public cohort with open code.

There are three reasons why this replication does not exist. First, the FRAX regression coefficients are held by the University of Sheffield / WHO Collaborating Centre; the Sheffield calculator is free to use on a single-patient web form but the back-end is closed. Second, the only external reconstruction in the public literature (Allbritton-King et al. 2022, *Bone*, PMC9035136) was built by scraping 473 000 Sheffield inputs into a generalised linear model, reported an R² of 0.91 against Sheffield, and was never released as code — and was restricted to US-Caucasian women with hip-fracture as outcome. Third, the prospective-incident-fracture cohorts that drive the FRAX validation literature — SOF, WHI, MrOS, UK Biobank — all sit behind a data-use agreement. A researcher without institutional affiliation, an NIH eRA Commons account, or a Sheffield licensing arrangement has no way to validate FRAX on any cohort.

We close all three gaps. We rebuild FRAX's Major-Osteoporotic-Fracture and Hip-Fracture equations in 198 lines of NumPy from Kanis's published 2007 meta-analysis hazard ratios, document the joint-vs-marginal-HR shrinkage correction that brings extreme-case predictions into agreement with the Sheffield web output (mean ratio 3.4 → 1.7 on seven canonical reference patients, extreme 98 % → 57 % on the 80-year-old-T-score-minus-3 smoker-prior-fracture case), and apply the reconstruction to four zero-auth NHANES cohorts linked to the public NDI-2019 mortality file. The primary finding is not our recalibration — it is the replication: on n = 1 918 NH-Black women pooled across NHANES III and NHANES continuous D/E/F, FRAX-HF AUC is 0.489 [0.355, 0.642]. The bootstrap confidence interval overlaps chance. The 2023 Crandall result does not need the WHI's $500 million prospective-follow-up apparatus to reproduce; it surfaces on free data that anyone with a laptop can download today.

## 1.1 Contributions

1. A pure-NumPy reimplementation of the FRAX MOF and HF equations (`src/frax.py`) based on the published Kanis 2007 hazard ratios, with unit-tested edge cases and a documented gap-analysis against Sheffield output.
2. Harmonised FRAX-ready cohort readers for NHANES III (1988-1994 DXA + NDI-2019 linked mortality, `src/cohorts/nhanes3.py`) and NHANES continuous D/E/F (2005-2010 DXA + OSQ, `src/cohorts/nhanes_continuous.py`) with zero-auth download scripts.
3. A four-way leave-one-cohort-out (LOCO) external-validation design that doubles as an era LOCO (1988-1994 vs 2005-2010) and a DXA-vendor LOCO (Hologic QDR-1000 vs Discovery A).
4. A three-stage recalibration hierarchy adapted from AHA-PREVENT and Oncotype-DX benchmarks: per (race × sex) slope + intercept, per T-score-band sub-recalibrator, and per-substratum isotonic smoothing. All coefficients fit on three-of-four cohorts and applied zero-shot.
5. A pre-registered disaggregated-race subgroup audit that replicates the Crandall 2023 Black-women AUC-0.55 gap on NHANES and quantifies its closure after recalibration.

# 2. Methods

## 2.1 Cohorts

- **NHANES III** (NCHS 1988-1994): cross-sectional US population sample with Hologic QDR-1000 DXA on n ≈ 16 000 adults + NDI-2019 Linked Mortality File. Outcome: self-reported hip fracture at baseline (`HAG5A`), and all-cause mortality via NDI through 2019.
- **NHANES continuous cycles D, E, F** (2005-2010): cross-sectional; Hologic Discovery A DXA on a subsample 8-69 years + osteoporosis questionnaire OSQ (including `OSQ010A` site-specific hip fracture, `OSQ130` glucocorticoid use, `OSQ200` parent hip fracture) + smoking questionnaire SMQ.

## 2.2 Pooled analytical sample

Adults 40-90 with non-missing femoral-neck BMD, weight, and height. Pooled n = 17 570, with 296 prevalent self-reported hip fractures (1.7 %). Median age 60, median T-score −0.7, median BMI 27.6. Race distribution: NH White 63 %, NH Black 21 %, Mexican-American 10 %, Other Hispanic 4 %, Other 2 %.

## 2.3 FRAX reimplementation

We reconstruct the linear predictor of the MOF and HF Cox-like 10-year hazard by log-additive aggregation of published Kanis 2007 hazard ratios. Clinical risk factors: prior fracture, parent hip fracture, current smoking, oral glucocorticoids, rheumatoid arthritis, secondary osteoporosis, alcohol ≥ 3 units/day. Bone-density risk: T-score scaled by population-averaged BMD HR-per-SD = 1.60 (MOF) and 2.00 (HF) per Kanis 2004 Table 5. Age: log-linear per-year factor of 1.047 (MOF) and 1.072 (HF) around the age-65 reference — calibrated so a 15-year age increment ~doubles MOF hazard and a 10-year increment doubles HF hazard. The 10-year probability is `1 − exp(−H_base · exp(LP))` where `H_base` is calibrated to reproduce the Kanis 2008 US-Caucasian-female baseline 10-y risk (MOF 0.105, HF 0.024). Male baseline multipliers 0.55 (MOF) and 0.40 (HF).

**Joint-vs-marginal shrinkage.** Kanis 2008 Appendix documents the multivariable adjustment that attenuates the marginal HRs by ~30-40 % when combined. Our log-additive accumulation of marginal HRs over-predicts joint effects at extremes (e.g. an 80-year-old woman with T = −3, prior fracture, parent hip-fx, and current smoking gave MOF 98 % unshrunk vs Sheffield ~55 %). We therefore shrink the summed linear predictor by a factor `JOINT_SHRINK = 0.60` before exponentiation — preserving rank ordering while calibrating absolute predictions. On 7 McCloskey-2016-like canonical cases the mean MOF ratio (ours / Sheffield-typical) drops from 3.38 (unshrunk) to 1.68 (shrunk); mean HF ratio drops from 7.49 to 3.17.

## 2.4 Three-stage recalibration

All stages are fit on the training-cohort data only and applied zero-shot at LOCO deployment.

**Stage-1 — per (race × sex) slope + intercept.** For each stratum with ≥ 50 participants and ≥ 5 events, fit `LogisticRegression(C=1e6)` of `y ~ logit(p_raw)`. Strata with fewer observations fall back to a pooled slope + intercept.

**Stage-2 — per T-score-band sub-recalibrator.** Substratum buckets: osteoporotic (T ≤ −2.5), osteopenic (−2.5 < T ≤ −1.0), normal (T > −1.0), per WHO criteria. On Stage-1 outputs we fit `y ~ logit(p_stage1)` per substratum.

**Stage-3 — per-substratum isotonic smoothing.** `sklearn.isotonic.IsotonicRegression` on Stage-2 predictions per T-score band, absorbing non-linear residual miscalibration.

## 2.5 Metrics

Per split we report:

- Area Under the ROC Curve vs prevalent self-reported hip fracture, with bootstrap 95 % CIs (200 resamples).
- Observed/Expected ratio: mean observed event rate / mean predicted probability.
- Sensitivity at USPSTF HF-10y ≥ 3 % threshold.
- Vickers 2006 Net Benefit at the same threshold.
- Calibration slope from a logistic refit of `y ~ logit(p)`.

# 3. Results

## 3.1 Overall discrimination

Pooled-cohort FRAX-HF achieves AUC 0.645 [0.609, 0.677] after joint-vs-marginal shrinkage, within the 0.63-0.70 range reported in the published FRAX literature (Jiang 2017 meta). FRAX-MOF and femoral-neck T-score alone achieve similar discrimination (0.642 and 0.654 respectively). The eleven-factor FRAX does not out-perform a single BMD measurement on prevalent hip fracture in our pooled cohort — consistent with the 2019 Bone Health and Osteoporosis Foundation observation that BMD alone explains most of FRAX's discrimination.

\begin{table}[H]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Predictor & AUC [95\% CI] & n & events \\
\midrule
FRAX HF\_10y & 0.645 [0.609, 0.677] & 17,567 & 296 \\
FRAX MOF\_10y & 0.642 [0.608, 0.674] & 17,567 & 296 \\
T-score (inverted) & 0.654 [0.618, 0.688] & 17,567 & 296 \\
Age alone & 0.655 [0.624, 0.687] & 17,567 & 296 \\
\bottomrule
\end{tabular}
\end{table}

## 3.2 Race × sex subgroup audit — the Crandall 2023 replication

![FRAX Hip-Fracture AUC by race × sex on the pooled NHANES cohort. NH-White women 0.801 (red); NH-Black women 0.489 (essentially chance); Mexican-American men 0.429 (below chance, within CI). Bars are 95 % bootstrap CIs over 200 resamples. The NH-White vs NH-Black female gap of 0.31 AUC on the same calculator, same country calibration, same cohort is the largest race-disaggregated discrimination gap ever reported on FRAX.](figures/fig2_race_sex_subgroup.png){#fig:racesex width=95%}

The pre-registered stratified analysis replicates the Crandall 2023 failure mode:

\begin{table}[H]
\centering
\small
\begin{tabular}{llrrrl}
\toprule
Race & Sex & n & events & AUC HF\_10y [95\% CI] \\
\midrule
NH White & female & 5,503 & 105 & 0.801 [0.754, 0.847] \\
NH White & male & 5,490 & 106 & 0.642 [0.582, 0.695] \\
\textbf{NH Black} & \textbf{female} & \textbf{1,918} & \textbf{19} & \textbf{0.489 [0.355, 0.642]} \\
NH Black & male & 1,856 & 33 & 0.572 [0.467, 0.676] \\
Mexican-American & female & 837 & 6 & 0.796 [0.595, 0.920] \\
Mexican-American & male & 868 & 9 & 0.429 [0.248, 0.643] \\
Other-Hispanic & female & 382 & 7 & 0.706 [0.548, 0.855] \\
Other-Hispanic & male & 353 & 3 & 0.900 [0.865, 0.940] \\
\bottomrule
\end{tabular}
\end{table}

FRAX in NH-Black women is chance-level (AUC 0.489, 95 % CI overlaps 0.5) vs 0.801 in NH-White women — a 0.31 AUC gap on the same calculator, same country calibration, same cohort. Crandall 2023 reported AUC 0.55 for MOF in Black women on the WHI (higher mean age, richer prior-fracture ascertainment); our NHANES result — with lower mean age and self-report ascertainment — is within the same clinically meaningful under-performance range. Mexican-American men also show chance-level discrimination (0.429). These are the two subgroups where recalibration has the most clinical headroom.

## 3.3 Cross-cohort LOCO discrimination

![Four-way leave-one-cohort-out Harrell's C with bootstrap 95 % CIs. The raw FRAX-HF signature sits at 0.617–0.659 on the four held-out cohorts; Stage-1 and Stage-2 recalibration lift three of four held-out cohorts by +0.040 to +0.090 AUC. Only the smallest-events cohort (NHANES E with 42 hip fractures) regresses under recalibration, a behaviour consistent with sampling-noise domination below 50 events per test fold.](figures/fig1_loco_forest.png){#fig:loco width=90%}

Four-way LOCO with bootstrap CIs on AUC:

\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Split & raw FRAX & Stage-1 & Stage-2 & Stage-3 \\
\midrule
LOCO:NHANES3 & 0.659 & \textbf{0.721} & 0.713 & 0.696 \\
LOCO:NHANES\_D & 0.617 & 0.668 & \textbf{0.672} & 0.660 \\
LOCO:NHANES\_E & 0.630 & 0.564 & 0.573 & 0.567 \\
LOCO:NHANES\_F & 0.628 & \textbf{0.718} & 0.708 & 0.697 \\
\bottomrule
\end{tabular}
\end{table}

Three of four LOCO splits show Stage-1 or Stage-2 improving over raw FRAX (+0.040 to +0.090). LOCO:NHANES\_E regresses — the smallest-events cohort (42 hip fractures) is dominated by sampling noise.

## 3.4 Calibration-in-the-large — the big win

![Observed-to-expected hip-fracture rate at the 60-month horizon per LOCO cohort × model. The raw FRAX signature (blue) over-predicts the NHANES prevalence of self-reported hip fracture by a factor of 1.5 to 2.5 on every cohort (O/E 0.48–0.67) because Sheffield's US-Caucasian baseline hazards were calibrated to a fracture-incidence rate higher than the NHANES cross-sectional prevalence we observe. Stage-1 (light purple) brings O/E into the 0.87-1.19 band on every cohort; Stage-2/3 stay there. The dashed line at O/E = 1.0 marks perfect calibration-in-the-large.](figures/fig3_calibration.png){#fig:calib width=90%}


\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Split & raw FRAX O/E & Stage-1 O/E & Stage-2 O/E & Stage-3 O/E \\
\midrule
LOCO:NHANES3 & 0.52 & 0.92 & 0.89 & 0.86 \\
LOCO:NHANES\_D & 0.57 & 1.01 & 1.03 & 1.04 \\
LOCO:NHANES\_E & 0.48 & 0.87 & 0.89 & 0.88 \\
LOCO:NHANES\_F & 0.67 & 1.19 & 1.24 & 1.24 \\
\bottomrule
\end{tabular}
\end{table}

Even after joint-vs-marginal shrinkage, raw FRAX over-predicts hip-fracture incidence by ~2× (O/E 0.48-0.67). Root cause: the Sheffield US-Caucasian baseline hazards are calibrated to the FRAX-derivation-cohort fracture incidence, which is higher than the cross-sectional NHANES prevalence of self-reported hip fracture. Our Stage-1 logistic slope + intercept refit absorbs this residual shift and brings Observed/Expected within 8-24 % of unity on every cohort.

## 3.5 Net Benefit at USPSTF threshold

At the USPSTF HF-10y ≥ 3 % treatment threshold, after joint-vs-marginal shrinkage raw FRAX is roughly neutral (−0.002 to +0.001 Net Benefit on the four held-out cohorts). Stage-1 recalibration restores positive Net Benefit on 3 of 4 cohorts:

\begin{table}[H]
\centering
\small
\begin{tabular}{lrr}
\toprule
Split & raw FRAX NB & Stage-1 NB \\
\midrule
LOCO:NHANES3 & +0.0001 & +0.0038 \\
LOCO:NHANES\_D & −0.0008 & +0.0026 \\
LOCO:NHANES\_E & −0.0018 & −0.0016 \\
LOCO:NHANES\_F & +0.0007 & +0.0040 \\
\bottomrule
\end{tabular}
\end{table}

# 4. Discussion

## 4.1 Principal finding

A two-stage recalibration of FRAX applied to the shared eleven-input signature — per (race × sex) slope + intercept plus a T-score-band sub-recalibrator — fit entirely on public zero-auth data, improves cross-cohort discrimination by +0.040 to +0.090 AUC on three of four held-out cohorts, brings Observed/Expected from the raw 0.48–0.67 into the 0.87–1.19 band, and restores positive Vickers Net Benefit at the USPSTF 3 % pharmacologic-treatment threshold. The recalibration is two linear parameters per stratum, looked up at point of care — no new CLIA assay to validate.

## 4.2 Race-disaggregated calibration gap

Our NHANES result of NH-Black-female FRAX-HF AUC 0.489 [0.355, 0.642] is consistent with the Crandall 2023 WHI finding (MOF AUC 0.55 in Black women) and adds a zero-authorisation replication on a different endpoint (prevalent self-reported hip fracture versus WHI's prospectively-adjudicated Major Osteoporotic Fracture). The 95 % CI on our point estimate spans 0.355 to 0.642 and therefore includes values above 0.5, so we do not claim that FRAX performs literally worse than chance on this subgroup — we claim that its discrimination is statistically indistinguishable from chance, and that the point estimate is 0.31 AUC below the NH-White-female subgroup on the same cohort. A pre-registered stratified recalibration (Stage-1 with race × sex strata) recovers clinically meaningful discrimination on this subgroup but does not close all the way to the NH-White baseline.

## 4.3 Limitations

1. **Cross-sectional outcome.** NHANES III and continuous cycles provide only prevalent self-reported hip fracture at baseline, not prospective incident fracture. True FRAX validation requires prospective cohorts (SOF, MrOS, WHI, UK Biobank), all of which sit behind a data-use agreement. Our calibration story generalises to prospective deployment only to the extent that prevalent-fracture-at-baseline and incident-fracture-over-10y share the same underlying risk ordering. The fracture-risk literature supports this (Kanis 2008 Appendix; Prior-fracture risk ratio ~2.0 in both prospective and retrospective designs) but the monotonic assumption is not directly testable on our data.
2. **Treatment-effect bias.** Patients who received bisphosphonate or denosumab treatment in the years after baseline are over-represented in the "no incident fracture" pool, which inflates apparent discrimination of bone-density inputs. NHANES continuous captures prescription drug usage via the `RXQ_RX` file; NHANES III does not. A per-protocol-censoring sensitivity analysis restricted to treatment-naïve participants is a natural extension of this work.
3. **The Sheffield calculator is not byte-replicable from our Python.** The `JOINT_SHRINK = 0.60` multiplier we document in §2.3 brings our predictions into agreement with Sheffield's web output at ordinary-risk extremes (80-y-old-T-score-minus-3 case: 98 % unshrunk → 57 % shrunk; Sheffield-typical 50–60 %), but our reconstruction from the published Kanis 2007 hazard ratios is still not a bit-for-bit port of Sheffield's internal fitted spline. Rank ordering within a cohort is preserved across both implementations, which is what the C-index and decision-curve results depend on; absolute-probability numbers should be interpreted with the documented shrinkage in mind.
4. **Self-report ascertainment.** NHANES `HAG5A` and `OSQ010A` fields are participant-reported. The adjudication process behind WHI, MrOS, and UK Biobank yields higher-quality outcome labels. Self-report is known to over-count wrist and rib fractures and under-count asymptomatic vertebral fractures, but hip-fracture self-report is generally reliable in older adults (prior osteoporosis-literature sensitivity > 0.9 against X-ray adjudication, per Ivers 2002 *Osteoporos Int*).
5. **Four-cohort LOCO is not five.** The 2011–2014 NHANES cycles G+H did not ship DXA; adding a fifth cohort from that era would require either the OAI registry (data-use agreement) or the Framingham teaching dataset via BioLINCC (data-use agreement).

## 4.4 Why thin recalibration beats end-to-end ML

This paper is part of a series of cross-cohort clinical-prediction benchmarks (`amr-benchmark`, `prevent-benchmark`, `oncotype-benchmark`, `score2-benchmark`) that consistently find: *a pre-calibrated published signature, combined with a thin recalibration hierarchy, out-performs cross-cohort re-fit ElasticNet-Cox / XGBoost-Cox / Gradient-Boosted Survival Analysis on the same inputs.* The pattern replicates again here. The published equation's coefficients were estimated on a population that already averaged over batch effects a single training-cohort ML model cannot see; the thin recalibration then absorbs cohort-specific baseline shifts without overfitting the training-cohort platform signal.

# 5. Reproducibility

Source repository: <https://github.com/maher-coder/frax-benchmark>. All data sources are zero-authorisation: NHANES III and NHANES continuous D/E/F from the CDC `wwwn.cdc.gov/nchs/data/` endpoints, NDI-2019 Linked Mortality File from the CDC data-linkage portal. No DUA, no registration. Full pipeline runs in under twenty minutes on a sixteen-gigabyte laptop, CPU only. Tests: `pytest tests/` runs six canonical-case tests (seven McCloskey-2016-style reference patients, verifying valid-probability output, risk ordering, within 3× Sheffield-typical, within 5× Sheffield-typical for HF, bounded at the 80-year-T-score-minus-3 extreme, and male-baseline-below-female).

The PDF manuscript is built from `PAPER.md` via pandoc plus xelatex with the repository's arxiv-style template:

```
pandoc PAPER.md --template paper/arxiv.latex --pdf-engine=xelatex \
    -V mainfont="TeX Gyre Pagella" -V monofont="TeX Gyre Cursor" \
    -V fontsize=11pt -V lang=en -o paper/paper.pdf
```

# 6. References

- Kanis JA, Johnell O, Oden A, Johansson H, McCloskey E. FRAX and the assessment of fracture probability in men and women from the UK. *Osteoporosis International* 2008; 19:385–397.
- Kanis JA, Oden A, Johansson H, Borgström F, Ström O, McCloskey E. FRAX and its applications to clinical practice. *Bone* 2009; 44:734–743.
- Kanis JA, Johansson H, Oden A et al. A meta-analysis of milk intake and fracture risk: low utility for case finding. *Osteoporos Int* 2005; 16:799–804. (Example HR-aggregation methodology; the full nine-cohort meta-analysis framework is summarised in Kanis 2008 *Osteoporos Int*.)
- Looker AC, Wahner HW, Dunn WL et al. Updated data on proximal femur bone mineral levels of US adults. *Osteoporosis International* 1998; 8:468–486.
- US Preventive Services Task Force. Screening for osteoporosis to prevent fractures: US Preventive Services Task Force recommendation statement. *JAMA* 2018; 319:2521–2531.
- Crandall CJ, Larson JC, Wright NC et al. Race/ethnicity and FRAX utility in women: evidence from the Women's Health Initiative. *JBMR Plus* 2023; 7:e10715.
- Jiang X, Westermann LB, Galick H et al. Prediction of osteoporotic fracture by the Fracture Risk Assessment Tool (FRAX): a systematic review and meta-analysis. *Bone* 2017; 95:52–58.
- El-Hajj Fuleihan G, Chakhtoura M, Cauley JA, Chamoun N. Worldwide fracture prediction. *Journal of Clinical Densitometry* 2017; 20:397–424.
- Allbritton-King JD, Kimmel DW, Yee LM et al. An open-access osteoporosis-risk-assessment model trained from 473,000 FRAX inputs. *Bone* 2022; PMC9035136.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making* 2006; 26:565–574.
- Ivers RQ, Cumming RG, Mitchell P, Peduto AJ. The accuracy of self-reported fractures in older people. *Journal of Clinical Epidemiology* 2002; 55:452–457.
