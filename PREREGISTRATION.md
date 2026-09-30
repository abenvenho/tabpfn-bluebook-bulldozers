# Pre-registration — "As is, where is": TabPFN-3.5 on the Blue Book for Bulldozers

Author: Agnaldo Calvi Benvenho. Date: 2026-09-30.

This document is committed **before the first TabPFN run on the real Kaggle data**. After
that commit, hypotheses and thresholds are frozen; any deviation is reported as a deviation
in the results section of the README.

## The bet

TabPFN-3.5, given the auction table **as it came from Kaggle** (YearMade = 1000, zeroed
hour meters, columns ~80% empty, thousands of model codes, free text) and used zero-shot,
prices the machines sold in the following months as well as a LightGBM that received all
the data cleaning and feature engineering of 2013. The design is set against the bet: the
LightGBM gets every advantage; TabPFN gets the raw table and no tuning.

## Data and splits

- Context (training): sales through 2011-12-31 (`Train.csv`).
- Evaluation (out-of-time): sales 2012-01-01 to 2012-04-30 (`Valid.csv`), with prices from
  `ValidSolution.csv`. This split mirrors the competition's public leaderboard.
- `Test.csv` (sales 2012-05-01 to 2012-11-30) has no usable referee: Kaggle no longer
  accepts late submissions for this competition (verified while signed in, 2026-09-30,
  before any run). The submission file is still produced and versioned; if Kaggle reopens
  late submissions before the hackathon deadline, its private score supersedes H5 below.

## Metric

RMSLE as in the competition: sqrt(mean((log1p(pred) − log1p(actual))²)). Predictions are
clipped below at 0.

## Arms

| Arm | Table given to TabPFN |
|---|---|
| `raw` (primary) | all 51 predictor columns exactly as they appear in the CSV; `saledate` as the original string (e.g. `3/14/2012 0:00`); no parsing, no imputation, no cleaning |
| `clean` | 2013-style cleaning: YearMade < 1900 → missing; hour meter = 0 → missing; age at sale; sale date decomposed |
| `appendix` | `raw` + `Machine_Appendix.csv` joined on MachineID ("corrected" year, manufacturer, size class) |

All arms: target = log1p(SalePrice); predictions and quantiles back-transformed with expm1.
TabPFN-3.5 via the Prior Labs API, default settings, zero-shot (no fine-tuning, no
hyperparameter search). LightGBM baseline: engineered features, internal early-stopping
holdout taken from the tail of the training period; no access to 2012 data.

## Primary context size

Chosen from the API cost quote **before any result is seen**: the largest of {all sales,
most recent 200k, 100k, 50k} that fits the token quota. Sensitivity runs (H3) use smaller
contexts.

## Hypotheses and refutation thresholds

| | Hypothesis | Refuted if |
|---|---|---|
| H1 | Raw TabPFN is not worse than engineered LightGBM | worse by > 0.005 RMSLE with 95% CI above 0 |
| H2 | Cleaning the table or joining the "corrected" appendix does not improve TabPFN | improvement > 0.005 RMSLE (same CI rule) |
| H3 | At equal context size, the most recent sales beat a random sample | random ≥ recent (diff ≤ 0) |
| H4 | The 80% predictive interval (q10–q90) holds four months ahead | coverage outside 75–85% on the validation split |
| H5 | The primary raw run scores in the 2013 top 10% on the leaderboard-mirror split | RMSLE on the validation split > 0.25339 |

## Statistical procedure

Paired comparisons (H1, H2, H3): bootstrap of the RMSLE difference over validation sales,
10,000 resamples, seed 42, percentile 95% CI. Coverage (H4): share of validation sales with
actual price inside [q10, q90]. H5: the primary run's RMSLE on the validation split,
against 0.25339 — the 47th of 477 (top 10%) on the 2013 final leaderboard. Caveat stated
in advance: the threshold comes from the final (hidden-test) leaderboard while the score
is computed on the public-leaderboard-mirror split, because Kaggle accepts no late
submission and the 2013 public leaderboard is not retrievable; the comparison is
indicative, not a like-for-like rank. A Kaggle private score, if one is ever obtained
(`results/kaggle_score.txt`), supersedes it.

## Reference points (2013 final leaderboard, hidden test set)

1st 0.22909 · 10th 0.23310 · 47th (top 10%) 0.25339 · 477 teams.

## Deviations

Anything not covered here (API failures, quota shortfalls forcing a smaller primary
context, model identifiers) is decided in favor of the *weakest* claim and logged in the
README results section.
