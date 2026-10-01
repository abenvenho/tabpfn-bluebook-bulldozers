# Pre-registered verdicts

Primary run: `primary_raw`. Margin (H1, H2): 0.005 RMSLE. H5 threshold: 0.25339.

## H1 — corroborated
RMSLE(TabPFN raw) − RMSLE(LightGBM) = -0.01245 (95% CI -0.01480 .. -0.01021); refuted iff diff > 0.005 with CI above 0.

## H2 (clean) — corroborated
RMSLE(raw) − RMSLE(clean) = +0.00175 (95% CI +0.00083 .. +0.00270); refuted iff the clean arm improves by more than 0.005 (CI above 0).
## H2 (appendix) — corroborated
RMSLE(raw) − RMSLE(appendix) = +0.00267 (95% CI +0.00150 .. +0.00388); refuted iff the appendix arm improves by more than 0.005 (CI above 0).

## H3 — not yet testable (needs recent vs random at equal context size)

## H4 — corroborated
80% interval coverage on the validation split: 82.0% (band 75%–85%).

## H5 — corroborated
Primary validation RMSLE 0.22001 vs 0.25339 (top 10% of the 2013 final leaderboard). Pre-registered caveat: leaderboard-mirror split, not a like-for-like rank — Kaggle late submissions are closed.
