# Data notice

This project uses the data of the Kaggle competition **Blue Book for Bulldozers**
(https://www.kaggle.com/c/bluebook-for-bulldozers), provided by Fast Iron for the 2013
competition.

**The Kaggle files are not redistributed in this repository.** Downloading them requires a
Kaggle account and acceptance of the competition rules. Place the following files in
`data/raw/` before running the pipeline:

- `Train.csv` — sales through the end of 2011 (~401k rows, 53 columns)
- `Valid.csv` — sales from Jan 1 to Apr 30, 2012 (~11.5k rows, no price)
- `ValidSolution.csv` — the prices of the validation sales (released after the competition)
- `Test.csv` — sales from May 1 to Nov 30, 2012 (no price; Kaggle late submissions are currently closed — see PREREGISTRATION.md)
- `Machine_Appendix.csv` — "corrected" machine attributes (used only by the `appendix` arm)
- `Data Dictionary.xlsx` — optional, for reference

`TrainAndValid.csv` may be used instead of `Train.csv` + `Valid.csv`; the pipeline splits it
by sale date (validation = 2012-01-01 to 2012-04-30).

## What the repository does version

- Prediction files under `results/preds/` contain, for the validation split, the actual
  prices of the ~11.5k validation sales (they come from `ValidSolution.csv`). These prices
  are already public on the competition page.
- Aggregate metrics, the noise audit, figures and verdicts.

## Synthetic data

`scripts/05_make_synthetic.py` generates a synthetic dataset with the same 53-column schema
and the same noise patterns (YearMade = 1000, zeroed hour meters, mostly-empty option
columns, high-cardinality model codes). It is used only by the smoke test
(`scripts/90_smoke.sh`) so the pipeline can be exercised end to end without the Kaggle
files. Synthetic data is regenerated on demand and not versioned.
