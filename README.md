# As is, where is — TabPFN-3.5 on the Blue Book for Bulldozers

Built with **TabPFN-3.5** for the **Prior Labs TabPFN-3.5 Hackathon** — *Take on a Kaggle
challenge*. Author: Agnaldo Calvi Benvenho, mechanical engineer and machinery & equipment
appraiser (this is the author's second entry; the first, on São Paulo real-estate transfer
prices, is [tabpfn-itbi-sp](https://github.com/abenvenho/tabpfn-itbi-sp)).

## The bet

TabPFN-3.5, given the auction table **as it came from Kaggle** — YearMade recorded as 1000,
zeroed hour meters, columns ~80% empty, thousands of model codes, free-text descriptions —
and used **zero-shot**, prices the machines sold in the following months as well as a
LightGBM that received all the data cleaning and feature engineering of 2013.

"As is, where is" is the auction clause under which this equipment actually trades: no
warranty, no cleanup. The question is whether a tabular foundation model can price under
the same clause — taking the data as is, where it is.

The design is set **against** the bet, and every hypothesis, threshold and analysis
decision is frozen in [PREREGISTRATION.md](PREREGISTRATION.md) *before* the first TabPFN
run on the real data:

| | Hypothesis | Refuted if |
|---|---|---|
| H1 | Raw TabPFN is not worse than engineered LightGBM | worse by > 0.005 RMSLE, 95% CI above 0 |
| H2 | Cleaning the table / joining the "corrected" Machine Appendix does not improve TabPFN | improvement > 0.005 |
| H3 | At equal context size, the most recent sales beat a random sample | random is as good or better |
| H4 | The 80% predictive interval holds four months ahead | coverage outside 75–85% |
| H5 | A late Kaggle submission lands in the top 10% of the 477 teams of 2013 | private score > 0.25339 |

Two referees keep the producer honest: the out-of-time validation split is scored against
`ValidSolution.csv`, and H5 is scored by **Kaggle itself** on the hidden test set
(May–Nov 2012) via late submission — a grade the author cannot touch.

## Why this dataset

The 2013 *Blue Book for Bulldozers* competition (Fast Iron / Kaggle) asked for auction
prices of ~412k pieces of heavy equipment from usage, type and configuration — a machinery
valuation problem, which is the author's professional practice. It is a canonical *dirty*
tabular benchmark: 53 columns full of missing and erroneous values, high-cardinality
codes, and a strict temporal split. The 2013 verdict that the "corrected" Machine Appendix
did **not** help (reported by competitors at the time) becomes a falsifiable hypothesis
here (H2).

## Results

*To be filled from `results/scoreboard.md` and `results/verdicts.md` after the
pre-registered runs on the real Kaggle data. Nothing is quoted here that the repository
cannot regenerate.*

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/90_smoke.sh          # end-to-end check on synthetic data — no API, no token
```

With the Kaggle files in `data/raw/` (see [DATA_NOTICE.md](DATA_NOTICE.md)) and a Prior
Labs token in `TABPFN_TOKEN`:

```bash
.venv/bin/python scripts/00_check_api.py            # API + quantiles + cost quote
.venv/bin/python scripts/01_prepare.py              # splits + noise audit
bash scripts/10_baselines.sh                        # median + engineered LightGBM
bash scripts/20_tabpfn.sh pilot                     # 50k-context pilot
bash scripts/20_tabpfn.sh primary [context]         # H1, H4
bash scripts/20_tabpfn.sh arms                      # H2 (clean, appendix)
bash scripts/20_tabpfn.sh context                   # H3 (recent vs random)
bash scripts/20_tabpfn.sh modes                     # fast, thinking
bash scripts/20_tabpfn.sh kaggle                    # H5 submission file
.venv/bin/python -m src.compare --primary primary_raw
.venv/bin/python -m src.make_figures --primary primary_raw
```

The primary context size is chosen from the cost quote **before any result is seen**
(PREREGISTRATION.md). The Kaggle score is transcribed to `results/kaggle_score.txt`.

## Repository layout

```
PREREGISTRATION.md      frozen hypotheses, thresholds and analysis decisions
DATA_NOTICE.md          what to download from Kaggle; what is (not) redistributed
scripts/                00 api check · 01 prepare · 05 synthetic · 10 baselines ·
                        20 tabpfn blocks · 90 smoke test
src/                    schema · prepare · features · baselines · tabpfn_runner ·
                        metrics · compare · make_figures
data/raw/               Kaggle files (not versioned)
results/                predictions, metrics, scoreboard, verdicts, figures
```

## License and credits

Code under [Apache-2.0](LICENSE). TabPFN and TabPFN-3.5 are developed by
[Prior Labs](https://priorlabs.ai) and used here through the Prior Labs API. Competition
data © Fast Iron / Kaggle, under the competition rules (not redistributed). See
[CITATION.cff](CITATION.cff).
