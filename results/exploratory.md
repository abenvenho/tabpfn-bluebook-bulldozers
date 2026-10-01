# Exploratory analyses (not pre-registered)

Nothing in this file changes a pre-registered verdict (see `verdicts.md`). Paired bootstrap as in the pre-registration (10,000 resamples, seed 42, percentile 95% CI); positive difference = the first run is worse.

## TabPFN-3.5 modes

| Comparison | RMSLE A | RMSLE B | A − B | 95% CI |
|---|---|---|---|---|
| fast (all sales) vs base (all sales) | 0.22050 | 0.22001 | +0.00049 | -0.00065 .. +0.00163 |
| thinking (50k recent) vs base (50k recent) | 0.22782 | 0.22630 | +0.00152 | +0.00033 .. +0.00274 |

Thinking mode failed server-side twice at the pre-planned 200k context ("The worker failed to process this request"; request_ids `a724cd8f2ce84f9cabad7e6de3d06c74`, `ca766e3690b6410588dcc297d6439758`) and ran at 50k. It optimises RMSE of log1p(price) with its own internal validation; no `time_col` was passed, because the raw arm keeps the date as the CSV string and the API requires datetimes or numbers there.

## Every TabPFN configuration against the engineered LightGBM

10 of 11 distinct TabPFN configurations on the validation split have a lower RMSLE than the engineered LightGBM (0.23246); the pilot is not counted, as it repeats the 50k-recent configuration. Not lower: `tabpfn_base_raw_valid_50000_random`. Point estimates; the paired test for the primary run is H1.

## Reproducibility of the API

The pilot (2026-09-30) and the 50k-recent run of the context block (2026-10-01) use the same configuration. Over 11,573 validation sales, the largest absolute difference across mean and q10, q25, q50, q75, q90 is 0 USD.

## Measured API spend (2026-10-01)

Readings of the API's monthly usage counter (`results/usage_log.txt`) after each block, against the server quote.

| Interval | Measured tokens | Quote | Note |
|---|---|---|---|
| apos-arms (2026-10-01T08:36:05Z) | 732,938 | 732,938 | counter assumed at 0 on Oct 1 (monthly reset); mid-block reading = first call's quote |
| apos-context (2026-10-01T09:43:40Z) | 326,828 | 326,828 |  |
| apos-fast (2026-10-01T11:30:34Z) | 335,092 | 284,596 | includes the first failed thinking fit (200k) |
| apos-kaggle (2026-10-01T11:56:48Z) | 730,546 | 365,272 | 2x the quote; consistent with an automatic client retry, not confirmed |
| apos-thinking (2026-10-01T12:32:47Z) | 232,296 |  | second failed thinking fit (200k) + thinking 50k |

Total on 2026-10-01: 2,357,700 tokens. Daily limit: 5,000,000 tokens, as stated by the API (`results/api_errors.txt`).

## Server-side timings (seconds)

| Run | Context rows | Eval rows | Train-set transform | Fit | Predict (last call) | Wall (client) |
|---|---|---|---|---|---|---|
| `kaggle_primary` | 401,125 | 12,457 | 534.9 | 0.0 | 503.1 | 1,570.8 |
| `tabpfn_fast_raw_valid_all_recent` | 401,125 | 11,573 | 526.3 | 0.0 | 134.5 | 812.7 |
| `tabpfn_base_raw_valid_200000_random` | 200,000 | 11,573 | 284.2 | 0.0 | 161.9 | 628.6 |
| `tabpfn_base_raw_valid_200000_recent` | 200,000 | 11,573 | 284.6 | 0.0 | 160.4 | 627.5 |
| `tabpfn_base_raw_valid_100000_random` | 100,000 | 11,573 | 148.8 | 0.0 | 58.6 | 285.4 |
| `tabpfn_base_raw_valid_100000_recent` | 100,000 | 11,573 | 145.3 | 0.0 | 63.6 | 295.4 |
| `tabpfn_base_raw_valid_50000_random` | 50,000 | 11,573 | 73.4 | 0.0 | 21.6 | 146.8 |
| `tabpfn_base_raw_valid_50000_recent` | 50,000 | 11,573 | 75.5 | 0.0 | 22.2 | 142.3 |
| `tabpfn_thinking_raw_valid_50000_recent` | 50,000 | 11,573 | 72.9 | 381.9 | 27.9 | 508.2 |

Runs before the 2026-10-01 runner update (pilot, primary, arms) did not log server timings.
