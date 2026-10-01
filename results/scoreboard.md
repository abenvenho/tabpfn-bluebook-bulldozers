# Scoreboard

| tag                                    | split   | arm      | context   | sampling   | mode     | n_context   | rmsle   | coverage80   | wall_seconds   |
|:---------------------------------------|:--------|:---------|:----------|:-----------|:---------|:------------|:--------|:-------------|:---------------|
| tabpfn_base_appendix_valid_all_recent  | valid   | appendix | all       | recent     | base     | 401125      | 0.21734 | 81.9%        | 1805.4         |
| tabpfn_base_clean_valid_all_recent     | valid   | clean    | all       | recent     | base     | 401125      | 0.21826 | 84.3%        | 1670.3         |
| primary_raw                            | valid   | raw      | all       | recent     | base     | 401125      | 0.22001 | 82.0%        | 1654.2         |
| tabpfn_fast_raw_valid_all_recent       | valid   | raw      | all       | recent     | fast     | 401125      | 0.22050 | 81.4%        | 812.7          |
| tabpfn_base_raw_valid_200000_random    | valid   | raw      | 200000    | random     | base     | 200000      | 0.22139 | 82.3%        | 628.6          |
| tabpfn_base_raw_valid_200000_recent    | valid   | raw      | 200000    | recent     | base     | 200000      | 0.22373 | 81.8%        | 627.5          |
| tabpfn_base_raw_valid_50000_recent     | valid   | raw      | 50000     | recent     | base     | 50000       | 0.22630 | 82.4%        | 142.3          |
| pilot_raw_50k                          | valid   | raw      | 50000     | recent     | base     | 50000       | 0.22630 | 82.4%        | 148.6          |
| tabpfn_base_raw_valid_100000_recent    | valid   | raw      | 100000    | recent     | base     | 100000      | 0.22712 | 79.7%        | 295.4          |
| tabpfn_base_raw_valid_100000_random    | valid   | raw      | 100000    | random     | base     | 100000      | 0.22744 | 80.2%        | 285.4          |
| tabpfn_thinking_raw_valid_50000_recent | valid   | raw      | 50000     | recent     | thinking | 50000       | 0.22782 | —            | 508.2          |
| lgbm_engineered                        | valid   | —        | —         | —          | —        | —           | 0.23246 | —            | 125.7          |
| tabpfn_base_raw_valid_50000_random     | valid   | raw      | 50000     | random     | base     | 50000       | 0.23268 | 79.7%        | 146.8          |
| median_benchmark                       | valid   | —        | —         | —          | —        | —           | 0.74414 | —            | —              |
| kaggle_primary                         | test    | raw      | all       | recent     | base     | 401125      | —       | —            | 1570.8         |
