# Scoreboard

| tag                                   | split   | arm      | context   | sampling   | mode   | n_context   |   rmsle | coverage80   | wall_seconds   |
|:--------------------------------------|:--------|:---------|:----------|:-----------|:-------|:------------|--------:|:-------------|:---------------|
| tabpfn_base_appendix_valid_all_recent | valid   | appendix | all       | recent     | base   | 401125      | 0.21734 | 81.9%        | 1805.4         |
| tabpfn_base_clean_valid_all_recent    | valid   | clean    | all       | recent     | base   | 401125      | 0.21826 | 84.3%        | 1670.3         |
| primary_raw                           | valid   | raw      | all       | recent     | base   | 401125      | 0.22001 | 82.0%        | 1654.2         |
| pilot_raw_50k                         | valid   | raw      | 50000     | recent     | base   | 50000       | 0.2263  | 82.4%        | 148.6          |
| lgbm_engineered                       | valid   | —        | —         | —          | —      | —           | 0.23246 | —            | 125.7          |
| median_benchmark                      | valid   | —        | —         | —          | —      | —           | 0.74414 | —            | —              |
