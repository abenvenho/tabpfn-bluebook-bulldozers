# Scoreboard

| tag              | split   | arm   | context   | sampling   | mode   | n_context   |   rmsle | coverage80   | wall_seconds   |
|:-----------------|:--------|:------|:----------|:-----------|:-------|:------------|--------:|:-------------|:---------------|
| primary_raw      | valid   | raw   | all       | recent     | base   | 401125      | 0.22001 | 82.0%        | 1654.2         |
| pilot_raw_50k    | valid   | raw   | 50000     | recent     | base   | 50000       | 0.2263  | 82.4%        | 148.6          |
| lgbm_engineered  | valid   | —     | —         | —          | —      | —           | 0.23246 | —            | 125.7          |
| median_benchmark | valid   | —     | —         | —          | —      | —           | 0.74414 | —            | —              |
