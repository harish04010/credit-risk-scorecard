# Tableau dashboards

The extracts in `results/tableau/` are built for three dashboards. Connect to them as text files; `accounts.csv` is the main table, and the others are small lookups.

## 1. Portfolio overview

- **KPIs across the top:** accounts, exposure (sum of `ead`), mean `pd`, expected loss (sum of `el`).
- **Expected loss by `segment`:** bar chart, sorted descending, labeled with each segment's share of loss.
- **Score distribution:** histogram of `score`, colored by `default`, filtered to `sample` = holdout.
- **Filters:** `segment`, `score_band`, `sample`.

## 2. Scorecard performance

- **Deciles** from `holdout_deciles.csv`: bars for `default_rate`, a line for `predicted`, x-axis `decile`.
- **Scorecard table** from `scorecard.csv`: `feature`, `bin`, `points`, with `default_rate` as a color scale.
- **Fairness view:** from `accounts.csv`, holdout only. Columns are `score_band`, color is `sex`, and the measure is the average of `default`. Add the calculated field `[eligible] = [score] >= cutoff`, with the cutoff from `metrics.json`, to show eligibility share by sex.

## 3. Stress test

- From `stress_test.csv`: `expected_loss` by `scenario`, with `loss_rate` as a label and `story` in the tooltip.
- **A what-if parameter for LGD (0.6 to 0.9):** calculated field `[el_whatif] = [pd] * 0.052 * [LGD parameter] * [ead]` on `accounts.csv`, where 0.052 is the missed-payment-to-charge-off bridge from `results/roll_rates.json`.

Publish to Tableau Public and add the link to the README.
