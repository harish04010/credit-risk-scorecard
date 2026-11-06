# Model development document: behavioral card scorecard

*Generated from `results/` by `src/report.py`. Numbers here match the build.*

## 1. Purpose and use

Estimates each active card account's probability of missing next month's payment, as a points score. Intended uses: credit line increase eligibility, collections prioritization, and as the PD input to expected-loss forecasting, combined with the roll-rate bridge to charge-off. Not intended for: origination decisions on new applicants (no application data), or regulatory capital without recalibration to a charge-off definition.

## 2. Data

- **Source:** Default of Credit Card Clients (UCI, CC BY 4.0), 30,000 accounts, April to September 2005. Two copies reconciled cell by cell.
- **Target:** missed the October 2005 payment (22.1%).
- **Exclusions:** 966 accounts with no balance in all six months are indeterminate (their labeled default rate of 35.9% is implausible) and are excluded from development. They're scored for portfolio reporting, with this caveat attached.
- **Samples:** stratified development / validation / holdout of 17,420 / 5,807 / 5,807. The data is a single snapshot, so no out-of-time sample is possible. That's the first limitation below.
- **Prohibited and sensitive attributes:** sex, marital status and age are excluded from the model. Sex is retained for fairness testing only.

## 3. Methodology

- Monotonic WoE binning on the development sample only: fine quantile bins merged until each holds at least 5% of accounts and default rates move in one direction.
- Screening: IV of at least 0.02; pairwise WoE correlation no more than 0.70 (the stronger feature is kept).
- Logistic regression on WoE, unpenalized. Features with a wrong-signed or non-significant (|z| < 1.96) coefficient are removed, and the model is refit.
- Scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds. Points are rounded to integers.
- Final features: Months delinquent, last 6, Last payment / amount owed, Credit limit (NT$), Average utilization, last 6 months.

## 4. Performance

| Sample | AUC | Gini | KS |
|---|---|---|---|
| development | 0.755 | 0.510 | 0.416 |
| validation | 0.754 | 0.509 | 0.426 |
| holdout | 0.751 | 0.502 | 0.413 |

- Hosmer-Lemeshow (holdout, 10 groups): chi-square 12.9, p = 0.115.
- Rank ordering: no reversal significant at 95%.
- PSI development to holdout: 0.0019.
- Benchmark: gradient boosting, AUC 0.784.

## 5. Independent replication

Base R `glm()` refit; coefficients agree to 3.5e-08. Holdout AUC 0.751176 (R) vs 0.751176 (Python); KS 0.413259 vs 0.413259. Minimum |z| 2.0.

## 6. Fairness

Adverse impact ratio 0.91 for line-increase eligibility by sex. Within score bands, observed default rates differ by sex by -1.2 points on average (female minus male).

## 7. Limitations

1. **No out-of-time validation.** The data covers six months from 2005, during Taiwan's card debt crisis. Default rates then were far above normal, and stability over time can't be tested.
2. **Definition of default.** The target is a missed payment, not a charge-off. Losses rely on the roll-rate bridge, which is estimated from the same six months.
3. **Inconsistent status coding** across months (section 2). Codes 1 and 2 are merged, which loses the distinction between one and two months past due before September.
4. **Loss assumptions.** LGD (75% to 85%) and credit conversion factors (50% to 70%) are industry-typical assumptions, not estimates from this bank's recoveries.
5. **Indeterminate accounts** are excluded, not resolved. The labeling question needs the data owner.

## 8. Monitoring plan

| Metric | Frequency | Warning | Action |
|---|---|---|---|
| Score PSI vs development | Monthly | 0.10 | Investigate; above 0.25, review for redevelopment |
| Characteristic PSI per feature | Monthly | 0.10 | Find the driver |
| KS on matured accounts | Quarterly | Drop of 10% relative | Recalibrate or redevelop |
| Actual vs predicted by band | Quarterly | Outside the 95% binomial interval | Recalibrate intercept |
| Challenger AUC gap | Quarterly | Above 0.05 | Review for missed structure |
| Adverse impact ratio | Quarterly | Below 0.80 | Fair lending review |
