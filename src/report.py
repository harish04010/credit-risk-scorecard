"""Writes README.md and docs/model_documentation.md from results/, so every number matches the build."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "metrics.json").read_text())
R = json.loads((ROOT / "results" / "roll_rates.json").read_text())
RV = json.loads((ROOT / "results" / "r_validation.json").read_text())
T = pd.read_csv(ROOT / "results" / "scorecard.csv")
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
nt = lambda v: f"NT${v / 1e6:,.1f}M"
P = M["performance"]
ST = {s["scenario"]: s for s in M["stress"]}
F = M["fairness"]
LABEL = {"months_delinquent_6m": "Months delinquent, last 6", "payment_ratio": "Last payment / amount owed",
         "limit": "Credit limit (NT$)", "avg_utilization_6m": "Average utilization, last 6 months",
         "avg_payment_ratio_5m": "Average payment ratio, last 5", "status_now": "Current repayment status",
         "worst_status_6m": "Worst status, last 6", "status_last_month": "Status last month",
         "months_paid_nothing_5m": "Months with no payment, last 5", "utilization": "Utilization now",
         "utilization_trend": "Utilization change over 6 months"}


def scorecard_md():
    out = []
    for f, g in T.groupby("feature", sort=False):
        out.append(f"\n**{LABEL.get(f, f)}**\n\n| Range | Points | Accounts | Default rate |\n|---|---|---|---|")
        out += [f"| {r.bin} | {r.points} | {r.accounts:,} | {pct(r.default_rate)} |" for r in g.itertuples()]
    return "\n".join(out)


def selection_md():
    return "\n".join(f"| {LABEL.get(s['feature'], s['feature'])} | {s['decision']} | {s['why']} |" for s in M["selection"])


dq = "\n".join(f"| {q['check']} | {q['count']:,} of {q['of']:,} | {q['treatment']} |" for q in M["data"]["quality"])
deciles = "\n".join(f"| {d['decile']} | {d['min_score']:.0f} to {d['max_score']:.0f} | {pct(d['default_rate'])} | {pct(d['predicted'])} |"
                    for d in M["holdout_deciles"])
stress = "\n".join(f"| {s['scenario']} | {s['story']} | {pct(s['mean_pd'])} | {pct(s['mean_charge_off_probability'], 2)} | "
                   f"{s['lgd']:.0%} | {nt(s['expected_loss'])} | {pct(s['loss_rate'], 2)} |" for s in M["stress"])
segs = "\n".join(f"| {s['segment']} | {s['share']:.0%} | {s['avg_utilization_6m']:.0%} | {s['avg_payment_ratio_5m']:.0%} | "
                 f"{s['months_delinquent_6m']:.1f} | {pct(s['default_rate'])} | {s['share_of_loss']:.0%} |" for s in M["segments"])
roll = "\n".join(f"| {a} | " + " | ".join(f"{R['matrix_dollar_weighted'][a][b]:.1%}" for b in ["current", "1-2", "3-4", "5+"]) + " |"
                 for a in ["current", "1-2", "3-4", "5+"])
bucket = "\n".join(f"| {b} | {v['accounts']:,} | {nt(v['balance'])} | {pct(v['charge_off_12m'], 2)} | {nt(v['expected_loss_12m'])} |"
                   for b, v in R["by_bucket"].items())
base, adv, sev = ST["Baseline"], ST["Adverse"], ST["Severely adverse"]

readme = f"""# Credit Risk Scorecard

A behavioral credit scorecard for a credit card book, built the way a bank's risk team builds one: data quality checks, monotonic weight-of-evidence binning, a points-based logistic scorecard, validation, an independent re-implementation in R, fairness testing, roll-rate loss forecasting in SQL, a Basel-style stress test, and customer segmentation. The data is 30,000 real accounts.

![Score separation](docs/separation.png)

## Results at a glance

| | |
|---|---|
| Holdout Gini / KS | {P['holdout']['gini']:.3f} / {P['holdout']['ks']:.3f}, stable across development, validation and holdout |
| Calibration | Mean predicted {pct(M['calibration']['mean_pd_holdout'])} against {pct(M['calibration']['default_rate_holdout'])} actual; Hosmer-Lemeshow p = {M['hosmer_lemeshow']['p_value']:.2f} |
| Rank ordering | No statistically significant reversal across score deciles |
| Stability | PSI {M['psi_dev_vs_holdout']:.4f} between development and holdout |
| What transparency costs | A gradient-boosting challenger scores {M['challenger']['auc_given_up']:.3f} more AUC |
| Fairness | Adverse impact ratio {F['adverse_impact_ratio']:.2f} by sex for line-increase eligibility, above the 0.80 four-fifths line |
| Independent validation | R's `glm()` reproduces the coefficients to {RV['max_coefficient_difference']:.0e}, and AUC and KS exactly |
| Baseline 12-month expected loss | {nt(base['expected_loss'])} ({pct(base['loss_rate'], 2)} of exposure), rising to {nt(sev['expected_loss'])} ({pct(sev['loss_rate'], 2)}) under severe stress |

## Data

The [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) data from a Taiwanese bank, April to September 2005 (Yeh & Lien, 2009; UCI, CC BY 4.0). It has 30,000 accounts with credit limits, six months of bills, payments and repayment status, and whether the account missed its October payment. Two independent copies, the original UCI spreadsheet and a version distributed in an R package, were compared cell by cell before anything else, and they match.

## What the data quality checks found

| Check | Found | Treatment |
|---|---|---|
{dq}

Two findings changed the whole analysis.

**Status codes aren't consistent across months.** Code 1 ("one month late") appears 3,688 times in September, 28 times in August, and almost never before that. Earlier months jump from 0 to 2. Roll rates computed on raw codes made bucket 1 look like a trap nobody ever left. Codes 1 and 2 are merged into one "1-2 months past due" bucket in every month.

**"Default" in this data means missing a payment, not a loss.** {pct(M['data']['default_rate'])} of accounts "default" next month. Treating that as the probability of loss, the way many projects with this data do, implies a {pct(M['el_missed_payment_basis'] / ST['Baseline']['ead'], 1)} loss rate, which is absurd for a card book. The roll-rate chain shows that only {pct(M['missed_payment_to_charge_off'])} of customers who miss a payment reach charge-off within 12 months. The expected-loss and stress figures multiply the scorecard's probability by that bridge. After that, the scorecard-based loss ({nt(base['expected_loss'])}) and the independent roll-rate forecast ({nt(R['total_expected_loss_12m'])}) agree to within a factor of {R['total_expected_loss_12m'] / base['expected_loss']:.1f}, instead of 60.

## The scorecard

Sex, marital status and age are left out of the model. The first two are prohibited bases under the Equal Credit Opportunity Act, and age can only be used under narrow conditions. Sex is kept aside for fairness testing.

Eleven behavioral features were built from the statements: delinquency counts, utilization, payment ratios, and limit. Each was binned so its default rate moves in one direction only, then screened:

| Feature | Decision | Why |
|---|---|---|
{selection_md()}

The final model is a logistic regression on weight of evidence, scaled so 600 points means 50:1 odds and every 20 points doubles them. An account's score is the sum of its points:
{scorecard_md()}

## Validation

| Sample | Accounts | Default rate | AUC | Gini | KS |
|---|---|---|---|---|---|
""" + "\n".join(f"| {k.capitalize()} | {v['accounts']:,} | {pct(v['default_rate'])} | {v['auc']:.3f} | {v['gini']:.3f} | {v['ks']:.3f} |" for k, v in P.items()) + f"""

![Deciles](docs/deciles.png)

| Score decile | Score range | Actual default rate | Predicted |
|---|---|---|---|
{deciles}

The middle deciles are crowded: a points scorecard has a limited number of distinct scores, so many accounts tie. The small reversals there are tested with a two-proportion z-test rather than eyeballed, and none is significant (largest z = {max(r['z'] for r in M['reversals']):.2f}).

**Independent validation in R.** `r/validate_scorecard.R` refits the model with base R's `glm()` and recomputes AUC (by the Mann-Whitney formula) and KS from the raw holdout scores, using its own code. It also reports each coefficient's significance, which scikit-learn doesn't. It caught two real problems on the way:

- A feature whose coefficient wasn't distinguishable from zero. The Python pipeline now computes Wald z-statistics and drops such features.
- A KS calculation that split tied scores arbitrarily, overstating KS slightly. It now evaluates at each distinct score.

All five checks now pass: coefficients match to {RV['max_coefficient_difference']:.0e}, and AUC and KS match exactly.

**The challenger.** Gradient boosting on the same features reaches AUC {M['challenger']['holdout_auc']:.3f}, against the scorecard's {P['holdout']['auc']:.3f}. That {M['challenger']['auc_given_up']:.3f} is the price of a model whose every decision can be read off a table, explained to a customer as an adverse-action reason, and reviewed by a regulator. For a decision model, it's usually worth paying. The challenger stays useful as a monitoring benchmark: if the gap widens, something in the data has changed.

## Fairness

The scorecard decides eligibility for a credit line increase: the top 60% of scores qualify. On the holdout set, {pct(F['eligible_share']['Female'])} of women and {pct(F['eligible_share']['Male'])} of men qualify, an adverse impact ratio of {F['adverse_impact_ratio']:.2f} (above the 0.80 four-fifths line). Within each score band, women default {abs(F['mean_gap_within_band']) * 100:.1f} points {'less' if F['mean_gap_within_band'] < 0 else 'more'} than men on average. So the score is, if anything, slightly conservative for women, which is worth monitoring but isn't adverse treatment.

## Loss forecasting with roll rates (SQL)

`sql/roll_rates.sql` reshapes six months of status into long form in DuckDB, buckets it, and counts how balances move between buckets from one month to the next. These are dollar-weighted, because losses are dollars:

| From \\ to | current | 1-2 | 3-4 | 5+ |
|---|---|---|---|---|
{roll}

Treating 5+ as charge-off and stepping the chain forward 12 months gives each bucket's chance of reaching charge-off:

| Bucket today | Accounts | Balance | Reaches charge-off within 12 months | Expected loss (LGD 75%) |
|---|---|---|---|---|
{bucket}

## Stress test

Each account's probability is stressed with the Vasicek single-factor model, the one behind Basel capital rules, using Basel's 4% asset correlation for card portfolios. Loss given default and drawdown of unused limits rise with the scenario:

| Scenario | Story | Mean P(miss) | Mean P(charge-off) | LGD | Expected loss | Loss rate |
|---|---|---|---|---|---|---|
{stress}

![Stress](docs/stress.png)

Losses roughly {'double' if 1.8 < sev['expected_loss'] / base['expected_loss'] < 2.6 else f"multiply by {sev['expected_loss']/base['expected_loss']:.1f}"} in the severe scenario. The biggest lever after PD is the assumed drawdown of unused limits, which the bank should estimate from its own history rather than take from here.

## Segments

K-means on utilization, payment behavior, delinquency and limit, with k chosen by silhouette score (k = {M['segmentation']['k']}):

| Segment | Share of book | Utilization | Payment ratio | Months delinquent | Default rate | Share of expected loss |
|---|---|---|---|---|---|---|
{segs}

One result is worth questioning rather than reporting: customers who pay in full show a {pct(next(s['default_rate'] for s in M['segments'] if s['segment'] == 'Pays in full'))} "default" rate, close to revolvers'. Some of that is the same labeling problem as the inactive accounts. Some is that their large unused limits dominate their exposure. A real bank would estimate drawdown by segment before acting on the loss shares.

## Running it

```bash
pip install -r requirements.txt
python -m src.roll                   # roll rates in SQL; writes the missed-payment to charge-off bridge
python -m src.build                  # scorecard, validation, fairness, stress, segments, Tableau extracts
Rscript r/validate_scorecard.R       # independent validation; exits non-zero if anything disagrees
python -m src.report                 # this README and the model documentation
pytest -q
```

## For Tableau

`results/tableau/` holds tidy extracts: scored accounts, the scorecard, deciles, segments, and stress results. `docs/tableau.md` specifies the dashboards they're built for. I didn't build the workbook itself in this environment. Publishing it to Tableau Public and linking it here is the next step.

## Documentation

`docs/model_documentation.md` is a model development document in the format a model risk team reviews: purpose, data, exclusions, methodology, performance, limitations, and a monitoring plan.
"""
(ROOT / "README.md").write_text(readme)

mdd = f"""# Model development document: behavioral card scorecard

*Generated from `results/` by `src/report.py`. Numbers here match the build.*

## 1. Purpose and use

Estimates each active card account's probability of missing next month's payment, as a points score. Intended uses: credit line increase eligibility, collections prioritization, and as the PD input to expected-loss forecasting, combined with the roll-rate bridge to charge-off. Not intended for: origination decisions on new applicants (no application data), or regulatory capital without recalibration to a charge-off definition.

## 2. Data

- **Source:** Default of Credit Card Clients (UCI, CC BY 4.0), 30,000 accounts, April to September 2005. Two copies reconciled cell by cell.
- **Target:** missed the October 2005 payment ({pct(M['data']['default_rate'])}).
- **Exclusions:** {M['excluded_inactive']['accounts']} accounts with no balance in all six months are indeterminate (their labeled default rate of {pct(M['excluded_inactive']['default_rate'])} is implausible) and are excluded from development. They're scored for portfolio reporting, with this caveat attached.
- **Samples:** stratified development / validation / holdout of {M['data']['dev']:,} / {M['data']['val']:,} / {M['data']['holdout']:,}. The data is a single snapshot, so no out-of-time sample is possible. That's the first limitation below.
- **Prohibited and sensitive attributes:** sex, marital status and age are excluded from the model. Sex is retained for fairness testing only.

## 3. Methodology

- Monotonic WoE binning on the development sample only: fine quantile bins merged until each holds at least 5% of accounts and default rates move in one direction.
- Screening: IV of at least 0.02; pairwise WoE correlation no more than 0.70 (the stronger feature is kept).
- Logistic regression on WoE, unpenalized. Features with a wrong-signed or non-significant (|z| < 1.96) coefficient are removed, and the model is refit.
- Scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds. Points are rounded to integers.
- Final features: {", ".join(LABEL.get(f, f) for f in M['features'])}.

## 4. Performance

| Sample | AUC | Gini | KS |
|---|---|---|---|
""" + "\n".join(f"| {k} | {v['auc']:.3f} | {v['gini']:.3f} | {v['ks']:.3f} |" for k, v in P.items()) + f"""

- Hosmer-Lemeshow (holdout, 10 groups): chi-square {M['hosmer_lemeshow']['chi2']:.1f}, p = {M['hosmer_lemeshow']['p_value']:.3f}.
- Rank ordering: no reversal significant at 95%.
- PSI development to holdout: {M['psi_dev_vs_holdout']:.4f}.
- Benchmark: gradient boosting, AUC {M['challenger']['holdout_auc']:.3f}.

## 5. Independent replication

Base R `glm()` refit; coefficients agree to {RV['max_coefficient_difference']:.1e}. Holdout AUC {RV['auc_r']:.6f} (R) vs {RV['auc_python']:.6f} (Python); KS {RV['ks_r']:.6f} vs {RV['ks_python']:.6f}. Minimum |z| {RV['min_abs_z']:.1f}.

## 6. Fairness

Adverse impact ratio {F['adverse_impact_ratio']:.2f} for line-increase eligibility by sex. Within score bands, observed default rates differ by sex by {F['mean_gap_within_band'] * 100:+.1f} points on average (female minus male).

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
"""
(ROOT / "docs" / "model_documentation.md").write_text(mdd)

tableau = """# Tableau dashboards

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
"""
(ROOT / "docs" / "tableau.md").write_text(tableau)
print("README", len(readme), "| MDD", len(mdd))
