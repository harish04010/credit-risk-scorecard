"""
Builds the scorecard and everything downstream of it.

    python -m src.build
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import portfolio
from .data import features, inactive, load_raw, protected, quality_report
from .scorecard import (BASE_ODDS, BASE_SCORE, PDO, deciles, fit, gini, hosmer_lemeshow, ks, pd_from_score,
                        points_table, psi, score, select_features, significant_reversals)
from .woe import fit_all, transform

ROOT = Path(__file__).resolve().parent.parent
RESULTS, DOCS, TABLEAU = ROOT / "results", ROOT / "docs", ROOT / "results" / "tableau"
SEED = 2005


def main():
    raw = load_raw()
    X, y, prot = features(raw), raw["default"], protected(raw)
    # Inactive accounts are indeterminate: kept in the book, left out of development and validation.
    excluded = inactive(raw).to_numpy()
    idx = np.arange(len(raw))
    modelable = idx[~excluded]
    dev, rest = train_test_split(modelable, test_size=0.4, stratify=y.iloc[modelable], random_state=SEED)
    val, hold = train_test_split(rest, test_size=0.5, stratify=y.iloc[rest], random_state=SEED)

    # Bins are learned on the development sample only.
    binnings = fit_all(X.iloc[dev], y.iloc[dev])
    woe_dev = transform(X.iloc[dev], binnings)
    candidates, selection_log = select_features(binnings, woe_dev)
    model, feats, sign_log = fit(woe_dev, y.iloc[dev], candidates)
    selection_log += sign_log
    table = points_table(model, feats, binnings)

    s_all = score(X, table, binnings)
    pd_all = pd_from_score(s_all)
    s_dev, s_val, s_hold = s_all[dev], s_all[val], s_all[hold]

    def perf(ix, s):
        return {"accounts": int(len(ix)), "default_rate": float(y.iloc[ix].mean()),
                "auc": float(roc_auc_score(y.iloc[ix], -s)), "gini": float(gini(y.iloc[ix], -s)), "ks": ks(y.iloc[ix], -s)}

    perf_dev, perf_val, perf_hold = perf(dev, s_dev), perf(val, s_val), perf(hold, s_hold)
    bands = np.quantile(s_dev, np.linspace(0, 1, 11))
    bands[0], bands[-1] = -np.inf, np.inf
    hl = hosmer_lemeshow(y.iloc[hold].to_numpy(), pd_all[hold])
    dec = deciles(y.iloc[hold].to_numpy(), s_hold, pd_all[hold])
    reversals = significant_reversals(dec)
    rank_ordered = not any(r["significant"] for r in reversals)

    # Challenger: gradient boosting on the same raw features. What does transparency cost?
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED)
    gbm.fit(X.iloc[dev], y.iloc[dev])
    gbm_auc = float(roc_auc_score(y.iloc[hold], gbm.predict_proba(X.iloc[hold])[:, 1]))

    # Fairness: line-increase eligibility for the top 60% of scores, by sex.
    cut = float(np.quantile(s_dev, 0.4))
    hold_df = pd.DataFrame({"score": s_hold, "default": y.iloc[hold].to_numpy(), "sex": prot["sex"].iloc[hold].to_numpy(),
                            "band": pd.cut(s_hold, bands, labels=False)})
    approval = hold_df.assign(eligible=hold_df["score"] >= cut).groupby("sex")["eligible"].mean()
    by_band = hold_df.groupby(["band", "sex"])["default"].mean().unstack()
    fairness = {
        "decision": f"eligible for a credit line increase if score >= {cut:.0f} (top 60% of the development sample)",
        "eligible_share": {k: float(v) for k, v in approval.items()},
        "adverse_impact_ratio": float(approval.min() / approval.max()),
        "lower_group": approval.idxmin(),
        "default_rate_by_band_and_sex": {str(int(b)): {k: float(v) for k, v in row.items()} for b, row in by_band.iterrows()},
        "mean_gap_within_band": float((by_band["Female"] - by_band["Male"]).mean()),
    }

    # Portfolio: score the whole book, then expected loss, stress and segments.
    bal, lim = raw["BILL_AMT1"].to_numpy(float), raw["LIMIT_BAL"].to_numpy(float)
    roll = json.loads((RESULTS / "roll_rates.json").read_text())
    conv = roll["missed_payment_to_charge_off"]
    stress = portfolio.stress_test(pd_all, bal, lim, conv)
    seg_labels, seg_info = portfolio.segment(X)
    book = X.assign(segment_id=seg_labels, pd=pd_all, default=y, score=s_all,
                    ead=portfolio.ead(bal, lim, 0.5), balance=np.clip(bal, 0, None))
    book["el"] = book["pd"] * conv * 0.75 * book["ead"]
    segments = []
    for sid, g in book.groupby("segment_id"):
        prof = g[portfolio.SEGMENT_FEATURES].mean().to_dict()
        segments.append({"segment": portfolio.name_segment(prof), "accounts": len(g), "share": len(g) / len(book),
                         **{k: float(v) for k, v in prof.items()}, "default_rate": float(g["default"].mean()),
                         "mean_score": float(g["score"].mean()), "expected_loss": float(g["el"].sum()),
                         "share_of_loss": float(g["el"].sum() / book["el"].sum())})
    names = {s["segment"] for s in segments}
    if len(names) < len(segments):  # two clusters with the same profile name: number them
        seen = {}
        for s in segments:
            seen[s["segment"]] = seen.get(s["segment"], 0) + 1
            if sum(t["segment"] == s["segment"] for t in segments) > 1:
                s["segment"] += f" {seen[s['segment']]}"
    seg_name = {sid: s["segment"] for sid, s in zip(sorted(book["segment_id"].unique()), segments)}
    book["segment"] = book["segment_id"].map(seg_name)

    metrics = {
        "data": {"accounts": len(raw), "default_rate": float(y.mean()), "dev": len(dev), "val": len(val), "holdout": len(hold),
                 "quality": quality_report(raw)},
        "scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        "binning": {f: {"iv": b.iv, "bins": b.stats} for f, b in binnings.items()},
        "selection": selection_log,
        "features": feats,
        "coefficients": dict(zip(feats, map(float, model.coef_[0]))), "intercept": float(model.intercept_[0]),
        "performance": {"development": perf_dev, "validation": perf_val, "holdout": perf_hold},
        "holdout_deciles": dec, "rank_ordered": rank_ordered, "reversals": reversals,
        "excluded_inactive": {"accounts": int(excluded.sum()), "default_rate": float(y[excluded].mean())},
        "hosmer_lemeshow": hl,
        "calibration": {"mean_pd_holdout": float(pd_all[hold].mean()), "default_rate_holdout": float(y.iloc[hold].mean())},
        "psi_dev_vs_holdout": psi(s_dev, s_hold, bands),
        "challenger": {"model": "HistGradientBoosting", "holdout_auc": gbm_auc,
                       "auc_given_up": gbm_auc - perf_hold["auc"]},
        "fairness": fairness,
        "stress": stress, "missed_payment_to_charge_off": conv,
        "el_missed_payment_basis": float((pd_all * 0.75 * portfolio.ead(bal, lim, 0.5)).sum()),
        "roll_rate_el_12m": roll["total_expected_loss_12m"],
        "segments": segments, "segmentation": seg_info,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    table.to_csv(RESULTS / "scorecard.csv", index=False)

    # Files for the independent R validation.
    woe_all = transform(X, binnings)[feats]
    pd.concat([woe_all.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True)], axis=1).to_csv(RESULTS / "r_dev_woe.csv", index=False)
    pd.DataFrame({"score": s_hold, "default": y.iloc[hold].to_numpy()}).to_csv(RESULTS / "r_holdout_scores.csv", index=False)
    pd.DataFrame({"feature": feats, "coefficient": model.coef_[0]}).to_csv(RESULTS / "py_coefficients.csv", index=False)
    pd.DataFrame([{"metric": "auc", "value": perf_hold["auc"]}, {"metric": "ks", "value": perf_hold["ks"]}]).to_csv(
        RESULTS / "py_holdout_metrics.csv", index=False)


    # Tableau-ready extracts: one tidy table per view.
    TABLEAU.mkdir(parents=True, exist_ok=True)
    out = book.assign(account_id=raw["ID"], sex=prot["sex"], age_band=prot["age_band"].astype(str),
                      sample=np.select([np.isin(idx, dev), np.isin(idx, val), excluded],
                                       ["development", "validation", "excluded (inactive)"], "holdout"),
                      score_band=pd.cut(s_all, bands, labels=[f"Band {i}" for i in range(10, 0, -1)]).astype(str))
    out[["account_id", "sample", "score", "score_band", "pd", "default", "segment", "limit", "balance", "ead", "el",
         "utilization", "months_delinquent_6m", "sex", "age_band"]].to_csv(TABLEAU / "accounts.csv", index=False)
    table.to_csv(TABLEAU / "scorecard.csv", index=False)
    pd.DataFrame(stress).to_csv(TABLEAU / "stress_test.csv", index=False)
    pd.DataFrame(segments).to_csv(TABLEAU / "segments.csv", index=False)
    pd.DataFrame(dec).to_csv(TABLEAU / "holdout_deciles.csv", index=False)

    figures(metrics, s_hold, y.iloc[hold].to_numpy(), dec)
    report(metrics)


def figures(m, s_hold, y_hold, dec):
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    navy, red = "#1f3a5f", "#b3263e"

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar([d["decile"] for d in dec], [d["default_rate"] for d in dec], color=navy, label="Actual default rate")
    ax.plot([d["decile"] for d in dec], [d["predicted"] for d in dec], "o-", color=red, label="Predicted (from score)")
    ax.set(xlabel="Score decile (1 = lowest scores)", ylabel="Default rate", title="Rank ordering and calibration, holdout")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(DOCS / "deciles.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.linspace(min(s_hold), max(s_hold), 40)
    ax.hist(s_hold[y_hold == 0], bins, alpha=0.75, color=navy, label="Paid", density=True)
    ax.hist(s_hold[y_hold == 1], bins, alpha=0.75, color=red, label="Defaulted", density=True)
    ax.set(xlabel="Score", ylabel="Density", title=f"Score separation, holdout (KS {m['performance']['holdout']['ks']:.2f})")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(DOCS / "separation.png", dpi=160); plt.close(fig)

    st = m["stress"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([s["scenario"] for s in st][::-1], [s["expected_loss"] / 1e6 for s in st][::-1], color=[red, "#d27a5a", navy])
    for i, s in enumerate(st[::-1]):
        ax.text(s["expected_loss"] / 1e6, i, f"  NT${s['expected_loss']/1e6:,.0f}M ({s['loss_rate']:.1%})", va="center")
    ax.set(xlabel="Expected loss, NT$ millions", title="Stress test: expected loss on the whole book")
    ax.set_xlim(0, max(s["expected_loss"] for s in st) / 1e6 * 1.45)
    fig.tight_layout(); fig.savefig(DOCS / "stress.png", dpi=160); plt.close(fig)


def report(m):
    p = m["performance"]
    print("features:", m["features"])
    for k in ("development", "validation", "holdout"):
        print(f"  {k:<12} AUC {p[k]['auc']:.3f}  Gini {p[k]['gini']:.3f}  KS {p[k]['ks']:.3f}")
    print("reversals:", [(r["deciles"], round(r["gap"], 3), round(r["z"], 2)) for r in m["reversals"]])
    print(f"excluded inactive: {m['excluded_inactive']}")
    print(f"rank ordered: {m['rank_ordered']}  HL chi2 {m['hosmer_lemeshow']['chi2']:.1f} p={m['hosmer_lemeshow']['p_value']:.3f}  "
          f"PSI {m['psi_dev_vs_holdout']:.4f}  mean PD {m['calibration']['mean_pd_holdout']:.3f} vs actual {m['calibration']['default_rate_holdout']:.3f}")
    print(f"challenger AUC {m['challenger']['holdout_auc']:.3f} (scorecard gives up {m['challenger']['auc_given_up']:.3f})")
    f = m["fairness"]
    print(f"fairness: eligible {f['eligible_share']}  AIR {f['adverse_impact_ratio']:.3f}  within-band gap F-M {f['mean_gap_within_band']:+.3f}")
    print(f"missed payment -> charge-off: {m['missed_payment_to_charge_off']:.3f}; EL if PD were taken as loss: "
          f"NT${m['el_missed_payment_basis']/1e6:,.0f}M; roll-rate 12m EL NT${m['roll_rate_el_12m']/1e6:,.0f}M")
    for s in m["stress"]:
        print(f"  {s['scenario']:<17} miss PD {s['mean_pd']:.3f}  charge-off {s['mean_charge_off_probability']:.4f}  "
              f"EL NT${s['expected_loss']/1e6:,.1f}M  loss rate {s['loss_rate']:.2%}")
    print("segments (k=%d):" % m["segmentation"]["k"])
    for s in m["segments"]:
        print(f"  {s['segment']:<20} {s['share']:.0%}  util {s['avg_utilization_6m']:.2f}  payratio {s['avg_payment_ratio_5m']:.2f} "
              f"delinq {s['months_delinquent_6m']:.2f}  default {s['default_rate']:.1%}  loss share {s['share_of_loss']:.0%}")


if __name__ == "__main__":
    main()
