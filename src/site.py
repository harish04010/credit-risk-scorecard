"""Exports the scorecard and its validation results to site/data.json for the web page."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export() -> str:
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    r = json.loads((ROOT / "results" / "r_validation.json").read_text())
    with open(ROOT / "results" / "scorecard.csv") as f:
        rows = list(csv.DictReader(f))
    card = {}
    for row in rows:
        card.setdefault(row["feature"], []).append(
            {"bin": row["bin"], "points": int(row["points"]), "accounts": int(row["accounts"]),
             "default_rate": float(row["default_rate"])})
    p = m["performance"]["holdout"]
    data = {
        "scaling": m["scaling"],
        "card": card,
        "cutoff": 538,
        "deciles": [{k: d[k] for k in ("decile", "min_score", "max_score", "default_rate", "predicted")} for d in m["holdout_deciles"]],
        "metrics": {"auc": p["auc"], "gini": p["gini"], "ks": p["ks"], "hl_p": m["hosmer_lemeshow"]["p_value"],
                    "psi": m["psi_dev_vs_holdout"], "holdout": p["accounts"]},
        "challenger": m["challenger"],
        "r": r,
        "fairness": {"air": m["fairness"]["adverse_impact_ratio"], "eligible": m["fairness"]["eligible_share"]},
        "stress": [{k: s[k] for k in ("scenario", "story", "lgd", "ccf", "expected_loss", "loss_rate")} for s in m["stress"]],
        "loss_basis": {"missed_payment": m["el_missed_payment_basis"], "charge_off": m["stress"][0]["expected_loss"],
                       "charge_off_rate": m["missed_payment_to_charge_off"]},
        "segments": [{k: s[k] for k in ("segment", "share", "default_rate", "mean_score", "share_of_loss", "avg_utilization_6m")}
                     for s in m["segments"]],
        "quality": m["data"]["quality"],
        "selection": m["selection"],
    }
    out = ROOT / "site" / "data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    return f"{out}: {out.stat().st_size:,} bytes"


if __name__ == "__main__":
    print(export())
