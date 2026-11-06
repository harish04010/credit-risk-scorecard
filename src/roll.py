"""
Runs the roll-rate SQL and turns the matrix into a loss forecast.

Net flow: the chance a dollar in a bucket rolls to the next worse one. Chain
them and you get each bucket's flow-to-loss: the share of today's balance
expected to reach the charge-off stage. It's an independent check on the
scorecard's expected loss, built from a completely different method.

    python -m src.roll
"""
import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["current", "1-2", "3-4", "5+"]
LGD = 0.75


HORIZON = 12


def charge_off_probability(matrix: dict, months: int = HORIZON) -> dict:
    """
    Treat 5+ as absorbing (charged off), and step the chain forward month by
    month. The result is each bucket's chance of reaching charge-off within
    the horizon, counting dollars that sit in a bucket for a while before
    rolling, which a single month's flow misses.
    """
    P = np.array([[matrix[a][b] for b in ORDER] for a in ORDER])
    P[3] = [0, 0, 0, 1]
    Pn = np.linalg.matrix_power(P, months)
    return {b: float(Pn[i, 3]) for i, b in enumerate(ORDER)}


def main():
    from .data import load_raw
    load_raw().to_csv(ROOT / "results" / "sql_input.csv", index=False)
    con = duckdb.connect()
    con.execute(f"set file_search_path = '{ROOT}'")
    con.execute((ROOT / "sql" / "roll_rates.sql").read_text())
    rows = con.execute("select from_bucket, to_bucket, dollar_rate, account_rate from roll_matrix").fetchall()
    rate = {(a, b): (d or 0.0) for a, b, d, _ in rows}
    book = {b: (n, bal) for b, n, bal in con.execute("select bucket, accounts, balance from book").fetchall()}

    forward = {ORDER[i]: rate.get((ORDER[i], ORDER[i + 1]), 0.0) for i in range(3)}
    flow_to_loss = {}
    for i, b in enumerate(ORDER):
        flow_to_loss[b] = float(np.prod([forward[ORDER[j]] for j in range(i, 3)])) if b != "5+" else rate.get(("5+", "5+"), 0.0)
    forecast = {b: {"accounts": int(book[b][0]), "balance": float(book[b][1]), "flow_to_loss": flow_to_loss[b],
                    "expected_loss": float(book[b][1] * flow_to_loss[b] * LGD)} for b in ORDER}
    total = sum(v["expected_loss"] for v in forecast.values())
    matrix = {a: {b: rate.get((a, b), 0.0) for b in ORDER} for a in ORDER}
    p12 = charge_off_probability(matrix)
    for b in ORDER:
        forecast[b]["charge_off_12m"] = p12[b]
        forecast[b]["expected_loss_12m"] = float(book[b][1] * p12[b] * LGD)
    total12 = sum(v["expected_loss_12m"] for v in forecast.values())
    out = {"matrix_dollar_weighted": matrix, "net_forward_roll": forward, "by_bucket": forecast,
           "charge_off_12m_by_bucket": p12, "total_expected_loss_12m": total12,
           # A customer who misses next month's payment lands in 1-2. This is how
           # often that ends in a charge-off: the bridge from the scorecard's PD to a loss.
           "missed_payment_to_charge_off": p12["1-2"],
           "total_expected_loss": total, "lgd": LGD,
           "balance": sum(v["balance"] for v in forecast.values())}
    (ROOT / "results" / "roll_rates.json").write_text(json.dumps(out, indent=1))
    print("roll matrix (dollar-weighted):")
    for a in ORDER:
        print(f"  {a:<8}", "  ".join(f"{b}:{matrix[a][b]:.3f}" for b in ORDER))
    for b in ORDER:
        f = forecast[b]
        print(f"  {b:<8} {f['accounts']:>6} accts  NT${f['balance']/1e6:>8,.1f}M  flow-to-loss {f['flow_to_loss']:.4f}  "
              f"EL NT${f['expected_loss']/1e6:,.1f}M")
    print(f"one-month flow EL NT${total/1e6:,.1f}M; 12-month EL NT${total12/1e6:,.1f}M on NT${out['balance']/1e6:,.0f}M "
          f"({total12/out['balance']:.2%}); P(charge-off | newly late) {p12['1-2']:.3f}; from current {p12['current']:.4f}")


if __name__ == "__main__":
    main()
