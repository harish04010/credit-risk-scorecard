"""
Portfolio views the risk team needs from the scorecard: expected loss,
stress scenarios, and segments.

Expected loss = PD x LGD x EAD
    PD   probability of default, from the scorecard
    LGD  share of the exposure lost when an account defaults
    EAD  exposure at default: today's balance plus some of the unused limit,
         because accounts heading for default tend to draw it down first
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Basel's asset correlation for qualifying revolving retail exposures (credit cards).
RHO = 0.04

SCENARIOS = {
    #                     systematic-factor quantile, LGD, credit conversion factor
    "Baseline":           {"quantile": None, "lgd": 0.75, "ccf": 0.50,
                           "story": "Today's predicted default rates."},
    "Adverse":            {"quantile": 0.90, "lgd": 0.80, "ccf": 0.60,
                           "story": "A downturn severe enough to happen about once a decade."},
    "Severely adverse":   {"quantile": 0.99, "lgd": 0.85, "ccf": 0.70,
                           "story": "About a 1-in-100-year shock: unemployment spikes, recoveries fall, customers draw down lines."},
}


def stressed_pd(pd_base, quantile, rho=RHO):
    """
    Vasicek single-factor model, the one behind Basel capital formulas. It gives
    an account's default probability in a bad state of the economy: the
    systematic factor at the given quantile, holding each account's own risk fixed.
    """
    if quantile is None:
        return pd_base
    pd_base = np.clip(pd_base, 1e-6, 1 - 1e-6)
    return norm.cdf((norm.ppf(pd_base) + np.sqrt(rho) * norm.ppf(quantile)) / np.sqrt(1 - rho))


def ead(balance, limit, ccf):
    bal = np.clip(balance, 0, None)
    return bal + ccf * np.clip(limit - bal, 0, None)


def stress_test(pd_base, balance, limit, to_charge_off: float) -> list[dict]:
    """
    pd_base is the scorecard's probability of missing next month's payment.
    A missed payment isn't a loss: to_charge_off (from the roll-rate chain)
    is how often one becomes a charge-off within 12 months.
    """
    rows = []
    for name, s in SCENARIOS.items():
        p = stressed_pd(pd_base, s["quantile"])
        e = ead(balance, limit, s["ccf"])
        el = p * to_charge_off * s["lgd"] * e
        rows.append({"scenario": name, "story": s["story"], "mean_pd": float(p.mean()),
                     "mean_charge_off_probability": float((p * to_charge_off).mean()), "lgd": s["lgd"], "ccf": s["ccf"],
                     "ead": float(e.sum()), "expected_loss": float(el.sum()), "loss_rate": float(el.sum() / e.sum())})
    return rows


SEGMENT_FEATURES = ["avg_utilization_6m", "avg_payment_ratio_5m", "months_delinquent_6m", "limit"]


def segment(X: pd.DataFrame, seed: int = 7) -> tuple[np.ndarray, dict]:
    """K-means on behavior, with k chosen by silhouette. Segments are named from their profiles, not assumed."""
    Z = X[SEGMENT_FEATURES].copy()
    Z["limit"] = np.log(Z["limit"])
    Z = StandardScaler().fit_transform(Z)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Z), 6000, replace=False)
    scores = {}
    for k in range(3, 7):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z[sample])
        scores[k] = float(silhouette_score(Z[sample], lab))
    k = max(scores, key=scores.get)
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    return labels, {"silhouette_by_k": scores, "k": k}


def name_segment(profile: dict) -> str:
    if profile["months_delinquent_6m"] >= 2:
        return "Behind on payments"
    if profile["avg_payment_ratio_5m"] >= 0.8 and profile["avg_utilization_6m"] < 0.3:
        return "Pays in full"
    if profile["avg_utilization_6m"] >= 0.6:
        return "Heavy revolvers"
    if profile["avg_utilization_6m"] < 0.15:
        return "Light users"
    return "Steady revolvers"
