"""Turning WoE features into a points-based scorecard, and validating it."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .woe import Binning

# Scaling: 600 points means odds of 50 goods to 1 bad, and every 20 points doubles the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def select_features(binnings: dict[str, Binning], woe_dev: pd.DataFrame, min_iv=0.02, max_corr=0.7) -> tuple[list[str], list[dict]]:
    """Strongest first; drop anything too weak, or too correlated with a stronger feature already in."""
    log, kept = [], []
    for f in sorted(binnings, key=lambda c: -binnings[c].iv):
        if binnings[f].iv < min_iv:
            log.append({"feature": f, "decision": "dropped", "why": f"IV {binnings[f].iv:.3f} below {min_iv}"})
            continue
        clash = next((k for k in kept if abs(np.corrcoef(woe_dev[f], woe_dev[k])[0, 1]) > max_corr), None)
        if clash:
            r = np.corrcoef(woe_dev[f], woe_dev[clash])[0, 1]
            log.append({"feature": f, "decision": "dropped", "why": f"correlation {r:.2f} with {clash}, which has higher IV"})
            continue
        kept.append(f)
        log.append({"feature": f, "decision": "kept", "why": f"IV {binnings[f].iv:.3f}"})
    return kept, log


def fit(woe_dev: pd.DataFrame, y, features: list[str]) -> tuple[LogisticRegression, list[str], list[dict]]:
    """
    Logistic regression on WoE. With WoE defined as ln(good/bad), every
    coefficient should be negative: higher WoE, lower chance of default. A
    positive one means the feature is fighting the others (collinearity), so
    it comes out and the model is refit.
    """
    log, feats = [], list(features)
    while True:
        m = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit(woe_dev[feats], y)
        wrong = [f for f, c in zip(feats, m.coef_[0]) if c > 0]
        if wrong:
            worst = max(wrong, key=lambda f: m.coef_[0][feats.index(f)])
            log.append({"feature": worst, "decision": "dropped", "why": "coefficient had the wrong sign once others were in"})
            feats.remove(worst)
            continue
        z = wald_z(m, woe_dev[feats])
        weak = [f for f, zz in zip(feats, z) if abs(zz) < 1.96]
        if weak:
            worst = min(weak, key=lambda f: abs(z[feats.index(f)]))
            log.append({"feature": worst, "decision": "dropped",
                        "why": f"not statistically significant once others were in (|z| = {abs(z[feats.index(worst)]):.2f})"})
            feats.remove(worst)
            continue
        return m, feats, log


def wald_z(model, X: pd.DataFrame) -> np.ndarray:
    """
    Coefficient / standard error, from the inverse of the Fisher information.
    scikit-learn doesn't report these, and a bank scorecard shouldn't carry a
    variable whose effect can't be told apart from zero.
    """
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    p = model.predict_proba(X)[:, 1]
    info = A.T @ (A * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(info)))[1:]
    return model.coef_[0] / se


def points_table(model, features, binnings) -> pd.DataFrame:
    b0, k = model.intercept_[0], len(features)
    rows = []
    for f, coef in zip(features, model.coef_[0]):
        for s in binnings[f].stats:
            pts = -(coef * s["woe"]) * FACTOR + (OFFSET - FACTOR * b0) / k
            rows.append({"feature": f, "bin": s["bin"], "points": int(round(pts)), "accounts": s["accounts"],
                         "default_rate": round(s["default_rate"], 4), "woe": round(s["woe"], 4)})
    return pd.DataFrame(rows)


def score(X: pd.DataFrame, table: pd.DataFrame, binnings) -> np.ndarray:
    total = np.zeros(len(X))
    for f, g in table.groupby("feature", sort=False):
        total += g["points"].to_numpy()[binnings[f].bin_index(X[f].to_numpy(dtype=float))]
    return total


def pd_from_score(s) -> np.ndarray:
    """Invert the scaling: score -> odds -> probability of default."""
    odds_good = np.exp((np.asarray(s) - OFFSET) / FACTOR)
    return 1 / (1 + odds_good)


# ------------------------------------------------------------------ validation

def ks(y, s) -> float:
    """
    Largest gap between the score distributions of goods and bads, evaluated
    at each distinct score. An earlier version walked through accounts one at
    a time, which splits tied scores arbitrarily and overstates KS when many
    accounts share a score, as they do on a points scorecard. The independent
    R validation caught it.
    """
    y, s = np.asarray(y), np.asarray(s)
    grid = np.unique(s)
    f_bad = np.searchsorted(np.sort(s[y == 1]), grid, side="right") / (y == 1).sum()
    f_good = np.searchsorted(np.sort(s[y == 0]), grid, side="right") / (y == 0).sum()
    return float(np.max(np.abs(f_bad - f_good)))


def deciles(y, s, pd_hat) -> list[dict]:
    q = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False)
    out = []
    for d in range(10):
        m = (q == d).to_numpy()
        out.append({"decile": d + 1, "min_score": float(np.min(s[m])), "max_score": float(np.max(s[m])),
                    "accounts": int(m.sum()), "default_rate": float(np.mean(np.asarray(y)[m])),
                    "predicted": float(np.mean(pd_hat[m]))})
    return out


def significant_reversals(dec: list[dict], z_crit: float = 1.96) -> list[dict]:
    """
    Rank ordering: each higher score decile should default less. Small wobbles
    are expected from sampling noise; a reversal only counts if a two-proportion
    z-test says the higher decile really defaults more.
    """
    out = []
    for lo, hi in zip(dec, dec[1:]):
        if hi["default_rate"] <= lo["default_rate"]:
            continue
        p = (lo["default_rate"] * lo["accounts"] + hi["default_rate"] * hi["accounts"]) / (lo["accounts"] + hi["accounts"])
        se = np.sqrt(p * (1 - p) * (1 / lo["accounts"] + 1 / hi["accounts"]))
        z = (hi["default_rate"] - lo["default_rate"]) / se
        out.append({"deciles": [lo["decile"], hi["decile"]], "gap": hi["default_rate"] - lo["default_rate"],
                    "z": float(z), "significant": bool(z > z_crit)})
    return out


def hosmer_lemeshow(y, p, groups=10) -> dict:
    """Do predicted default rates match observed ones across risk groups? Large p-value: no evidence they don't."""
    q = pd.qcut(pd.Series(p).rank(method="first"), groups, labels=False).to_numpy()
    y = np.asarray(y)
    chi = 0.0
    for g in range(groups):
        m = q == g
        n, obs, exp = m.sum(), y[m].sum(), p[m].sum()
        chi += (obs - exp) ** 2 / (exp * (1 - exp / n))
    return {"chi2": float(chi), "df": groups - 2, "p_value": float(1 - stats.chi2.cdf(chi, groups - 2))}


def psi(expected, actual, bins) -> float:
    e = np.histogram(expected, bins)[0] / len(expected)
    a = np.histogram(actual, bins)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def gini(y, s_risk) -> float:
    return 2 * roc_auc_score(y, s_risk) - 1
