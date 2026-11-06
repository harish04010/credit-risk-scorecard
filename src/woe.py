"""
Weight-of-evidence binning, the backbone of a traditional credit scorecard.

Each feature is cut into bins whose default rate moves in one direction only
(monotonic), so the scorecard never says "more late payments, less risk" for
some odd middle range. Each bin is then replaced by its weight of evidence:

    WoE = ln( share of all goods in the bin / share of all bads in the bin )

Positive WoE means safer than average. Information value sums how far the
bins pull goods and bads apart, and is the usual first screen for which
features are worth having.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Binning:
    feature: str
    edges: list[float]                  # upper edges; the last is +inf
    woe: list[float]
    stats: list[dict] = field(default_factory=list)
    iv: float = 0.0

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges[:-1]), x, side="left")

    def transform(self, x) -> np.ndarray:
        return np.asarray(self.woe)[self.bin_index(np.asarray(x, dtype=float))]

    def labels(self) -> list[str]:
        out, lo = [], None
        for hi in self.edges:
            if lo is None:
                out.append(f"<= {hi:g}")
            elif np.isinf(hi):
                out.append(f"> {lo:g}")
            else:
                out.append(f"{lo:g} to {hi:g}")
            lo = hi
        return out


def _table(x, y, edges):
    idx = np.searchsorted(np.asarray(edges[:-1]), x, side="left")
    n = len(edges)
    bad = np.bincount(idx, weights=y, minlength=n)
    tot = np.bincount(idx, minlength=n).astype(float)
    return tot, bad


def fit_bin(x: pd.Series, y: pd.Series, max_bins: int = 8, min_share: float = 0.05) -> Binning:
    x = x.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    # Start fine: distinct values for low-cardinality features, quantiles otherwise.
    uniq = np.unique(x)
    if len(uniq) <= 12:
        edges = list(uniq[:-1]) + [np.inf]
    else:
        qs = np.unique(np.quantile(x, np.linspace(0, 1, 21)[1:-1]))
        edges = list(qs) + [np.inf]

    # Merge until bins are big enough, monotonic in default rate, and few enough.
    while True:
        tot, bad = _table(x, y, edges)
        rate = bad / np.maximum(tot, 1)
        small = np.where(tot < min_share * len(x))[0]
        direction = np.sign(np.corrcoef(np.arange(len(rate)), rate)[0, 1]) if len(rate) > 2 else 1
        violations = [i for i in range(len(rate) - 1) if (rate[i + 1] - rate[i]) * direction < 0]
        if len(edges) <= 2:
            break
        if len(small):
            i = int(small[0])
            j = i if i < len(edges) - 1 else i - 1        # merge with the next bin (or previous, if last)
        elif violations:
            j = violations[0]
        elif len(edges) > max_bins:
            # merge the adjacent pair with the most similar default rates
            j = int(np.argmin(np.abs(np.diff(rate))))
        else:
            break
        del edges[j]  # removing edge j merges bins j and j+1

    tot, bad = _table(x, y, edges)
    good = tot - bad
    g_share = (good + 0.5) / (good.sum() + 0.5 * len(tot))  # +0.5 keeps an all-bad or all-good bin finite
    b_share = (bad + 0.5) / (bad.sum() + 0.5 * len(tot))
    woe = np.log(g_share / b_share)
    iv = float(np.sum((g_share - b_share) * woe))
    b = Binning(x.dtype.name and "", edges, woe.round(6).tolist(), iv=round(iv, 4))
    b.stats = [{"bin": lab, "accounts": int(t), "defaults": int(bd), "default_rate": float(bd / max(t, 1)),
                "woe": float(w)} for lab, t, bd, w in zip(b.labels(), tot, bad, woe)]
    return b


def fit_all(X: pd.DataFrame, y: pd.Series) -> dict[str, Binning]:
    out = {}
    for col in X.columns:
        b = fit_bin(X[col], y)
        b.feature = col
        out[col] = b
    return out


def transform(X: pd.DataFrame, binnings: dict[str, Binning]) -> pd.DataFrame:
    return pd.DataFrame({c: binnings[c].transform(X[c]) for c in binnings}, index=X.index)
