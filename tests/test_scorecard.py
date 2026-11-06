import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import features, inactive, load_raw
from src.portfolio import stressed_pd
from src.scorecard import BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO, ks, pd_from_score
from src.woe import fit_all

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_the_data_quality_findings_hold(raw):
    assert len(raw) == 30000 and raw["ID"].is_unique
    assert inactive(raw).sum() == 966
    # Status code 1 is almost only used in September: the reason buckets 1 and 2 are merged.
    assert (raw["PAY_1"] == 1).sum() > 3000 and (raw["PAY_3"] == 1).sum() < 10


def test_every_binning_is_monotonic(raw):
    X, y = features(raw), raw["default"]
    for name, b in fit_all(X, y).items():
        rates = [s["default_rate"] for s in b.stats]
        d = np.diff(rates)
        assert (d >= -1e-9).all() or (d <= 1e-9).all(), name


def test_scaling_matches_the_definition():
    odds_good = lambda s: (1 - pd_from_score(s)) / pd_from_score(s)
    assert odds_good(BASE_SCORE) == pytest.approx(BASE_ODDS)
    assert odds_good(BASE_SCORE + PDO) == pytest.approx(2 * BASE_ODDS)


def test_ks_handles_tied_scores():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([1, 1, 1, 2, 2, 2, 3, 3])  # lots of ties, as on a points scorecard
    grid = np.unique(s)
    brute = max(abs((s[y == 1] <= g).mean() - (s[y == 0] <= g).mean()) for g in grid)
    assert ks(y, s) == pytest.approx(brute)


def test_stress_raises_pd_and_baseline_is_unchanged():
    p = np.array([0.01, 0.1, 0.3])
    assert np.allclose(stressed_pd(p, None), p)
    assert (stressed_pd(p, 0.99) > stressed_pd(p, 0.9)).all() and (stressed_pd(p, 0.9) > p).all()


def test_model_passes_its_own_validation_criteria():
    assert M["rank_ordered"]
    assert M["hosmer_lemeshow"]["p_value"] > 0.05
    assert M["psi_dev_vs_holdout"] < 0.1
    assert all(c < 0 for c in M["coefficients"].values())
    assert M["fairness"]["adverse_impact_ratio"] >= 0.8


def test_r_independently_reproduces_the_model():
    r = json.loads((ROOT / "results" / "r_validation.json").read_text())
    assert all(r["checks"].values())


def test_roll_matrix_rows_sum_to_one_and_risk_rises_with_bucket():
    roll = json.loads((ROOT / "results" / "roll_rates.json").read_text())
    for row in roll["matrix_dollar_weighted"].values():
        assert sum(row.values()) == pytest.approx(1, abs=1e-9)
    p = roll["charge_off_12m_by_bucket"]
    assert p["current"] < p["1-2"] < p["3-4"] < p["5+"] == pytest.approx(1)
