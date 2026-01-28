import numpy as np

from src.modeling.forecast import compute_risk_score


def test_risk_score_on_track():
    score = compute_risk_score(due_weeks=10, eta_weeks=5)
    assert 0 <= score <= 1
    assert score < 0.5


def test_risk_score_overdue():
    score = compute_risk_score(due_weeks=4, eta_weeks=6)
    assert score == 1.0


def test_risk_score_zero_due():
    score = compute_risk_score(due_weeks=0, eta_weeks=2)
    assert score == 1.0


def test_risk_score_nan():
    score = compute_risk_score(due_weeks=np.nan, eta_weeks=2)
    assert np.isnan(score)
