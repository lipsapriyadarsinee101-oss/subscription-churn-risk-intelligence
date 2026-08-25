import numpy as np

from src.data.generate import generate_members
from src.features.build import calculate_clv, validate
from src.monitoring.drift import drift_status, population_stability_index
from src.scoring.cohorts import risk_cohort, score_members


def test_generated_data_contract():
    frame = generate_members(200, seed=7)
    validate(frame)
    assert frame.member_id.is_unique
    assert 0 < frame.churned_60d.mean() < 1


def test_cohort_boundaries():
    assert risk_cohort(.30) == "HIGH"
    assert risk_cohort(.15) == "MEDIUM"
    assert risk_cohort(.14) == "LOW"


def test_clv_decreases_with_risk():
    frame = generate_members(2)
    frame["monthly_fee"] = 50
    values = calculate_clv(frame, [.2, .8])
    assert values.iloc[0] > values.iloc[1]


def test_cdp_score_contract():
    frame = generate_members(3)
    output = score_members(frame, [.1, .2, .8])
    assert output.risk_cohort.tolist() == ["LOW", "MEDIUM", "HIGH"]
    assert output.member_id.is_unique


def test_drift_detection():
    rng = np.random.default_rng(1)
    psi = population_stability_index(rng.normal(0, 1, 2000), rng.normal(1, 1, 2000))
    assert drift_status(psi) == "ALERT"
