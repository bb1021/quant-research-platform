from src.acquisition_screening import AcquisitionAssumptions, screen_acquisition


def test_screening_outputs_core_metrics():
    assumptions = AcquisitionAssumptions(revenue=100, ebitda=25, cash=10, debt=40, purchase_price=300)
    result = screen_acquisition(assumptions)

    metrics = result["metrics"]
    assert metrics["enterprise_value"] == 330
    assert metrics["net_debt"] == 30
    assert metrics["net_leverage"] == 1.2
    assert metrics["ev_revenue"] == 3.3
    assert metrics["ev_ebitda"] == 13.2
    assert 0 <= metrics["score"] <= 100


def test_screening_flags_high_leverage():
    assumptions = AcquisitionAssumptions(revenue=80, ebitda=10, cash=5, debt=65, purchase_price=180)
    result = screen_acquisition(assumptions)

    assert any("leverage" in flag.lower() for flag in result["risk_flags"])
