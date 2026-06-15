from src.acquisition_screening import AcquisitionAssumptions
from src.scenario_analysis import run_scenarios


def test_scenarios_return_required_cases_and_outputs():
    assumptions = AcquisitionAssumptions()
    scenarios = run_scenarios(assumptions)

    assert set(scenarios["Case"]) == {"Downside", "Base", "Upside"}
    assert {"Revenue", "EBITDA", "FCF", "Net Debt", "Net Leverage", "Liquidity"}.issubset(scenarios.columns)


def test_downside_resilience_is_bounded():
    assumptions = AcquisitionAssumptions()
    scenarios = run_scenarios(assumptions)

    assert scenarios["Downside Resilience Score"].between(0, 100).all()
