from src.credit_metrics import CreditAssumptions
from src.scenario_analysis import run_scenarios


def test_scenario_analysis_outputs_all_cases():
    scenarios = run_scenarios(CreditAssumptions())

    assert set(scenarios["Case"]) == {"Base", "Downside", "Severe Downside"}
    assert {"Revenue", "EBITDA", "Free Cash Flow", "Net Debt", "Net Leverage", "Interest Coverage", "Liquidity"}.issubset(scenarios.columns)


def test_resilience_score_is_bounded():
    scenarios = run_scenarios(CreditAssumptions())

    assert scenarios["Downside Resilience Score"].between(0, 100).all()
