from src.acquisition_screening import AcquisitionAssumptions, screen_acquisition
from src.board_memo import generate_board_memo
from src.financial_due_diligence import analyse_financial_due_diligence
from src.integration_planning import generate_integration_plan
from src.scenario_analysis import run_scenarios


def test_board_memo_generates_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assumptions = AcquisitionAssumptions()
    screening = screen_acquisition(assumptions)
    diligence = analyse_financial_due_diligence(assumptions)
    scenarios = run_scenarios(assumptions)
    integration_plan = generate_integration_plan(assumptions)

    memo = generate_board_memo(assumptions, screening, diligence, scenarios, integration_plan, use_llm=True)

    assert "Executive Summary" in memo
    assert "Strategic Rationale" in memo
    assert "100-Day Integration Plan" in memo
    assert assumptions.company_name in memo
