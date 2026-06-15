from src.covenants import analyse_covenants
from src.credit_metrics import CreditAssumptions, calculate_credit_metrics, credit_risks, credit_strengths
from src.investment_memo import generate_investment_memo
from src.recommendation import recommend_credit
from src.scenario_analysis import run_scenarios


def test_recommendation_returns_valid_category():
    assumptions = CreditAssumptions()
    metrics = calculate_credit_metrics(assumptions)
    scenarios = run_scenarios(assumptions)
    covenants = analyse_covenants(scenarios)
    recommendation = recommend_credit(metrics, scenarios, covenants)

    assert recommendation["category"] in {"Attractive", "Watchlist", "High risk", "Avoid"}
    assert 0 <= recommendation["score"] <= 100


def test_memo_generates_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assumptions = CreditAssumptions()
    metrics = calculate_credit_metrics(assumptions)
    scenarios = run_scenarios(assumptions)
    covenants = analyse_covenants(scenarios)
    recommendation = recommend_credit(metrics, scenarios, covenants)
    memo = generate_investment_memo(
        assumptions,
        metrics,
        credit_strengths(assumptions, metrics),
        credit_risks(assumptions, metrics),
        scenarios,
        covenants,
        recommendation,
        use_llm=True,
    )

    assert "Executive Summary" in memo
    assert "Covenant Headroom" in memo
    assert "Investment Recommendation" in memo
