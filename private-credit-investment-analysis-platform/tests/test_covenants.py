from src.covenants import analyse_covenants
from src.credit_metrics import CreditAssumptions
from src.scenario_analysis import run_scenarios


def test_covenants_identify_breach():
    scenarios = run_scenarios(CreditAssumptions())
    result = analyse_covenants(scenarios, max_net_leverage=2.0, min_interest_coverage=5.0, min_liquidity=200.0)

    assert result["first_breach"] != "None"
    assert "Fail" in set(result["table"]["Status"])


def test_covenants_pass_with_loose_thresholds():
    scenarios = run_scenarios(CreditAssumptions())
    result = analyse_covenants(scenarios, max_net_leverage=10.0, min_interest_coverage=0.1, min_liquidity=1.0)

    assert result["first_breach"] == "None"
    assert set(result["table"]["Status"]) == {"Pass"}
