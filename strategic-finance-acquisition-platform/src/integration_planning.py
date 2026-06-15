from __future__ import annotations

import pandas as pd

from .acquisition_screening import AcquisitionAssumptions, accounting_alignment_risk


def generate_integration_plan(assumptions: AcquisitionAssumptions) -> pd.DataFrame:
    accounting_risk = accounting_alignment_risk(assumptions.accounting_framework)
    jurisdiction_risk = "High" if assumptions.jurisdictions >= 8 else "Medium" if assumptions.jurisdictions >= 5 else "Low"
    people_risk = "High" if assumptions.employees >= 1000 else "Medium" if assumptions.employees >= 300 else "Low"

    rows = [
        {
            "Workstream": "Finance systems integration",
            "Priority": "Critical",
            "Risk Level": "High" if accounting_risk != "Low" else "Medium",
            "Owner / Function": "Finance Transformation",
            "Timeline": "Day 0 to Day 60",
            "Rationale": "Ensure close process, entity ledger structure, billing feeds, and management reporting can be consolidated quickly.",
        },
        {
            "Workstream": "Reporting consolidation",
            "Priority": "Critical",
            "Risk Level": "High" if assumptions.jurisdictions > 6 else "Medium",
            "Owner / Function": "Group Controllership",
            "Timeline": "Day 0 to Day 45",
            "Rationale": "Create a reliable post-close reporting pack covering revenue, EBITDA, cash, debt, and working capital.",
        },
        {
            "Workstream": "Intercompany flows",
            "Priority": "High",
            "Risk Level": jurisdiction_risk,
            "Owner / Function": "Tax and Treasury",
            "Timeline": "Day 30 to Day 90",
            "Rationale": "Map invoicing, transfer pricing, cash pooling, and settlement mechanics across entities.",
        },
        {
            "Workstream": "Revenue recognition alignment",
            "Priority": "Critical",
            "Risk Level": accounting_risk,
            "Owner / Function": "Revenue Accounting",
            "Timeline": "Day 0 to Day 60",
            "Rationale": "Compare subscription, refund, deferred revenue, and contract-modification policies before consolidation.",
        },
        {
            "Workstream": "Chart of accounts alignment",
            "Priority": "High",
            "Risk Level": "Medium",
            "Owner / Function": "Controllership",
            "Timeline": "Day 15 to Day 75",
            "Rationale": "Map local accounts into the group reporting taxonomy and preserve audit trail integrity.",
        },
        {
            "Workstream": "Budgeting and forecasting integration",
            "Priority": "High",
            "Risk Level": "Medium",
            "Owner / Function": "FP&A",
            "Timeline": "Day 30 to Day 100",
            "Rationale": "Embed the target into rolling forecast cadence, scenario planning, and post-close synergy tracking.",
        },
        {
            "Workstream": "Tax and jurisdiction complexity",
            "Priority": "High",
            "Risk Level": jurisdiction_risk,
            "Owner / Function": "Tax",
            "Timeline": "Day 0 to Day 100",
            "Rationale": "Assess local filings, indirect tax, permanent establishment exposure, payroll taxes, and transfer pricing.",
        },
        {
            "Workstream": "Operational integration risks",
            "Priority": "Medium",
            "Risk Level": people_risk,
            "Owner / Function": "Operations and People",
            "Timeline": "Day 30 to Day 100",
            "Rationale": "Protect continuity in billing operations, customer support handoffs, procurement, hiring, and culture integration.",
        },
    ]
    return pd.DataFrame(rows)


def hundred_day_plan(plan: pd.DataFrame) -> pd.DataFrame:
    phases = [
        ("Day 0-30", "Stabilise reporting, cash visibility, governance cadence, and closing responsibilities."),
        ("Day 31-60", "Complete policy gap analysis, ledger mapping, forecast bridge, and initial synergy baseline."),
        ("Day 61-90", "Move to recurring integrated reporting, resolve high-risk intercompany and tax items."),
        ("Day 91-100", "Present board-ready integration scorecard, risk burndown, and updated forecast range."),
    ]
    return pd.DataFrame(phases, columns=["Phase", "Action"])
