"""Document templates for synthetic data generation.

Each template produces a realistic financial document with controllable
complexity via optional bespoke-clause slots.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from audit_agent.schemas import DocType

CLIENT_NAMES = [
    "Margaret Chen", "David Okonkwo", "Sofia Rodriguez", "James Whitfield",
    "Aisha Patel", "Robert Lindström", "Elena Vasquez", "Thomas Nakamura",
    "Catherine O'Brien", "Hassan Al-Rashid", "Priya Mehta", "William Frost",
]

RISK_LEVELS = ["conservative", "moderate", "moderately_aggressive", "aggressive"]
OBJECTIVES = [
    "capital_preservation", "income", "balanced_growth",
    "long_term_growth", "aggressive_growth",
]
ACCOUNT_TYPES = ["individual", "joint", "trust", "ira", "401k", "corporate"]
BENCHMARKS = ["S&P 500", "Bloomberg US Agg", "MSCI ACWI", "60/40 Blend"]

COMPLIANCE_FLAGS = [
    "PEP screening completed",
    "Sanctions screening: CLEAR",
    "AML verification: PASSED",
    "KYC documentation: COMPLETE",
    "Suitability assessment: CONFIRMED",
    "Risk disclosure: ACKNOWLEDGED",
    "Fee schedule: DISCLOSED",
    "Conflict of interest: NONE IDENTIFIED",
    "Adverse media screening: CLEAR",
    "Source of funds: VERIFIED",
]

BESPOKE_CLAUSES = [
    "Notwithstanding the general allocation guidelines, the client has requested "
    "a specific carve-out of up to 5% in direct private equity co-investments.",
    "The portfolio must maintain a minimum 15% allocation to ESG-screened securities "
    "as per the client's environmental mandate dated {date}.",
    "Except as otherwise provided herein, no single issuer shall represent more than "
    "3% of total portfolio value, with the exception of US Treasury obligations.",
    "The client has imposed a custom restriction prohibiting investment in companies "
    "deriving more than 10% of revenue from fossil fuel extraction.",
    "Subject to quarterly review, the portfolio will maintain a currency hedge ratio "
    "of at least 50% on non-USD denominated holdings.",
    "The client requires that all fixed income holdings maintain a minimum credit "
    "rating of BBB- (investment grade) as assessed by at least two major rating agencies.",
]


@dataclass
class TemplateSlots:
    """Random values to fill template placeholders."""

    client_name: str = ""
    account_type: str = ""
    risk_tolerance: str = ""
    investment_objective: str = ""
    time_horizon: int = 10
    total_value: float = 5_000_000.0
    equity_pct: float = 60.0
    fixed_income_pct: float = 30.0
    cash_pct: float = 10.0
    performance_ytd: float = 7.2
    performance_qtd: float = 2.1
    benchmark_return: float = 6.8
    benchmark: str = "S&P 500"
    compliance_flags: list[str] = field(default_factory=list)
    bespoke_clauses: list[str] = field(default_factory=list)
    liquidity_amount: float = 0.0
    liquidity_horizon_months: int = 0
    date: str = "2026-03-31"
    seed: int = 0

    @classmethod
    def random(cls, rng: random.Random, doc_type: DocType) -> "TemplateSlots":
        name = rng.choice(CLIENT_NAMES)
        risk = rng.choice(RISK_LEVELS)
        obj = rng.choice(OBJECTIVES)
        eq = round(rng.uniform(20, 80), 1)
        fi = round(rng.uniform(10, 80 - eq), 1)
        cash = round(100 - eq - fi, 1)
        total = round(rng.uniform(500_000, 50_000_000), 2)
        horizon = rng.randint(1, 30)
        perf_ytd = round(rng.uniform(-15, 25), 2)
        perf_qtd = round(rng.uniform(-8, 12), 2)
        bench_ret = round(perf_ytd + rng.uniform(-5, 5), 2)

        # Bespoke clauses: more likely for complex doc types
        n_bespoke = 0
        if doc_type in (DocType.QUARTERLY_PORTFOLIO_REVIEW, DocType.INVESTMENT_POLICY_STATEMENT):
            n_bespoke = rng.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]
        elif doc_type == DocType.COMPLIANCE_DISCLOSURE:
            n_bespoke = rng.choices([0, 1, 2], weights=[50, 35, 15])[0]

        bespoke = rng.sample(BESPOKE_CLAUSES, min(n_bespoke, len(BESPOKE_CLAUSES)))
        bespoke = [c.replace("{date}", "2026-01-15") for c in bespoke]

        flags = rng.sample(COMPLIANCE_FLAGS, rng.randint(5, len(COMPLIANCE_FLAGS)))

        liq_amount = round(rng.uniform(50_000, 500_000), 2) if rng.random() > 0.5 else 0
        liq_months = rng.randint(3, 24) if liq_amount > 0 else 0

        return cls(
            client_name=name,
            account_type=rng.choice(ACCOUNT_TYPES),
            risk_tolerance=risk,
            investment_objective=obj,
            time_horizon=horizon,
            total_value=total,
            equity_pct=eq,
            fixed_income_pct=fi,
            cash_pct=cash,
            performance_ytd=perf_ytd,
            performance_qtd=perf_qtd,
            benchmark_return=bench_ret,
            benchmark=rng.choice(BENCHMARKS),
            compliance_flags=flags,
            bespoke_clauses=bespoke,
            liquidity_amount=liq_amount,
            liquidity_horizon_months=liq_months,
            date="2026-03-31",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateSlots":
        return cls(**data)


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _bespoke_section(clauses: list[str]) -> str:
    if not clauses:
        return ""
    lines = "\n".join(f"  - {c}" for c in clauses)
    return f"\n\n## Special Provisions and Bespoke Constraints\n\n{lines}\n"


def render_account_opening(s: TemplateSlots) -> str:
    return f"""# Account Opening Form

## Client Information
- **Client Name:** {s.client_name}
- **Date of Birth:** 1968-04-12
- **Citizenship:** United States
- **Employment Status:** Self-employed

## Account Details
- **Account Type:** {s.account_type}
- **Date Opened:** {s.date}

## Financial Profile
- **Annual Income:** {_fmt_money(s.total_value * 0.08)}
- **Net Worth:** {_fmt_money(s.total_value * 1.5)}
- **Source of Funds:** Business income and inheritance
- **Investment Objective:** {s.investment_objective.replace('_', ' ').title()}
- **Risk Tolerance:** {s.risk_tolerance.replace('_', ' ').title()}

## Regulatory Disclosures
- PEP Status: No
- {chr(10).join('- ' + f for f in s.compliance_flags[:4])}
"""


def render_ips(s: TemplateSlots) -> str:
    liq_section = ""
    if s.liquidity_amount > 0:
        liq_section = (
            f"\n### Liquidity Requirements\n"
            f"The client anticipates withdrawing approximately {_fmt_money(s.liquidity_amount)} "
            f"within the next {s.liquidity_horizon_months} months for personal expenditures.\n"
        )

    return f"""# Investment Policy Statement

**Prepared for:** {s.client_name}
**Date:** {s.date}
**Review Frequency:** Annual

## Investment Objective
The primary investment objective is {s.investment_objective.replace('_', ' ')}. The client
seeks to achieve long-term capital appreciation while managing downside risk consistent with
their stated tolerance level.

## Risk Profile
- **Risk Tolerance:** {s.risk_tolerance.replace('_', ' ').title()}
- **Time Horizon:** {s.time_horizon} years
- **Maximum Acceptable Drawdown:** {15 + RISK_LEVELS.index(s.risk_tolerance) * 10}%
{liq_section}
## Target Asset Allocation
| Asset Class    | Target (%) | Range (%)     |
|---------------|-----------|---------------|
| Equities      | {s.equity_pct}      | {max(0, s.equity_pct-10)}-{min(100, s.equity_pct+10)} |
| Fixed Income  | {s.fixed_income_pct}      | {max(0, s.fixed_income_pct-10)}-{min(100, s.fixed_income_pct+10)} |
| Cash & Equiv. | {s.cash_pct}       | 0-{s.cash_pct+10}         |

## Constraints
- No investments in tobacco or firearms manufacturers
- Maximum single-issuer concentration: 5% of portfolio value
- All fixed income must be investment grade (BBB- or above)
{_bespoke_section(s.bespoke_clauses)}
## Rebalancing
The portfolio shall be rebalanced quarterly or when any asset class deviates more than
5 percentage points from its target allocation.

## Benchmark
Performance will be measured against the {s.benchmark}.

## Tax Considerations
The client is in the highest marginal tax bracket. Tax-loss harvesting should be
pursued opportunistically. Municipal bonds are preferred for fixed income allocation.
"""


def render_quarterly_review(s: TemplateSlots) -> str:
    eq_val = round(s.total_value * s.equity_pct / 100, 2)
    fi_val = round(s.total_value * s.fixed_income_pct / 100, 2)
    cash_val = round(s.total_value * s.cash_pct / 100, 2)

    return f"""# Quarterly Portfolio Review

**Client:** {s.client_name}
**Period:** Q1 2026 (January 1 – March 31, 2026)
**Total Portfolio Value:** {_fmt_money(s.total_value)}

## Performance Summary
| Metric                | Value    |
|----------------------|----------|
| QTD Return           | {s.performance_qtd}%   |
| YTD Return           | {s.performance_ytd}%   |
| Benchmark ({s.benchmark}) | {s.benchmark_return}%   |
| Excess Return        | {round(s.performance_ytd - s.benchmark_return, 2)}%   |

## Asset Allocation
| Asset Class    | Value           | Weight (%) |
|---------------|----------------|-----------|
| Equities      | {_fmt_money(eq_val)}  | {s.equity_pct}      |
| Fixed Income  | {_fmt_money(fi_val)}  | {s.fixed_income_pct}      |
| Cash & Equiv. | {_fmt_money(cash_val)}  | {s.cash_pct}       |
| **Total**     | **{_fmt_money(s.total_value)}** | **100.0** |

## Risk Metrics
- Portfolio Volatility (annualized): {round(8 + RISK_LEVELS.index(s.risk_tolerance) * 4, 1)}%
- Sharpe Ratio: {round(0.5 + s.performance_ytd / 20, 2)}
- Maximum Drawdown (QTD): {round(abs(min(s.performance_qtd, 0)) + 2, 1)}%

## Commentary
The portfolio {"outperformed" if s.performance_ytd > s.benchmark_return else "underperformed"} \
its benchmark by {abs(round(s.performance_ytd - s.benchmark_return, 2))} percentage points \
year-to-date. {"The equity sleeve was the primary contributor to outperformance, driven by overweight positions in technology and healthcare sectors." if s.performance_ytd > s.benchmark_return else "Underperformance was primarily attributable to the fixed income allocation, which was negatively impacted by rising interest rates during the quarter."}

Fees charged during the quarter: {_fmt_money(s.total_value * 0.0025)}
{_bespoke_section(s.bespoke_clauses)}
## Strategy Outlook
{"No changes to the current strategy are recommended at this time." if abs(s.performance_ytd - s.benchmark_return) < 3 else "Given the significant deviation from benchmark, we recommend reviewing the current allocation and considering a tactical rebalancing."}
"""


def render_compliance_disclosure(s: TemplateSlots) -> str:
    flags_section = "\n".join(f"- {f}" for f in s.compliance_flags)

    return f"""# Compliance Disclosure Report

**Client:** {s.client_name}
**Report Date:** {s.date}
**Regulatory Jurisdiction:** United States (SEC/FINRA)

## Client Due Diligence Summary

### KYC Status
Know Your Customer documentation has been collected and verified. All identity
documents are current and on file.

### PEP Screening
The client {"is" if "PEP" in str(s.bespoke_clauses) else "is not"} identified as a Politically Exposed Person. {"Enhanced due diligence measures have been applied." if "PEP" in str(s.bespoke_clauses) else "Standard due diligence applies."}

### Sanctions Screening
The client has been screened against OFAC, EU, and UN sanctions lists.
Result: CLEAR — no matches identified.

### Anti-Money Laundering
AML verification has been completed. Transaction monitoring is active.
Source of funds has been verified as: Business income and inheritance.

## Compliance Flags and Attestations
{flags_section}

## Suitability Assessment
The recommended investment strategy has been assessed for suitability given the
client's stated objectives ({s.investment_objective.replace('_', ' ')}), risk tolerance
({s.risk_tolerance.replace('_', ' ')}), and financial situation. The assessment confirms
the strategy is suitable.
{_bespoke_section(s.bespoke_clauses)}
## Risk Disclosures
The client has been provided with and acknowledged the following risk disclosures:
- Market risk and potential for loss of principal
- Liquidity risk for alternative investments
- Interest rate risk for fixed income holdings
- Currency risk for international investments
- Concentration risk limitations

## Fee Disclosure
All applicable fees have been disclosed to the client in writing.
Management fee: 1.00% per annum on assets under management.
"""


RENDERERS = {
    DocType.ACCOUNT_OPENING: render_account_opening,
    DocType.INVESTMENT_POLICY_STATEMENT: render_ips,
    DocType.QUARTERLY_PORTFOLIO_REVIEW: render_quarterly_review,
    DocType.COMPLIANCE_DISCLOSURE: render_compliance_disclosure,
}


def render_document(doc_type: DocType, slots: TemplateSlots) -> str:
    return RENDERERS[doc_type](slots)


def generate_batch(
    n: int = 50,
    seed: int = 42,
    weights: dict[DocType, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate n documents with the specified type distribution.

    Returns list of {doc_id, doc_type, text, slots} dicts.
    """
    if weights is None:
        weights = {
            DocType.ACCOUNT_OPENING: 0.30,
            DocType.INVESTMENT_POLICY_STATEMENT: 0.30,
            DocType.QUARTERLY_PORTFOLIO_REVIEW: 0.25,
            DocType.COMPLIANCE_DISCLOSURE: 0.15,
        }

    rng = random.Random(seed)
    types = list(weights.keys())
    probs = list(weights.values())

    docs = []
    for i in range(n):
        dt = rng.choices(types, probs)[0]
        slots = TemplateSlots.random(rng, dt)
        text = render_document(dt, slots)
        docs.append({
            "doc_id": f"DOC_{i:04d}",
            "doc_type": dt.value,
            "text": text,
            "slots": slots,
        })
    return docs
