"""Primary LLM extractor — simulates the document-processing system.

Extracts structured fields with the mandatory citation contract
(raw_quote + char_span per field) and a prose summary.
"""

from __future__ import annotations

import logging
from typing import Any

from audit_agent.docgen.templates import TemplateSlots
from audit_agent.nim_client import NIMClient
from audit_agent.schemas import DocType, ExtractionOutput, FieldExtraction

logger = logging.getLogger(__name__)

# Fields to extract per document type
EXTRACTION_SCHEMAS: dict[str, list[str]] = {
    "account_opening": [
        "client_name", "account_type", "risk_tolerance", "investment_objective",
        "date_of_birth", "citizenship", "employment_status", "annual_income",
        "net_worth", "source_of_funds", "pep_status",
    ],
    "investment_policy_statement": [
        "client_name", "risk_tolerance", "investment_objective", "time_horizon",
        "liquidity_needs", "return_target", "constraints",
        "asset_allocation_target", "rebalancing_frequency", "benchmark",
    ],
    "quarterly_portfolio_review": [
        "total_portfolio_value", "asset_allocations", "performance_ytd",
        "performance_qtd", "benchmark_comparison", "risk_metrics",
        "strategy_changes", "fees_charged", "commentary",
    ],
    "compliance_disclosure": [
        "pep_flag", "sanctions_screening", "aml_status", "kyc_status",
        "suitability_confirmed", "risk_disclosure_acknowledged",
        "conflict_of_interest", "fee_disclosure", "adverse_media",
    ],
}

SYSTEM_PROMPT = """You are a financial document extraction system. Given a source document,
extract the requested fields into a structured JSON output.

CRITICAL CONTRACT: For every extracted field you MUST provide:
1. "value": the extracted value
2. "raw_quote": the EXACT verbatim text snippet from the source that supports this value
3. "char_span": [start, end] character offsets of the raw_quote in the source document

Also provide a "summary" field: a concise 2-4 sentence prose summary of the document
suitable for a relationship manager preparing for a client meeting.

Output format:
{
  "predicted_doc_type": "<doc_type>",
  "fields": [
    {"field_name": "<name>", "value": <value>, "raw_quote": "<exact text>", "char_span": [start, end]},
    ...
  ],
  "summary": "<prose summary>"
}"""


def _build_user_prompt(doc_type: str, text: str, field_names: list[str]) -> str:
    fields_str = ", ".join(field_names)
    return (
        f"Document type: {doc_type}\n"
        f"Fields to extract: {fields_str}\n\n"
        f"--- SOURCE DOCUMENT ---\n{text}\n--- END ---"
    )


def _find_span(text: str, quote: str) -> tuple[int, int]:
    """Find the character span of a quote in the source text. Fallback to fuzzy."""
    idx = text.find(quote)
    if idx >= 0:
        return (idx, idx + len(quote))
    # Try case-insensitive
    idx = text.lower().find(quote.lower())
    if idx >= 0:
        return (idx, idx + len(quote))
    # Try first 60 chars of quote
    if len(quote) > 60:
        short = quote[:60]
        idx = text.find(short)
        if idx >= 0:
            return (idx, idx + len(quote))
    return (0, 0)


def extract_document(
    client: NIMClient,
    model: str,
    doc_id: str,
    doc_type: str,
    text: str,
) -> ExtractionOutput:
    """Run the primary extraction on a single document."""
    field_names = EXTRACTION_SCHEMAS.get(doc_type, [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(doc_type, text, field_names)},
    ]

    parsed = client.chat_json(model=model, messages=messages)

    # Normalize fields and fix spans
    fields = []
    for f in parsed.get("fields", []):
        quote = f.get("raw_quote", "")
        span = f.get("char_span", [0, 0])

        # Validate / fix the span
        if isinstance(span, list) and len(span) == 2:
            start, end = span
            actual_text = text[start:end] if 0 <= start < end <= len(text) else ""
            if actual_text != quote:
                start, end = _find_span(text, quote)
            span_tuple = (start, end)
        else:
            span_tuple = _find_span(text, quote)

        fields.append(FieldExtraction(
            field_name=f.get("field_name", "unknown"),
            value=f.get("value"),
            raw_quote=quote,
            char_span=span_tuple,
        ))

    return ExtractionOutput(
        doc_id=doc_id,
        predicted_doc_type=DocType(doc_type),
        fields=fields,
        summary=parsed.get("summary", ""),
    )


def extract_document_offline(
    doc_id: str,
    doc_type: str,
    text: str,
    slots: Any,
) -> ExtractionOutput:
    """Deterministic extraction from template slots — no LLM call needed.

    Used when generating frozen fixtures without an API key.
    """
    if isinstance(slots, dict):
        slots = TemplateSlots.from_dict(slots)

    dt = DocType(doc_type)
    fields: list[FieldExtraction] = []

    def _add(name: str, value: Any, search_text: str | None = None):
        if search_text is None:
            search_text = str(value)
        idx = text.find(search_text)
        if idx < 0:
            # Try case variants
            for variant in [str(value).title(), str(value).upper(), str(value).lower()]:
                idx = text.find(variant)
                if idx >= 0:
                    search_text = variant
                    break
        if idx < 0:
            idx = 0
            search_text = str(value)[:50]

        span = (idx, idx + len(search_text)) if idx >= 0 and search_text else (0, 0)
        fields.append(FieldExtraction(
            field_name=name,
            value=value,
            raw_quote=text[span[0]:span[1]] if span != (0, 0) else search_text,
            char_span=span,
        ))

    if dt == DocType.ACCOUNT_OPENING:
        _add("client_name", slots.client_name)
        _add("account_type", slots.account_type)
        _add("risk_tolerance", slots.risk_tolerance, slots.risk_tolerance.replace("_", " ").title())
        _add("investment_objective", slots.investment_objective,
             slots.investment_objective.replace("_", " ").title())
        _add("date_of_birth", "1968-04-12")
        _add("citizenship", "United States")
        _add("employment_status", "Self-employed")
        _add("annual_income", round(slots.total_value * 0.08, 2),
             f"${slots.total_value * 0.08:,.2f}")
        _add("net_worth", round(slots.total_value * 1.5, 2),
             f"${slots.total_value * 1.5:,.2f}")
        _add("source_of_funds", "Business income and inheritance")
        _add("pep_status", False, "PEP Status: No")

    elif dt == DocType.INVESTMENT_POLICY_STATEMENT:
        _add("client_name", slots.client_name)
        _add("risk_tolerance", slots.risk_tolerance, slots.risk_tolerance.replace("_", " ").title())
        _add("investment_objective", slots.investment_objective,
             slots.investment_objective.replace("_", " "))
        _add("time_horizon", slots.time_horizon, f"{slots.time_horizon} years")
        if slots.liquidity_amount > 0:
            _add("liquidity_needs", {
                "amount": slots.liquidity_amount,
                "horizon_months": slots.liquidity_horizon_months,
            }, f"${slots.liquidity_amount:,.2f}")
        else:
            _add("liquidity_needs", "none", "long-term capital appreciation")
        _add("return_target", "capital appreciation",
             "long-term capital appreciation")
        _add("constraints", [
            "No tobacco or firearms",
            "Max 5% single issuer",
            "Investment grade fixed income only",
        ] + [c[:80] for c in slots.bespoke_clauses], "No investments in tobacco")
        _add("asset_allocation_target", {
            "equities": slots.equity_pct,
            "fixed_income": slots.fixed_income_pct,
            "cash": slots.cash_pct,
        }, f"Equities      | {slots.equity_pct}")
        _add("rebalancing_frequency", "quarterly", "quarterly")
        _add("benchmark", slots.benchmark, slots.benchmark)

    elif dt == DocType.QUARTERLY_PORTFOLIO_REVIEW:
        _add("total_portfolio_value", slots.total_value, f"${slots.total_value:,.2f}")
        eq_val = round(slots.total_value * slots.equity_pct / 100, 2)
        fi_val = round(slots.total_value * slots.fixed_income_pct / 100, 2)
        cash_val = round(slots.total_value * slots.cash_pct / 100, 2)
        _add("asset_allocations", {
            "equities": {"value": eq_val, "weight": slots.equity_pct},
            "fixed_income": {"value": fi_val, "weight": slots.fixed_income_pct},
            "cash": {"value": cash_val, "weight": slots.cash_pct},
        }, f"Equities      | ${eq_val:,.2f}")
        _add("performance_ytd", slots.performance_ytd, f"{slots.performance_ytd}%")
        _add("performance_qtd", slots.performance_qtd, f"{slots.performance_qtd}%")
        _add("benchmark_comparison", {
            "benchmark": slots.benchmark,
            "benchmark_return": slots.benchmark_return,
            "excess_return": round(slots.performance_ytd - slots.benchmark_return, 2),
        }, f"{slots.benchmark_return}%")
        risk_vol = round(8 + RISK_LEVELS_IDX.get(slots.risk_tolerance, 1) * 4, 1)
        _add("risk_metrics", {
            "volatility": risk_vol,
            "sharpe_ratio": round(0.5 + slots.performance_ytd / 20, 2),
            "max_drawdown": round(abs(min(slots.performance_qtd, 0)) + 2, 1),
        }, f"{risk_vol}%")
        _add("strategy_changes",
             "no changes" if abs(slots.performance_ytd - slots.benchmark_return) < 3
             else "rebalancing recommended",
             "No changes" if abs(slots.performance_ytd - slots.benchmark_return) < 3
             else "recommend")
        _add("fees_charged", round(slots.total_value * 0.0025, 2),
             f"${slots.total_value * 0.0025:,.2f}")
        _add("commentary",
             "outperformed" if slots.performance_ytd > slots.benchmark_return else "underperformed",
             "outperformed" if slots.performance_ytd > slots.benchmark_return else "underperformed")

    elif dt == DocType.COMPLIANCE_DISCLOSURE:
        is_pep = "PEP" in str(slots.bespoke_clauses)
        _add("pep_flag", is_pep, "is not" if not is_pep else "is")
        _add("sanctions_screening", "clear", "CLEAR")
        _add("aml_status", "passed", "PASSED" if "PASSED" in text else "completed")
        _add("kyc_status", "complete", "COMPLETE" if "COMPLETE" in text else "collected")
        _add("suitability_confirmed", True, "CONFIRMED" if "CONFIRMED" in text else "suitable")
        _add("risk_disclosure_acknowledged", True, "acknowledged")
        _add("conflict_of_interest", "none", "NONE IDENTIFIED" if "NONE" in text else "none")
        _add("fee_disclosure", "1.00% per annum", "1.00% per annum")
        _add("adverse_media", "clear", "CLEAR")

    # Build summary
    summary = _build_summary(dt, slots)

    return ExtractionOutput(
        doc_id=doc_id,
        predicted_doc_type=dt,
        fields=fields,
        summary=summary,
    )


RISK_LEVELS_IDX = {
    "conservative": 0, "moderate": 1, "moderately_aggressive": 2, "aggressive": 3,
}


def _build_summary(dt: DocType, slots: Any) -> str:
    name = slots.client_name
    risk = slots.risk_tolerance.replace("_", " ")

    if dt == DocType.ACCOUNT_OPENING:
        return (
            f"New {slots.account_type} account opened for {name}. "
            f"Client has a {risk} risk tolerance with an objective of "
            f"{slots.investment_objective.replace('_', ' ')}. "
            f"Source of funds is business income and inheritance. "
            f"PEP status: No."
        )
    elif dt == DocType.INVESTMENT_POLICY_STATEMENT:
        liq = ""
        if slots.liquidity_amount > 0:
            liq = (
                f" The client anticipates withdrawing ${slots.liquidity_amount:,.0f} "
                f"within {slots.liquidity_horizon_months} months."
            )
        return (
            f"IPS for {name} with {risk} risk tolerance and "
            f"{slots.time_horizon}-year time horizon. "
            f"Target allocation: {slots.equity_pct}% equities, "
            f"{slots.fixed_income_pct}% fixed income, {slots.cash_pct}% cash.{liq}"
        )
    elif dt == DocType.QUARTERLY_PORTFOLIO_REVIEW:
        perf = "outperformed" if slots.performance_ytd > slots.benchmark_return else "underperformed"
        return (
            f"Q1 2026 portfolio review for {name}. "
            f"Total value: ${slots.total_value:,.0f}. "
            f"YTD return: {slots.performance_ytd}% ({perf} benchmark by "
            f"{abs(round(slots.performance_ytd - slots.benchmark_return, 2))}pp). "
            f"Allocation: {slots.equity_pct}% equities, {slots.fixed_income_pct}% fixed income, "
            f"{slots.cash_pct}% cash."
        )
    else:  # compliance
        return (
            f"Compliance disclosure for {name}. "
            f"KYC complete, sanctions screening clear, AML verification passed. "
            f"Suitability confirmed for {slots.investment_objective.replace('_', ' ')} "
            f"strategy with {risk} risk tolerance."
        )
