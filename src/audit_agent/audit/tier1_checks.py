"""Tier-1 deterministic checks — no LLM calls, pure logic.

Each check: (source_text, extraction, claims) → list[CheckResult]
These are the cheapest, strongest signals.
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from audit_agent.config import doc_type_policies, scoring_weights
from audit_agent.schemas import (
    AuditTag,
    CheckResult,
    CheckStatus,
    Claim,
    ClaimSource,
    DocType,
    ExtractionOutput,
)


NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*\.?\d*")

# Words that look like proper names in template/doc context but are not client names
_DOCUMENT_TERM_WORDS = frozenset({
    "quarterly", "portfolio", "review", "investment", "policy", "statement",
    "account", "opening", "compliance", "disclosure", "advisory", "management",
    "performance", "financial", "client", "current", "risk", "following",
    "comprehensive", "strategic", "tactical", "emerging", "markets", "developed",
    "domestic", "international", "sharpe", "ratio", "adjusted", "benchmark",
})

_FULL_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")


def run_all_tier1(
    source_text: str,
    extraction: ExtractionOutput,
    claims: list[Claim],
) -> list[CheckResult]:
    """Run every Tier-1 check and return all results."""
    results: list[CheckResult] = []
    results.extend(check_required_fields(extraction))
    results.extend(check_footing(extraction))
    results.extend(check_numeric_ranges(extraction))
    results.extend(check_citation_grounding(source_text, claims))
    results.extend(check_quote_value_consistency(claims))
    results.extend(check_cross_field_consistency(extraction))
    results.extend(check_type_schema_match(source_text, extraction))
    results.extend(check_summary_extraction_consistency(source_text, extraction, claims))
    results.extend(check_summary_client_name_consistency(source_text, extraction, claims))
    return results


def check_required_fields(extraction: ExtractionOutput) -> list[CheckResult]:
    results = []
    policies = doc_type_policies()
    doc_policy = policies.get(extraction.predicted_doc_type.value, {})
    required = doc_policy.get("required_fields", [])
    present = {f.field_name for f in extraction.fields}

    for field_name in required:
        claim_id = f"{extraction.doc_id}_field_{field_name}"
        if field_name in present:
            results.append(CheckResult(
                check_name="required_fields",
                claim_id=claim_id,
                status=CheckStatus.PASS,
                reason=f"Required field '{field_name}' is present",
            ))
        else:
            results.append(CheckResult(
                check_name="required_fields",
                claim_id=claim_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.MISSING_REQUIRED_FIELD,
                reason=f"Required field '{field_name}' is MISSING",
            ))
    return results


def check_footing(extraction: ExtractionOutput) -> list[CheckResult]:
    results = []
    weights = scoring_weights()
    tol = weights.get("routing", {}).get("footing_tolerance_pct", 2.0)

    # Percentage footing: check allocation weights sum to ~100
    alloc_field = None
    total_field = None
    for f in extraction.fields:
        if f.field_name in ("asset_allocations", "asset_allocation_target"):
            alloc_field = f
        if f.field_name == "total_portfolio_value":
            total_field = f

    if alloc_field and isinstance(alloc_field.value, dict):
        pct_sum = _sum_allocation_weights(alloc_field.value)
        if pct_sum is not None:
            claim_id = f"{extraction.doc_id}_field_{alloc_field.field_name}"
            if abs(pct_sum - 100.0) <= tol:
                results.append(CheckResult(
                    check_name="footing_pct",
                    claim_id=claim_id,
                    status=CheckStatus.PASS,
                    reason=f"Allocation weights sum to {pct_sum:.1f}% (within {tol}% tolerance)",
                ))
            else:
                results.append(CheckResult(
                    check_name="footing_pct",
                    claim_id=claim_id,
                    status=CheckStatus.FAIL,
                    tag=AuditTag.FOOTING_ERROR,
                    reason=f"Allocation weights sum to {pct_sum:.1f}%, expected ~100% (±{tol}%)",
                ))

    # Dollar footing: total_portfolio_value vs sum of dollar allocations
    if alloc_field and total_field and isinstance(alloc_field.value, dict):
        dollar_sum = _sum_allocation_values(alloc_field.value)
        if dollar_sum is not None and isinstance(total_field.value, (int, float)):
            stated = float(total_field.value)
            if stated > 0:
                pct_diff = abs(dollar_sum - stated) / stated * 100
                claim_id = f"{extraction.doc_id}_field_total_portfolio_value"
                if pct_diff <= tol:
                    results.append(CheckResult(
                        check_name="footing_dollar",
                        claim_id=claim_id,
                        status=CheckStatus.PASS,
                        reason=f"Dollar sum ${dollar_sum:,.0f} matches total ${stated:,.0f}",
                    ))
                else:
                    results.append(CheckResult(
                        check_name="footing_dollar",
                        claim_id=claim_id,
                        status=CheckStatus.FAIL,
                        tag=AuditTag.FOOTING_ERROR,
                        reason=(
                            f"Dollar sum ${dollar_sum:,.0f} does NOT match "
                            f"stated total ${stated:,.0f} (diff {pct_diff:.1f}%)"
                        ),
                    ))
    return results


def _sum_allocation_weights(alloc: dict) -> float | None:
    """Sum percentage weights from an allocation dict."""
    total = 0.0
    for v in alloc.values():
        if isinstance(v, (int, float)):
            total += v
        elif isinstance(v, dict):
            w = v.get("weight") or v.get("pct") or v.get("percentage")
            if isinstance(w, (int, float)):
                total += w
            else:
                return None
        else:
            return None
    return round(total, 2)


def _sum_allocation_values(alloc: dict) -> float | None:
    """Sum dollar values from an allocation dict."""
    total = 0.0
    found_any = False
    for v in alloc.values():
        if isinstance(v, dict):
            val = v.get("value") or v.get("amount") or v.get("dollar_value")
            if isinstance(val, (int, float)):
                total += val
                found_any = True
    return round(total, 2) if found_any else None


def check_numeric_ranges(extraction: ExtractionOutput) -> list[CheckResult]:
    results = []
    for f in extraction.fields:
        if not isinstance(f.value, (int, float)):
            continue
        claim_id = f"{extraction.doc_id}_field_{f.field_name}"

        # Portfolio values should be positive
        if "value" in f.field_name or "income" in f.field_name or "worth" in f.field_name:
            if f.value < 0:
                results.append(CheckResult(
                    check_name="numeric_range",
                    claim_id=claim_id,
                    status=CheckStatus.FAIL,
                    tag=AuditTag.NUMERIC_RANGE_VIOLATION,
                    reason=f"{f.field_name} = {f.value} is negative",
                ))

        # Percentages should be in [-100, 200] (allowing for leveraged returns)
        if "pct" in f.field_name or "ytd" in f.field_name or "qtd" in f.field_name:
            if not (-100 <= f.value <= 200):
                results.append(CheckResult(
                    check_name="numeric_range",
                    claim_id=claim_id,
                    status=CheckStatus.FAIL,
                    tag=AuditTag.NUMERIC_RANGE_VIOLATION,
                    reason=f"{f.field_name} = {f.value} outside plausible range",
                ))

        # Time horizon should be 1-50 years
        if f.field_name == "time_horizon":
            if not (1 <= f.value <= 50):
                results.append(CheckResult(
                    check_name="numeric_range",
                    claim_id=claim_id,
                    status=CheckStatus.FAIL,
                    tag=AuditTag.NUMERIC_RANGE_VIOLATION,
                    reason=f"Time horizon of {f.value} years is implausible",
                ))
    return results


def check_citation_grounding(
    source_text: str,
    claims: list[Claim],
) -> list[CheckResult]:
    results = []
    weights = scoring_weights()
    min_score = weights.get("routing", {}).get("citation_min_score", 75)

    for claim in claims:
        if claim.source != ClaimSource.STRUCTURED_FIELD:
            continue  # summary claims don't have citations
        if not claim.raw_quote:
            results.append(CheckResult(
                check_name="citation_grounding",
                claim_id=claim.claim_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.CITATION_NOT_FOUND,
                reason=f"No raw_quote provided for '{claim.field_name}'",
            ))
            continue

        # Check if the quote exists at the claimed span
        span_match = False
        if claim.char_span and claim.char_span != (0, 0):
            start, end = claim.char_span
            if 0 <= start < end <= len(source_text):
                actual = source_text[start:end]
                span_score = fuzz.ratio(claim.raw_quote, actual)
                span_match = span_score >= min_score

        # Fuzzy match anywhere in source
        if not span_match:
            best_score = fuzz.partial_ratio(claim.raw_quote, source_text)
        else:
            best_score = 100  # span matched

        if best_score >= min_score:
            results.append(CheckResult(
                check_name="citation_grounding",
                claim_id=claim.claim_id,
                status=CheckStatus.PASS,
                reason=f"Citation grounded (score={best_score})",
            ))
        else:
            results.append(CheckResult(
                check_name="citation_grounding",
                claim_id=claim.claim_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.CITATION_NOT_FOUND,
                reason=f"Citation NOT grounded: '{claim.raw_quote[:60]}...' (best_score={best_score})",
            ))
    return results


def check_quote_value_consistency(claims: list[Claim]) -> list[CheckResult]:
    results = []
    for claim in claims:
        if claim.source != ClaimSource.STRUCTURED_FIELD:
            continue
        if not isinstance(claim.value, (int, float)) or not claim.raw_quote:
            continue

        quote_numbers = _extract_numeric_literals(claim.raw_quote)
        if not quote_numbers:
            continue

        value = float(claim.value)
        if any(_numbers_close(value, number, rel_tol=_quote_value_tolerance(value)) for number in quote_numbers):
            results.append(CheckResult(
                check_name="quote_value_consistency",
                claim_id=claim.claim_id,
                status=CheckStatus.PASS,
                reason="Extracted numeric value matches the supporting quote",
                evidence_span=claim.raw_quote,
            ))
            continue

        quoted = ", ".join(_format_numeric_literal(number) for number in quote_numbers[:3])
        results.append(CheckResult(
            check_name="quote_value_consistency",
            claim_id=claim.claim_id,
            status=CheckStatus.FAIL,
            tag=AuditTag.QUOTE_VALUE_MISMATCH,
            reason=(
                f"Extracted value {claim.value} does not match numeric evidence in quote "
                f"({quoted})"
            ),
            evidence_span=claim.raw_quote,
        ))
    return results


def check_cross_field_consistency(extraction: ExtractionOutput) -> list[CheckResult]:
    results = []
    fields = {f.field_name: f.value for f in extraction.fields}

    # Risk vs allocation consistency
    risk = str(fields.get("risk_tolerance", "")).lower()
    alloc = fields.get("asset_allocations") or fields.get("asset_allocation_target")

    if risk and alloc and isinstance(alloc, dict):
        equity_pct = _get_equity_pct(alloc)
        fi_pct = _get_fi_pct(alloc)

        if risk == "conservative" and equity_pct is not None and equity_pct > 60:
            results.append(CheckResult(
                check_name="cross_field",
                claim_id=f"{extraction.doc_id}_field_risk_tolerance",
                status=CheckStatus.FAIL,
                tag=AuditTag.CROSS_FIELD_VIOLATION,
                reason=(
                    f"Conservative risk tolerance with {equity_pct}% equity allocation "
                    f"is inconsistent"
                ),
            ))

        if risk == "aggressive" and fi_pct is not None and fi_pct > 70:
            results.append(CheckResult(
                check_name="cross_field",
                claim_id=f"{extraction.doc_id}_field_risk_tolerance",
                status=CheckStatus.FAIL,
                tag=AuditTag.CROSS_FIELD_VIOLATION,
                reason=(
                    f"Aggressive risk tolerance with {fi_pct}% fixed income "
                    f"is inconsistent"
                ),
            ))

    # Time horizon vs risk tolerance
    horizon = fields.get("time_horizon")
    if risk and horizon is not None and isinstance(horizon, (int, float)):
        if risk == "aggressive" and horizon < 3:
            results.append(CheckResult(
                check_name="cross_field",
                claim_id=f"{extraction.doc_id}_field_time_horizon",
                status=CheckStatus.FAIL,
                tag=AuditTag.CROSS_FIELD_VIOLATION,
                reason=(
                    f"Aggressive risk tolerance with a {horizon}-year horizon is inconsistent "
                    f"(short horizons require capital preservation, not aggressive growth)"
                ),
            ))
        elif risk == "conservative" and horizon > 20:
            results.append(CheckResult(
                check_name="cross_field",
                claim_id=f"{extraction.doc_id}_field_time_horizon",
                status=CheckStatus.FAIL,
                tag=AuditTag.CROSS_FIELD_VIOLATION,
                reason=(
                    f"Conservative risk tolerance with a {horizon}-year horizon is unusual "
                    f"(long horizons typically support at least moderate growth exposure)"
                ),
            ))

    return results


def _get_equity_pct(alloc: dict) -> float | None:
    eq = alloc.get("equities") or alloc.get("equity")
    if isinstance(eq, (int, float)):
        return eq
    if isinstance(eq, dict):
        return eq.get("weight") or eq.get("pct")
    return None


def _get_fi_pct(alloc: dict) -> float | None:
    fi = alloc.get("fixed_income") or alloc.get("bonds")
    if isinstance(fi, (int, float)):
        return fi
    if isinstance(fi, dict):
        return fi.get("weight") or fi.get("pct")
    return None


def check_type_schema_match(
    source_text: str,
    extraction: ExtractionOutput,
) -> list[CheckResult]:
    results = []
    dt = extraction.predicted_doc_type
    text_lower = source_text.lower()

    # Simple keyword signals
    type_signals: dict[DocType, list[str]] = {
        DocType.ACCOUNT_OPENING: ["account opening", "date of birth", "employment status"],
        DocType.INVESTMENT_POLICY_STATEMENT: [
            "investment policy", "time horizon", "target asset allocation",
        ],
        DocType.QUARTERLY_PORTFOLIO_REVIEW: [
            "quarterly", "portfolio review", "qtd return", "ytd return",
        ],
        DocType.COMPLIANCE_DISCLOSURE: [
            "compliance", "kyc", "aml", "sanctions screening",
        ],
    }

    claimed_signals = type_signals.get(dt, [])
    hits = sum(1 for s in claimed_signals if s in text_lower)
    claim_id = f"{extraction.doc_id}_field_predicted_doc_type"

    if hits >= 2:
        results.append(CheckResult(
            check_name="type_schema_match",
            claim_id=claim_id,
            status=CheckStatus.PASS,
            reason=f"Doc type '{dt.value}' supported by {hits}/{len(claimed_signals)} signals",
        ))
    elif hits == 1:
        results.append(CheckResult(
            check_name="type_schema_match",
            claim_id=claim_id,
            status=CheckStatus.PASS,
            reason=f"Doc type '{dt.value}' weakly supported ({hits} signal)",
        ))
    else:
        # Check if a different type matches better
        best_type = None
        best_hits = 0
        for candidate_type, sigs in type_signals.items():
            c_hits = sum(1 for s in sigs if s in text_lower)
            if c_hits > best_hits:
                best_hits = c_hits
                best_type = candidate_type

        results.append(CheckResult(
            check_name="type_schema_match",
            claim_id=claim_id,
            status=CheckStatus.FAIL,
            tag=AuditTag.TYPE_MISMATCH,
            reason=(
                f"Doc type '{dt.value}' has 0 keyword signals. "
                f"Better match: '{best_type.value}' ({best_hits} hits)"
                if best_type else f"Doc type '{dt.value}' has 0 keyword signals"
            ),
        ))
    return results


def check_summary_extraction_consistency(
    source_text: str,
    extraction: ExtractionOutput,
    claims: list[Claim],
) -> list[CheckResult]:
    """Flag summary claims that go beyond the extracted or cited evidence.

    Uses simple heuristic: numbers in summary should appear in structured fields.
    """
    results = []
    summary = extraction.summary
    if not summary:
        return results

    # Extract numbers from summary
    summary_numbers = _extract_numeric_literals(summary)

    # Extract numbers from structured fields
    field_numbers: list[float] = []
    for f in extraction.fields:
        _extract_numbers_from_value(f.value, field_numbers)

    source_numbers = _extract_numeric_literals(source_text)

    # Numbers in summary but not in any field → possible overreach
    summary_claims = [c for c in claims if c.source == ClaimSource.SUMMARY]

    for num in summary_numbers:
        if num == 0 or abs(num) in (0, 1, 2):
            continue  # trivial numbers

        found = any(_numbers_close(num, field_num) for field_num in field_numbers)
        if not found:
            found = any(_numbers_close(num, source_num, rel_tol=0.001) for source_num in source_numbers)
        if not found and len(summary_claims) > 0:
            # Attach to first summary claim
            results.append(CheckResult(
                check_name="summary_consistency",
                claim_id=summary_claims[0].claim_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.SUMMARY_OVERREACH,
                reason=(
                    f"Summary mentions number '{_format_numeric_literal(num)}' not found in "
                    f"structured fields — possible overreach"
                ),
            ))

    # Length-gated zero-grounding heuristic: financial documents with long, number-free
    # summaries are a signal of schema-valid garbage (plausible text with no specific evidence).
    _NUMERIC_HEAVY_DOCTYPES = {
        DocType.QUARTERLY_PORTFOLIO_REVIEW,
        DocType.INVESTMENT_POLICY_STATEMENT,
    }
    summary_word_count = len(summary.split())
    if (
        extraction.predicted_doc_type in _NUMERIC_HEAVY_DOCTYPES
        and summary_word_count > 30
        and not summary_numbers
        and summary_claims
    ):
        results.append(CheckResult(
            check_name="summary_consistency",
            claim_id=summary_claims[0].claim_id,
            status=CheckStatus.FAIL,
            tag=AuditTag.SUMMARY_OVERREACH,
            reason=(
                f"Summary is {summary_word_count} words but contains no specific numeric values "
                f"for a {extraction.predicted_doc_type.value} — may be generic or unsupported"
            ),
        ))

    return results


def check_summary_client_name_consistency(
    source_text: str,
    extraction: ExtractionOutput,
    claims: list[Claim],
) -> list[CheckResult]:
    """Detect stale-template client names that leak into the summary.

    Catches stale-template-bleed where a prior client's name leaks into the summary.
    Uses two paths:
    - Path 1: compare against extracted client_name field (fast, precise)
    - Path 2: if no client_name field, check that summary names appear in source document
    """
    results = []

    summary = extraction.summary
    if not summary:
        return results

    # Find candidate person names in summary
    summary_name_pairs = [
        (first, last)
        for first, last in _FULL_NAME_PATTERN.findall(summary)
        if first.lower() not in _DOCUMENT_TERM_WORDS
        and last.lower() not in _DOCUMENT_TERM_WORDS
    ]
    if not summary_name_pairs:
        return results

    summary_claims = [c for c in claims if c.source == ClaimSource.SUMMARY]
    target_id = (
        summary_claims[0].claim_id
        if summary_claims
        else f"{extraction.doc_id}_field_client_name"
    )

    # Path 1: client_name field is available — compare directly
    client_name_field = next(
        (f for f in extraction.fields if f.field_name == "client_name"), None
    )
    if client_name_field and isinstance(client_name_field.value, str):
        client_tokens = set(client_name_field.value.lower().split())
        for first, last in summary_name_pairs:
            if not {first.lower(), last.lower()} & client_tokens:
                results.append(CheckResult(
                    check_name="summary_client_name",
                    claim_id=target_id,
                    status=CheckStatus.FAIL,
                    tag=AuditTag.SUMMARY_CLIENT_MISMATCH,
                    reason=(
                        f"Summary references '{first} {last}' but extracted client_name is "
                        f"'{client_name_field.value}' — possible stale template bleed"
                    ),
                ))
                break
        return results

    # Path 2: no client_name field — verify each summary name appears in source document
    for first, last in summary_name_pairs:
        full_name = f"{first} {last}"
        if full_name not in source_text:
            results.append(CheckResult(
                check_name="summary_client_name",
                claim_id=target_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.SUMMARY_CLIENT_MISMATCH,
                reason=(
                    f"Summary references '{full_name}' which does not appear in the source "
                    f"document — possible stale template bleed"
                ),
            ))
            break

    return results


def _extract_numbers_from_value(value: Any, acc: list[float]) -> None:
    if isinstance(value, (int, float)):
        acc.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            _extract_numbers_from_value(v, acc)
    elif isinstance(value, list):
        for v in value:
            _extract_numbers_from_value(v, acc)
    elif isinstance(value, str):
        acc.extend(_extract_numeric_literals(value))


def _extract_numeric_literals(text: str) -> list[float]:
    numbers = []
    for token in NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return numbers


def _format_numeric_literal(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _quote_value_tolerance(value: float) -> float:
    magnitude = abs(value)
    if magnitude >= 1000:
        return 0.005
    return 0.02


def _numbers_close(a: float, b: float, rel_tol: float = 0.05) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) / max(abs(a), abs(b)) <= rel_tol
