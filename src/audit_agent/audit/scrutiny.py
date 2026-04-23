"""Scrutiny scorer — computes per-document complexity and audit priority.

Uses three vectors (structural density, regulatory density, financial-network
density) plus doc-type prior and stakes flags.
"""

from __future__ import annotations

import re

from audit_agent.config import scoring_weights
from audit_agent.schemas import (
    CheckResult,
    CheckStatus,
    Claim,
    Criticality,
    ExtractionOutput,
    ScrutinyLevel,
)


REGULATORY_TERMS = re.compile(
    r"(lock[\s-]?up\s+period|fiduciary|aml|kyc|anti[\s-]?money|know\s+your\s+customer|"
    r"sanctions|compliance|notwithstanding|except\s+as|custom\s+restriction|"
    r"politically\s+exposed|enhanced\s+due\s+diligence|suitability|"
    r"risk\s+disclosure|conflict\s+of\s+interest|regulatory)",
    re.IGNORECASE,
)

FINANCIAL_NETWORK_TERMS = re.compile(
    r"(special\s+purpose\s+vehicle|spv|sub[\s-]?account|subsidiary|"
    r"derivative|structured\s+product|swap|option|futures|"
    r"multi[\s-]?asset|co[\s-]?investment|private\s+equity|hedge\s+fund|"
    r"trust|estate|joint\s+account|household)",
    re.IGNORECASE,
)

TABLE_PATTERN = re.compile(r"\|.*\|.*\|", re.MULTILINE)


def compute_scrutiny(
    source_text: str,
    extraction: ExtractionOutput,
    claims: list[Claim],
    tier1_results: list[CheckResult] | None = None,
) -> ScrutinyLevel:
    cfg = scoring_weights()
    scr = cfg.get("scrutiny", {})
    weights = scr.get("weights", {})
    thresholds = scr.get("thresholds", {})
    type_scores = scr.get("doc_type_scores", {})

    # 1. Doc-type prior
    doc_type_score = type_scores.get(extraction.predicted_doc_type.value, 0.5)

    # 2. Structural density: ratio of table lines to total lines
    lines = source_text.split("\n")
    n_lines = max(len(lines), 1)
    table_lines = len(TABLE_PATTERN.findall(source_text))
    structural_density = min(table_lines / n_lines * 3, 1.0)  # scale up, cap at 1

    # Also factor in document length
    token_estimate = len(source_text.split())
    length_factor = min(token_estimate / 1500, 1.0)  # long docs → more complex
    structural_density = (structural_density + length_factor) / 2

    # 3. Regulatory density: compliance term hit rate
    reg_hits = len(REGULATORY_TERMS.findall(source_text))
    regulatory_density = min(reg_hits / 15, 1.0)  # normalize to [0, 1]

    # 4. Financial-network density: entity/structure complexity
    net_hits = len(FINANCIAL_NETWORK_TERMS.findall(source_text))
    financial_density = min(net_hits / 10, 1.0)

    # Weighted combination
    raw_score = (
        weights.get("doc_type_prior", 0.3) * doc_type_score
        + weights.get("structural_density", 0.25) * structural_density
        + weights.get("regulatory_density", 0.25) * regulatory_density
        + weights.get("financial_network_density", 0.2) * financial_density
    )

    # 5. Stakes flag: any critical-criticality claim lifts to >= MEDIUM
    has_critical = any(c.criticality == Criticality.CRITICAL for c in claims)

    # 6. Tier-1 escalation: any Tier-1 failure bumps one level
    has_tier1_fail = False
    if tier1_results:
        has_tier1_fail = any(r.status == CheckStatus.FAIL for r in tier1_results)

    # Determine level from thresholds
    low_t = thresholds.get("low", 0.3)
    med_t = thresholds.get("medium", 0.5)
    high_t = thresholds.get("high", 0.7)

    if raw_score <= low_t:
        level = ScrutinyLevel.LOW
    elif raw_score <= med_t:
        level = ScrutinyLevel.MEDIUM
    elif raw_score <= high_t:
        level = ScrutinyLevel.HIGH
    else:
        level = ScrutinyLevel.CRITICAL

    # Apply escalations
    if has_critical and level == ScrutinyLevel.LOW:
        level = ScrutinyLevel.MEDIUM
    if has_tier1_fail:
        level = _bump(level)

    return level


def _bump(level: ScrutinyLevel) -> ScrutinyLevel:
    order = [ScrutinyLevel.LOW, ScrutinyLevel.MEDIUM, ScrutinyLevel.HIGH, ScrutinyLevel.CRITICAL]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]
