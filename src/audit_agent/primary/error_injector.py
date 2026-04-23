"""Error injector — mutates primary extraction outputs to simulate LLM failures.

Each error type maps to a brief-cited or real-world LLM pathology.
Ground truth labels are written to sidecar JSON so the audit pipeline
never sees them — only evaluate.py does.
"""

from __future__ import annotations

import random

from audit_agent.schemas import (
    DocType,
    ErrorLabel,
    ExtractionOutput,
    FieldExtraction,
    GroundTruth,
)

ERROR_TYPES = [
    "hallucinated_allocation",
    "numeric_drift",
    "logical_footing",
    "temporal_misattribution",
    "dropped_compliance_flag",
    "stale_template_bleed",
    "doc_misclassification",
    "schema_valid_garbage",
]

# Per-error-type injection probabilities (within the corrupted subset)
ERROR_WEIGHTS = {
    "hallucinated_allocation": 0.15,
    "numeric_drift": 0.15,
    "logical_footing": 0.12,
    "temporal_misattribution": 0.12,
    "dropped_compliance_flag": 0.15,
    "stale_template_bleed": 0.08,
    "doc_misclassification": 0.08,
    "schema_valid_garbage": 0.15,
}


def inject_errors(
    extraction: ExtractionOutput,
    source_text: str,
    rng: random.Random,
    corruption_rate: float = 0.30,
    max_errors_per_doc: int = 2,
) -> tuple[ExtractionOutput, GroundTruth]:
    """Maybe inject errors into an extraction. Returns (mutated_extraction, ground_truth)."""

    if rng.random() > corruption_rate:
        return extraction, GroundTruth(
            doc_id=extraction.doc_id,
            is_corrupted=False,
        )

    mutated = extraction.model_copy(deep=True)
    n_errors = rng.randint(1, max_errors_per_doc)

    # Choose error types weighted by probability
    types = list(ERROR_WEIGHTS.keys())
    weights = list(ERROR_WEIGHTS.values())
    chosen = rng.choices(types, weights, k=n_errors)
    # Deduplicate
    chosen = list(dict.fromkeys(chosen))

    labels: list[ErrorLabel] = []
    for error_type in chosen:
        label = _apply_error(mutated, source_text, error_type, rng)
        if label:
            labels.append(label)

    is_corrupted = len(labels) > 0
    return mutated, GroundTruth(
        doc_id=extraction.doc_id,
        is_corrupted=is_corrupted,
        injected_errors=labels,
    )


def _find_field(extraction: ExtractionOutput, name: str) -> FieldExtraction | None:
    for f in extraction.fields:
        if f.field_name == name:
            return f
    return None


def _apply_error(
    ext: ExtractionOutput,
    source: str,
    error_type: str,
    rng: random.Random,
) -> ErrorLabel | None:

    if error_type == "hallucinated_allocation":
        return _hallucinated_allocation(ext, rng)
    elif error_type == "numeric_drift":
        return _numeric_drift(ext, rng)
    elif error_type == "logical_footing":
        return _logical_footing(ext, rng)
    elif error_type == "temporal_misattribution":
        return _temporal_misattribution(ext, rng)
    elif error_type == "dropped_compliance_flag":
        return _dropped_compliance_flag(ext, rng)
    elif error_type == "stale_template_bleed":
        return _stale_template_bleed(ext, rng)
    elif error_type == "doc_misclassification":
        return _doc_misclassification(ext, rng)
    elif error_type == "schema_valid_garbage":
        return _schema_valid_garbage(ext, rng)
    return None


def _hallucinated_allocation(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Invent a portfolio allocation not in the source."""
    alloc = _find_field(ext, "asset_allocations") or _find_field(ext, "asset_allocation_target")
    if not alloc or not isinstance(alloc.value, dict):
        return None

    # Add a hallucinated asset class
    hallucinated_classes = ["alternatives", "commodities", "real_estate", "crypto", "hedge_funds"]
    new_class = rng.choice(hallucinated_classes)
    new_pct = round(rng.uniform(3, 15), 1)

    if isinstance(alloc.value, dict):
        alloc.value[new_class] = {"value": 0, "weight": new_pct} if "value" in str(alloc.value) else new_pct
        # Corrupt the raw_quote to include the hallucinated class
        alloc.raw_quote = alloc.raw_quote + f" | {new_class.title()} {new_pct}%"

    return ErrorLabel(
        error_type="hallucinated_allocation",
        affected_fields=["asset_allocations"],
        description=f"Hallucinated {new_class} allocation of {new_pct}%",
    )


def _numeric_drift(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Small (<5%) tweak to a numeric value."""
    numeric_fields = [
        "total_portfolio_value", "performance_ytd", "performance_qtd",
        "annual_income", "net_worth", "fees_charged",
    ]
    candidates = [f for f in ext.fields if f.field_name in numeric_fields]
    if not candidates:
        return None

    target = rng.choice(candidates)
    if isinstance(target.value, (int, float)):
        drift = rng.uniform(0.02, 0.05) * rng.choice([-1, 1])
        original = target.value
        target.value = round(target.value * (1 + drift), 2)
        return ErrorLabel(
            error_type="numeric_drift",
            affected_fields=[target.field_name],
            description=f"Drifted {target.field_name} from {original} to {target.value}",
        )
    return None


def _logical_footing(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Break the footing: sub-allocations no longer sum to total."""
    total_field = _find_field(ext, "total_portfolio_value")
    alloc_field = _find_field(ext, "asset_allocations")

    if not total_field or not alloc_field:
        return None

    if isinstance(total_field.value, (int, float)):
        # Shift total to 85% of actual sum
        total_field.value = round(total_field.value * rng.uniform(0.80, 0.92), 2)
        return ErrorLabel(
            error_type="logical_footing",
            affected_fields=["total_portfolio_value", "asset_allocations"],
            description="Total portfolio value no longer matches sum of allocations",
        )
    return None


def _temporal_misattribution(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Swap time horizon or misattribute a temporal constraint."""
    horizon = _find_field(ext, "time_horizon")
    risk = _find_field(ext, "risk_tolerance")

    if horizon and isinstance(horizon.value, (int, float)):
        original = horizon.value
        horizon.value = int(horizon.value) + rng.choice([3, 5, -3, -5])
        horizon.value = max(1, horizon.value)
        return ErrorLabel(
            error_type="temporal_misattribution",
            affected_fields=["time_horizon"],
            description=f"Time horizon changed from {original} to {horizon.value} years",
        )

    if risk:
        levels = ["conservative", "moderate", "moderately_aggressive", "aggressive"]
        original = risk.value
        new_risk = rng.choice([level for level in levels if level != original])
        risk.value = new_risk
        return ErrorLabel(
            error_type="temporal_misattribution",
            affected_fields=["risk_tolerance"],
            description=f"Risk tolerance misattributed from {original} to {new_risk}",
        )
    return None


def _dropped_compliance_flag(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Silently omit a compliance field."""
    compliance_fields = [
        "pep_flag", "sanctions_screening", "aml_status", "kyc_status",
        "suitability_confirmed", "adverse_media",
    ]
    candidates = [f for f in ext.fields if f.field_name in compliance_fields]
    if not candidates:
        return None

    target = rng.choice(candidates)
    ext.fields = [f for f in ext.fields if f.field_name != target.field_name]
    return ErrorLabel(
        error_type="dropped_compliance_flag",
        affected_fields=[target.field_name],
        description=f"Silently dropped {target.field_name}",
    )


def _stale_template_bleed(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Content from a different client bleeds into the summary."""
    fake_names = ["Alexander Thornberry", "Yuki Tanaka", "Lars Johansson"]
    fake_name = rng.choice(fake_names)
    original_summary = ext.summary
    ext.summary = ext.summary.replace(
        ext.summary.split(".")[0],
        f"Portfolio review for {fake_name} shows strong performance",
        1,
    )
    if ext.summary == original_summary:
        ext.summary = f"Based on {fake_name}'s investment preferences, " + ext.summary

    return ErrorLabel(
        error_type="stale_template_bleed",
        affected_fields=["summary"],
        description=f"Bled in reference to {fake_name} from another document",
    )


TYPE_SWAPS = {
    DocType.ACCOUNT_OPENING: DocType.COMPLIANCE_DISCLOSURE,
    DocType.INVESTMENT_POLICY_STATEMENT: DocType.QUARTERLY_PORTFOLIO_REVIEW,
    DocType.QUARTERLY_PORTFOLIO_REVIEW: DocType.INVESTMENT_POLICY_STATEMENT,
    DocType.COMPLIANCE_DISCLOSURE: DocType.ACCOUNT_OPENING,
}


def _doc_misclassification(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Classify the document as the wrong type."""
    original = ext.predicted_doc_type
    ext.predicted_doc_type = TYPE_SWAPS.get(original, DocType.ACCOUNT_OPENING)
    return ErrorLabel(
        error_type="doc_misclassification",
        affected_fields=["predicted_doc_type"],
        description=f"Misclassified from {original.value} to {ext.predicted_doc_type.value}",
    )


_GARBAGE_SUMMARIES = [
    "The client's portfolio demonstrates excellent risk-adjusted returns with a "
    "Sharpe ratio exceeding industry benchmarks across all time periods analyzed. "
    "The diversification strategy has effectively minimized drawdown risk while "
    "capturing upside potential in both domestic and international markets.",
    "Following a comprehensive review of the client's financial situation, we "
    "recommend maintaining the current strategic allocation with a tactical "
    "overweight to emerging market equities given the favorable macroeconomic "
    "backdrop and attractive valuations relative to developed markets.",
    "The client has expressed satisfaction with portfolio performance and has "
    "confirmed that their investment objectives, risk tolerance, and financial "
    "circumstances remain unchanged since the last review period.",
]


def _schema_valid_garbage(ext: ExtractionOutput, rng: random.Random) -> ErrorLabel | None:
    """Replace summary with schema-valid but narratively wrong text."""
    ext.summary = rng.choice(_GARBAGE_SUMMARIES)
    return ErrorLabel(
        error_type="schema_valid_garbage",
        affected_fields=["summary"],
        description="Summary replaced with plausible but unsupported text",
    )
