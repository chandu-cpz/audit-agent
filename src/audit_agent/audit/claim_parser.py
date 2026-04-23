"""Claim parser — decomposes the primary LLM's output into atomic auditable claims.

Field claims come straight from the structured extraction (already atomic).
Summary claims are extracted from the prose summary by splitting on sentence
boundaries — no LLM call required for the prototype since summaries are short.
"""

from __future__ import annotations

import re

from audit_agent.config import field_criticality
from audit_agent.schemas import (
    Claim,
    ClaimSource,
    Criticality,
    ExtractionOutput,
    FieldExtraction,
)


def _get_criticality(doc_type: str, field_name: str) -> Criticality:
    """Look up criticality from configs/field_criticality.yaml."""
    fc = field_criticality()
    doc_fields = fc.get(doc_type, {})
    level = doc_fields.get(field_name, "medium")
    return Criticality(level)


def _field_to_claim(field: FieldExtraction, doc_id: str, doc_type: str) -> Claim:
    """Convert a structured FieldExtraction to an auditable Claim."""
    value_str = str(field.value)
    if isinstance(field.value, dict):
        value_str = ", ".join(f"{k}={v}" for k, v in field.value.items())
    elif isinstance(field.value, list):
        value_str = "; ".join(str(v) for v in field.value)

    return Claim(
        claim_id=f"{doc_id}_field_{field.field_name}",
        doc_id=doc_id,
        source=ClaimSource.STRUCTURED_FIELD,
        field_name=field.field_name,
        statement=f"{field.field_name} = {value_str}",
        value=field.value,
        raw_quote=field.raw_quote,
        char_span=field.char_span,
        criticality=_get_criticality(doc_type, field.field_name),
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out trivial ones."""
    # Split on period, exclamation, question mark followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]


def _summary_to_claims(summary: str, doc_id: str, doc_type: str) -> list[Claim]:
    """Decompose a prose summary into atomic sentence-level claims."""
    sentences = _split_sentences(summary)
    claims = []
    for i, sentence in enumerate(sentences):
        claims.append(Claim(
            claim_id=f"{doc_id}_summary_{i}",
            doc_id=doc_id,
            source=ClaimSource.SUMMARY,
            field_name=None,
            statement=sentence,
            value=None,
            raw_quote=None,
            char_span=None,
            # Summary claims default to medium; upgraded by scrutiny if doc is complex
            criticality=Criticality.MEDIUM,
        ))
    return claims


def parse_claims(extraction: ExtractionOutput) -> list[Claim]:
    """Parse all claims from an extraction output (fields + summary)."""
    doc_type = extraction.predicted_doc_type.value
    claims: list[Claim] = []

    # Field claims
    for field in extraction.fields:
        claims.append(_field_to_claim(field, extraction.doc_id, doc_type))

    # Summary claims
    if extraction.summary:
        claims.extend(_summary_to_claims(
            extraction.summary, extraction.doc_id, doc_type
        ))

    return claims
