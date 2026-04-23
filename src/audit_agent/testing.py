from __future__ import annotations

from typing import Any

from audit_agent.schemas import (
    Claim,
    ClaimSource,
    Criticality,
    DocType,
    ExtractionOutput,
    FieldExtraction,
)


def make_field(
    name: str,
    value: Any,
    raw_quote: str = "test quote",
    span: tuple[int, int] = (0, 10),
) -> FieldExtraction:
    return FieldExtraction(field_name=name, value=value, raw_quote=raw_quote, char_span=span)


def make_extraction(
    doc_id: str = "test-001",
    doc_type: DocType = DocType.QUARTERLY_PORTFOLIO_REVIEW,
    fields: list[FieldExtraction] | None = None,
    summary: str = "All is well.",
) -> ExtractionOutput:
    return ExtractionOutput(
        doc_id=doc_id,
        predicted_doc_type=doc_type,
        fields=fields or [],
        summary=summary,
    )


def make_claim(
    *,
    claim_id: str = "DOC_0000_field_client_name",
    doc_id: str = "DOC_0000",
    source: ClaimSource = ClaimSource.STRUCTURED_FIELD,
    field_name: str = "client_name",
    statement: str = "client_name = Jane Doe",
    criticality: Criticality = Criticality.LOW,
    value: Any = None,
    raw_quote: str | None = None,
    char_span: tuple[int, int] | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        doc_id=doc_id,
        source=source,
        field_name=field_name,
        statement=statement,
        value=value,
        raw_quote=raw_quote,
        char_span=char_span,
        criticality=criticality,
    )