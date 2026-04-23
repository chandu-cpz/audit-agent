"""Pydantic v2 models — the contract that makes non-replicative auditing possible.

Key design choice: FieldExtraction requires `raw_quote` and `char_span` per field.
This is the citation contract the entire Tier-1 grounding check depends on.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class DocType(str, enum.Enum):
    ACCOUNT_OPENING = "account_opening"
    INVESTMENT_POLICY_STATEMENT = "investment_policy_statement"
    QUARTERLY_PORTFOLIO_REVIEW = "quarterly_portfolio_review"
    COMPLIANCE_DISCLOSURE = "compliance_disclosure"


class ScrutinyLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingDecision(str, enum.Enum):
    TRUST = "trust"
    HUMAN_REVIEW = "human_review"
    REPROCESS = "reprocess"


class CheckStatus(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class AbstainReason(str, enum.Enum):
    AMBIGUOUS_SOURCE = "ambiguous_source"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    INDIRECT_SUPPORT = "indirect_support"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    SOURCE_DEFECT = "source_defect"


class Criticality(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClaimSource(str, enum.Enum):
    """Where the claim originated from."""
    STRUCTURED_FIELD = "structured_field"
    SUMMARY = "summary"


class AuditTag(str, enum.Enum):
    FOOTING_ERROR = "FOOTING_ERROR"
    CITATION_NOT_FOUND = "CITATION_NOT_FOUND"
    QUOTE_VALUE_MISMATCH = "QUOTE_VALUE_MISMATCH"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    CROSS_FIELD_VIOLATION = "CROSS_FIELD_VIOLATION"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    NUMERIC_RANGE_VIOLATION = "NUMERIC_RANGE_VIOLATION"
    SUMMARY_OVERREACH = "SUMMARY_OVERREACH"
    QAG_MISMATCH = "QAG_MISMATCH"
    MISSED_FLAG = "MISSED_FLAG"
    SUMMARY_CLIENT_MISMATCH = "SUMMARY_CLIENT_MISMATCH"
    LOW_TOKEN_CONFIDENCE = "LOW_TOKEN_CONFIDENCE"
    SELF_CONSISTENCY_FAIL = "SELF_CONSISTENCY_FAIL"
    ABSTAIN = "ABSTAIN"


class SourceDocument(BaseModel):
    doc_id: str
    doc_type: DocType
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FieldExtraction(BaseModel):
    """A single extracted field with the mandatory citation contract."""

    field_name: str
    value: Any
    raw_quote: str = Field(
        description="Verbatim snippet from the source that supports the extracted value."
    )
    char_span: tuple[int, int] = Field(
        description="(start, end) character offsets of raw_quote in the source document."
    )


class ExtractionOutput(BaseModel):
    """The primary LLM's full output for one document."""

    doc_id: str
    predicted_doc_type: DocType
    fields: list[FieldExtraction]
    summary: str = Field(description="Prose summary the relationship manager reads.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """Atomic auditable claim — either from a structured field or the summary."""

    claim_id: str
    doc_id: str
    source: ClaimSource
    field_name: str | None = None  # set for structured field claims
    statement: str  # human-readable claim text
    value: Any = None
    raw_quote: str | None = None
    char_span: tuple[int, int] | None = None
    criticality: Criticality = Criticality.MEDIUM


class CheckResult(BaseModel):
    check_name: str
    claim_id: str
    status: CheckStatus
    tag: AuditTag | None = None
    reason: str = ""
    evidence_span: str | None = None
    abstain_reason: AbstainReason | None = None


class ClaimVerdict(BaseModel):
    """Aggregated verdict for a single claim across all tiers."""

    claim_id: str
    field_name: str | None = None
    statement: str
    raw_quote: str | None = None
    char_span: tuple[int, int] | None = None
    criticality: Criticality
    confidence: CheckStatus  # pass/fail/abstain used as high/low/uncertain
    checks: list[CheckResult] = Field(default_factory=list)
    tags: list[AuditTag] = Field(default_factory=list)


class AuditFinding(BaseModel):
    """One discrepancy to show the analyst."""

    claim_id: str
    tag: AuditTag
    severity: Criticality
    claimed_value: str
    source_evidence: str | None = None
    char_span: tuple[int, int] | None = None
    reason: str
    causal_trace: str = ""


class AuditReport(BaseModel):
    """The full audit output for one document."""

    doc_id: str
    doc_type: DocType
    scrutiny_level: ScrutinyLevel
    routing_decision: RoutingDecision
    overall_confidence: float = Field(ge=0.0, le=1.0)
    headline_reasons: list[str] = Field(default_factory=list)
    human_review_focus: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    tier1_pass: bool = True
    tier2_ran: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorLabel(BaseModel):
    """Ground truth for an injected error — never visible to the audit pipeline."""

    error_type: str
    affected_fields: list[str] = Field(default_factory=list)
    description: str = ""


class GroundTruth(BaseModel):
    doc_id: str
    is_corrupted: bool
    injected_errors: list[ErrorLabel] = Field(default_factory=list)
