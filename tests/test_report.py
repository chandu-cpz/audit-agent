from __future__ import annotations

from audit_agent.audit.report import build_report
from audit_agent.schemas import (
    AuditTag,
    CheckResult,
    CheckStatus,
    ClaimVerdict,
    Criticality,
    DocType,
    RoutingDecision,
    ScrutinyLevel,
)


def test_build_report_carries_source_evidence_and_span() -> None:
    verdict = ClaimVerdict(
        claim_id="DOC_0001_field_total_portfolio_value",
        field_name="total_portfolio_value",
        statement="total_portfolio_value = 1050000",
        raw_quote="Total Portfolio Value: $1,000,000.00",
        char_span=(24, 60),
        criticality=Criticality.HIGH,
        confidence=CheckStatus.FAIL,
        checks=[
            CheckResult(
                check_name="quote_value_consistency",
                claim_id="DOC_0001_field_total_portfolio_value",
                status=CheckStatus.FAIL,
                tag=AuditTag.QUOTE_VALUE_MISMATCH,
                reason="Extracted value 1050000 does not match numeric evidence in quote",
            )
        ],
        tags=[AuditTag.QUOTE_VALUE_MISMATCH],
    )

    report = build_report(
        doc_id="DOC_0001",
        doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
        scrutiny=ScrutinyLevel.HIGH,
        routing=RoutingDecision.REPROCESS,
        verdicts=[verdict],
        headline_reasons=["Reprocessable: total_portfolio_value mismatch"],
        source_text="Header\nTotal Portfolio Value: $1,000,000.00\nFooter",
        tier2_ran=False,
    )

    assert report.findings[0].source_evidence == "Total Portfolio Value: $1,000,000.00"
    assert report.findings[0].char_span == (24, 60)
    assert "chars 24:60" in report.human_review_focus[0]
    assert "Evidence:" in report.human_review_focus[0]