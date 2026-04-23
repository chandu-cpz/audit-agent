from __future__ import annotations

from audit_agent.audit.router import route
from audit_agent.audit.scrutiny import compute_scrutiny
from audit_agent.schemas import (
    AuditTag,
    CheckResult,
    CheckStatus,
    Criticality,
    DocType,
    RoutingDecision,
    ScrutinyLevel,
)
from audit_agent.testing import make_claim as _make_claim
from audit_agent.testing import make_extraction as _make_extraction


def test_compute_scrutiny_simple_account_opening_is_low():
    extraction = _make_extraction(doc_id="DOC_0000", doc_type=DocType.ACCOUNT_OPENING)
    claims = [_make_claim()]

    scrutiny = compute_scrutiny("Client Name: Jane Doe", extraction, claims)

    assert scrutiny == ScrutinyLevel.LOW


def test_compute_scrutiny_critical_claim_escalates_to_medium():
    extraction = _make_extraction(doc_id="DOC_0000", doc_type=DocType.ACCOUNT_OPENING)
    claims = [_make_claim(criticality=Criticality.CRITICAL)]

    scrutiny = compute_scrutiny("Client Name: Jane Doe", extraction, claims)

    assert scrutiny == ScrutinyLevel.MEDIUM


def test_compute_scrutiny_tier1_failure_bumps_one_level():
    extraction = _make_extraction(doc_id="DOC_0000", doc_type=DocType.ACCOUNT_OPENING)
    claims = [_make_claim()]
    tier1_results = [
        CheckResult(
            check_name="required_fields",
            claim_id="DOC_0000_field_client_name",
            status=CheckStatus.FAIL,
            tag=AuditTag.MISSING_REQUIRED_FIELD,
            reason="Missing required field",
        )
    ]

    scrutiny = compute_scrutiny("Client Name: Jane Doe", extraction, claims, tier1_results)

    assert scrutiny == ScrutinyLevel.MEDIUM


def test_route_critical_mechanical_failure_reprocesses():
    claim = _make_claim(
        claim_id="DOC_0000_field_total_portfolio_value",
        field_name="total_portfolio_value",
        statement="total_portfolio_value = 1000000",
        criticality=Criticality.CRITICAL,
    )
    checks = [
        CheckResult(
            check_name="footing_dollar",
            claim_id=claim.claim_id,
            status=CheckStatus.FAIL,
            tag=AuditTag.FOOTING_ERROR,
            reason="Dollar sum does not reconcile",
        )
    ]

    decision, _, reasons = route(
        DocType.QUARTERLY_PORTFOLIO_REVIEW,
        ScrutinyLevel.MEDIUM,
        [claim],
        checks,
    )

    assert decision == RoutingDecision.REPROCESS
    assert any("Reprocessable:" in reason for reason in reasons)


def test_route_conflicting_tier1_and_tier2_signals_escalates():
    claim = _make_claim(criticality=Criticality.HIGH)
    checks = [
        CheckResult(
            check_name="required_fields",
            claim_id=claim.claim_id,
            status=CheckStatus.PASS,
            reason="OK",
        ),
        CheckResult(
            check_name="tier2_reverify",
            claim_id=claim.claim_id,
            status=CheckStatus.FAIL,
            tag=AuditTag.QAG_MISMATCH,
            reason="Tier-2 found a contradiction",
        ),
    ]

    decision, _, reasons = route(
        DocType.ACCOUNT_OPENING,
        ScrutinyLevel.LOW,
        [claim],
        checks,
    )

    assert decision == RoutingDecision.HUMAN_REVIEW
    assert any("Conflicting signals" in reason for reason in reasons)


def test_route_high_scrutiny_without_tier2_flags_for_review():
    claim = _make_claim()
    checks = [
        CheckResult(
            check_name="required_fields",
            claim_id=claim.claim_id,
            status=CheckStatus.PASS,
            reason="OK",
        )
    ]

    decision, _, reasons = route(
        DocType.ACCOUNT_OPENING,
        ScrutinyLevel.HIGH,
        [claim],
        checks,
    )

    assert decision == RoutingDecision.HUMAN_REVIEW
    assert any("Tier-2 did not run" in reason for reason in reasons)