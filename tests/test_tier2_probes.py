from __future__ import annotations

from audit_agent.audit.tier2_probes import (
    probe_missed_compliance_flags,
    probe_narrative_entailment,
    probe_reverify_claim,
    run_tier2,
)
from audit_agent.schemas import (
    AbstainReason,
    AuditTag,
    Claim,
    ClaimSource,
    CheckStatus,
    Criticality,
    ScrutinyLevel,
)
from audit_agent.testing import make_claim as _make_field_claim


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, model, messages):
        return self.payload


class RaisingClient:
    def __init__(self, error: Exception | None = None):
        self.error = error or RuntimeError("probe unavailable")

    def chat_json(self, model, messages):
        raise self.error


def test_probe_narrative_entailment_softens_rounding_only_difference():
    client = StubClient(
        {
            "verdict": "disagree",
            "evidence_span": "Total Portfolio Value: $33,996,624.63",
            "reason": "The summary rounds the total value to $33,996,625.",
        }
    )
    claim = Claim(
        claim_id="DOC_0000_summary_1",
        doc_id="DOC_0000",
        source=ClaimSource.SUMMARY,
        statement="Total value: $33,996,625.",
        criticality=Criticality.MEDIUM,
    )

    result = probe_narrative_entailment(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.PASS
    assert result.tag is None
    assert result.evidence_span == "Total Portfolio Value: $33,996,624.63"
    assert "Rounded or formatted summary wording" in result.reason


def test_probe_narrative_entailment_keeps_material_numeric_shift():
    client = StubClient(
        {
            "verdict": "disagree",
            "evidence_span": "Investment Time Horizon: 15 years",
            "reason": "The summary says 10 years but the source says 15 years.",
        }
    )
    claim = Claim(
        claim_id="DOC_0001_summary_0",
        doc_id="DOC_0001",
        source=ClaimSource.SUMMARY,
        statement="The investment time horizon is 10 years.",
        criticality=Criticality.MEDIUM,
    )

    result = probe_narrative_entailment(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.FAIL
    assert result.tag == AuditTag.QAG_MISMATCH


def test_probe_reverify_claim_disagree_tags_qag_mismatch():
    client = StubClient(
        {
            "verdict": "disagree",
            "evidence_span": "Risk Tolerance: Conservative",
            "reason": "The claim says moderate but the source says conservative.",
        }
    )
    claim = _make_field_claim(
        claim_id="DOC_0000_field_risk_tolerance",
        field_name="risk_tolerance",
        statement="risk_tolerance = moderate",
        criticality=Criticality.CRITICAL,
    )

    result = probe_reverify_claim(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.FAIL
    assert result.tag == AuditTag.QAG_MISMATCH
    assert result.evidence_span == "Risk Tolerance: Conservative"


def test_probe_reverify_claim_failure_abstains_with_reason():
    client = RaisingClient()
    claim = _make_field_claim(
        claim_id="DOC_0000_field_total_portfolio_value",
        field_name="total_portfolio_value",
        statement="total_portfolio_value = 2500000",
        criticality=Criticality.CRITICAL,
    )

    result = probe_reverify_claim(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.ABSTAIN
    assert result.tag == AuditTag.ABSTAIN
    assert result.abstain_reason == AbstainReason.INSUFFICIENT_EVIDENCE


def test_probe_missed_compliance_flags_detects_missing_flags():
    client = StubClient(
        {
            "missed_flags": ["enhanced due diligence", "sanctions review"],
            "verdict": "disagree",
            "evidence_span": "Enhanced due diligence required before onboarding.",
            "reason": "The extraction omits escalation requirements.",
        }
    )
    claim = _make_field_claim(
        claim_id="DOC_0002_field_aml_status",
        field_name="aml_status",
        statement="aml_status = cleared",
        criticality=Criticality.CRITICAL,
    )

    result = probe_missed_compliance_flags(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.FAIL
    assert result.tag == AuditTag.MISSED_FLAG
    assert "enhanced due diligence" in result.reason


def test_probe_missed_compliance_flags_failure_abstains():
    client = RaisingClient()
    claim = _make_field_claim(
        claim_id="DOC_0002_field_pep_flag",
        field_name="pep_flag",
        statement="pep_flag = false",
        criticality=Criticality.CRITICAL,
    )

    result = probe_missed_compliance_flags(client, "model", "ignored", claim)

    assert result is not None
    assert result.status == CheckStatus.ABSTAIN
    assert result.tag == AuditTag.ABSTAIN
    assert result.abstain_reason == AbstainReason.INSUFFICIENT_EVIDENCE


def test_run_tier2_low_scrutiny_only_checks_high_stakes_claims():
    client = StubClient(
        {
            "verdict": "agree",
            "evidence_span": "Risk Tolerance: Moderate",
            "reason": "The claim is directly supported by the source.",
        }
    )
    claims = [
        _make_field_claim(
            claim_id="DOC_0000_field_risk_tolerance",
            field_name="risk_tolerance",
            statement="risk_tolerance = moderate",
            criticality=Criticality.CRITICAL,
        ),
        _make_field_claim(
            claim_id="DOC_0000_field_client_name",
            field_name="client_name",
            statement="client_name = Jane Doe",
            criticality=Criticality.LOW,
        ),
    ]

    results = run_tier2(
        client,
        "model",
        "Risk Tolerance: Moderate\nClient Name: Jane Doe",
        claims,
        ScrutinyLevel.LOW,
        "account_opening",
    )

    assert len(results) == 1
    assert results[0].claim_id == "DOC_0000_field_risk_tolerance"
    assert results[0].check_name == "tier2_reverify"