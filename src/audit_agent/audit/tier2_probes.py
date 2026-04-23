"""Tier-2 targeted LLM probes — focused verification, NOT full re-extraction.

Each probe asks one question with structured output and returns
agree/disagree/abstain with an evidence span.
Only fired for high-stakes claims or scrutiny >= MEDIUM.
"""

from __future__ import annotations

import logging
import re

from audit_agent.nim_client import NIMClient
from audit_agent.schemas import (
    AbstainReason,
    AuditTag,
    CheckResult,
    CheckStatus,
    Claim,
    ClaimSource,
    Criticality,
    ScrutinyLevel,
)

logger = logging.getLogger(__name__)

NUMBER_TOKEN_PATTERN = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def run_tier2(
    client: NIMClient,
    model: str,
    source_text: str,
    claims: list[Claim],
    scrutiny: ScrutinyLevel,
    doc_type: str,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    # Only run on claims that are high/critical criticality OR scrutiny >= MEDIUM
    should_run = scrutiny in (ScrutinyLevel.MEDIUM, ScrutinyLevel.HIGH, ScrutinyLevel.CRITICAL)

    for claim in claims:
        if not should_run and claim.criticality not in (Criticality.HIGH, Criticality.CRITICAL):
            continue

        # Probe 1: Reverify high-stakes claims
        if claim.criticality in (Criticality.HIGH, Criticality.CRITICAL):
            result = probe_reverify_claim(client, model, source_text, claim)
            if result:
                results.append(result)

        # Probe 2: Check for missed compliance flags (compliance docs only)
        if doc_type == "compliance_disclosure" and claim.source == ClaimSource.STRUCTURED_FIELD:
            if claim.field_name and "flag" in claim.field_name or claim.field_name in (
                "pep_flag", "sanctions_screening", "aml_status", "adverse_media"
            ):
                result = probe_missed_compliance_flags(client, model, source_text, claim)
                if result:
                    results.append(result)

        # Probe 3: Narrative entailment for summary claims
        if claim.source == ClaimSource.SUMMARY and claim.criticality in (
            Criticality.HIGH, Criticality.CRITICAL, Criticality.MEDIUM
        ):
            result = probe_narrative_entailment(client, model, source_text, claim)
            if result:
                results.append(result)

    return results


def probe_reverify_claim(
    client: NIMClient,
    model: str,
    source_text: str,
    claim: Claim,
) -> CheckResult | None:
    messages = [
        {"role": "system", "content": REVERIFY_SYSTEM},
        {"role": "user", "content": (
            f"CLAIM: {claim.statement}\n\n"
            f"SOURCE DOCUMENT (excerpt, max 3000 chars):\n"
            f"{source_text[:3000]}\n\n"
            f"Respond in JSON: {{\"verdict\": \"agree|disagree|abstain\", "
            f"\"evidence_span\": \"exact quote from source\", "
            f"\"reason\": \"brief explanation\"}}"
        )},
    ]

    try:
        parsed = client.chat_json(model=model, messages=messages)
        verdict = parsed.get("verdict", "abstain").lower()
        evidence = parsed.get("evidence_span", "")
        reason = parsed.get("reason", "")

        status = {
            "agree": CheckStatus.PASS,
            "disagree": CheckStatus.FAIL,
        }.get(verdict, CheckStatus.ABSTAIN)

        tag = None
        abstain_reason = None
        if status == CheckStatus.FAIL:
            tag = AuditTag.QAG_MISMATCH
        elif status == CheckStatus.ABSTAIN:
            tag = AuditTag.ABSTAIN
            abstain_reason = AbstainReason.INSUFFICIENT_EVIDENCE

        return CheckResult(
            check_name="tier2_reverify",
            claim_id=claim.claim_id,
            status=status,
            tag=tag,
            reason=f"Reverify: {reason}",
            evidence_span=evidence if evidence else None,
            abstain_reason=abstain_reason,
        )
    except Exception as e:
        logger.warning("Tier-2 reverify failed for %s: %s", claim.claim_id, e)
        return CheckResult(
            check_name="tier2_reverify",
            claim_id=claim.claim_id,
            status=CheckStatus.ABSTAIN,
            tag=AuditTag.ABSTAIN,
            reason=f"Probe failed: {e}",
            abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE,
        )


def probe_missed_compliance_flags(
    client: NIMClient,
    model: str,
    source_text: str,
    claim: Claim,
) -> CheckResult | None:
    messages = [
        {"role": "system", "content": COMPLIANCE_SYSTEM},
        {"role": "user", "content": (
            f"The primary extraction claims: {claim.statement}\n\n"
            f"SOURCE DOCUMENT:\n{source_text[:3000]}\n\n"
            f"Are there any compliance flags, risk warnings, or regulatory "
            f"requirements in the source that are NOT captured by this claim? "
            f"Respond in JSON: {{\"missed_flags\": [\"flag1\", ...], "
            f"\"verdict\": \"agree|disagree|abstain\", "
            f"\"evidence_span\": \"exact quote\", \"reason\": \"...\"}}"
        )},
    ]

    try:
        parsed = client.chat_json(model=model, messages=messages)
        missed = parsed.get("missed_flags", [])

        if missed:
            return CheckResult(
                check_name="tier2_compliance_flags",
                claim_id=claim.claim_id,
                status=CheckStatus.FAIL,
                tag=AuditTag.MISSED_FLAG,
                reason=f"Missed compliance flags: {', '.join(missed)}",
                evidence_span=parsed.get("evidence_span"),
            )
        else:
            return CheckResult(
                check_name="tier2_compliance_flags",
                claim_id=claim.claim_id,
                status=CheckStatus.PASS,
                reason="No missed compliance flags detected",
            )
    except Exception as e:
        logger.warning("Tier-2 compliance probe failed: %s", e)
        return CheckResult(
            check_name="tier2_compliance_flags",
            claim_id=claim.claim_id,
            status=CheckStatus.ABSTAIN,
            tag=AuditTag.ABSTAIN,
            reason=f"Compliance probe failed: {e}",
            abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE,
        )


def probe_narrative_entailment(
    client: NIMClient,
    model: str,
    source_text: str,
    claim: Claim,
) -> CheckResult | None:
    messages = [
        {"role": "system", "content": ENTAILMENT_SYSTEM},
        {"role": "user", "content": (
            f"SUMMARY CLAIM: \"{claim.statement}\"\n\n"
            f"SOURCE DOCUMENT (excerpt):\n{source_text[:3000]}\n\n"
            f"Is this claim fully supported by the source? "
            f"Respond in JSON: {{\"verdict\": \"agree|disagree|abstain\", "
            f"\"evidence_span\": \"exact supporting quote or empty\", "
            f"\"reason\": \"brief explanation\"}}"
        )},
    ]

    try:
        parsed = client.chat_json(model=model, messages=messages)
        verdict = parsed.get("verdict", "abstain").lower()
        evidence = parsed.get("evidence_span", "")
        reason = parsed.get("reason", "")

        verdict, reason = _normalize_summary_verdict(claim, verdict, evidence, reason)

        status = {
            "agree": CheckStatus.PASS,
            "disagree": CheckStatus.FAIL,
        }.get(verdict, CheckStatus.ABSTAIN)

        tag = None
        abstain_reason = None
        if status == CheckStatus.FAIL:
            tag = AuditTag.QAG_MISMATCH
        elif status == CheckStatus.ABSTAIN:
            tag = AuditTag.ABSTAIN
            abstain_reason = AbstainReason.INDIRECT_SUPPORT

        return CheckResult(
            check_name="tier2_entailment",
            claim_id=claim.claim_id,
            status=status,
            tag=tag,
            reason=f"Entailment: {reason}",
            evidence_span=evidence if evidence else None,
            abstain_reason=abstain_reason,
        )
    except Exception as e:
        logger.warning("Tier-2 entailment failed for %s: %s", claim.claim_id, e)
        return CheckResult(
            check_name="tier2_entailment",
            claim_id=claim.claim_id,
            status=CheckStatus.ABSTAIN,
            tag=AuditTag.ABSTAIN,
            reason=f"Entailment probe failed: {e}",
            abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE,
        )


def _normalize_summary_verdict(
    claim: Claim,
    verdict: str,
    evidence: str,
    reason: str,
) -> tuple[str, str]:
    """Downgrade over-literal summary mismatches when only rounding differs."""
    if verdict != "disagree" or not evidence:
        return verdict, reason
    if claim.source != ClaimSource.SUMMARY:
        return verdict, reason
    if _is_rounding_only_difference(claim.statement, evidence):
        return "agree", "Rounded or formatted summary wording preserved the same underlying fact"
    return verdict, reason


def _is_rounding_only_difference(claim_text: str, evidence_text: str) -> bool:
    claim_numbers = _extract_number_tokens(claim_text)
    evidence_numbers = _extract_number_tokens(evidence_text)
    if not claim_numbers or not evidence_numbers:
        return False

    return all(
        any(_numbers_match_with_rounding(claim_number, evidence_number) for evidence_number in evidence_numbers)
        for claim_number in claim_numbers
    )


def _extract_number_tokens(text: str) -> list[tuple[float, int]]:
    tokens: list[tuple[float, int]] = []
    for token in NUMBER_TOKEN_PATTERN.findall(text):
        normalized = token.replace(",", "")
        try:
            value = float(normalized)
        except ValueError:
            continue
        decimals = len(normalized.split(".", 1)[1]) if "." in normalized else 0
        tokens.append((value, decimals))
    return tokens


def _numbers_match_with_rounding(
    claim_number: tuple[float, int],
    evidence_number: tuple[float, int],
) -> bool:
    claim_value, claim_decimals = claim_number
    evidence_value, evidence_decimals = evidence_number
    tolerance = max(_rounding_step(claim_decimals), _rounding_step(evidence_decimals)) / 2
    return abs(claim_value - evidence_value) <= tolerance


def _rounding_step(decimals: int) -> float:
    return 10 ** (-decimals) if decimals > 0 else 1.0


REVERIFY_SYSTEM = """\
You are a financial document audit system. Your job is to verify a single specific
claim against a source document. You must respond ONLY with a JSON object.

Rules:
- "agree" means the claim is directly supported by evidence in the source
- "disagree" means the source contradicts the claim or the claim is not supported
- "abstain" means the source is ambiguous or you cannot determine either way
- Treat harmless formatting changes, currency punctuation, and rounded renderings of the same value as "agree"
- Always provide the exact evidence_span from the source that supports your verdict
- If you cannot find relevant text, set evidence_span to empty string and verdict to "abstain"
"""

COMPLIANCE_SYSTEM = """\
You are a financial compliance audit system. Check whether the source document
contains any compliance flags, regulatory requirements, or risk warnings that
are NOT captured by the given extraction claim. Respond ONLY with JSON.

Focus on: PEP status, sanctions, AML, KYC, suitability, risk disclosures,
conflicts of interest, fee disclosures, adverse media, and regulatory flags.
"""

ENTAILMENT_SYSTEM = """\
You are a financial document verification system. Determine whether a summary
claim is fully supported (entailed) by the source document. Respond ONLY with JSON.

Rules:
- "agree" = the source directly supports this claim
- "disagree" = the source contradicts this claim or the claim adds information not in the source
- "abstain" = the source is ambiguous about this specific claim
- Rounded numbers and faithful paraphrases still count as "agree" when they preserve the same underlying fact
- Only use "disagree" for material numeric shifts, unsupported details, or substantive contradiction
- Provide the exact evidence_span from the source
"""
