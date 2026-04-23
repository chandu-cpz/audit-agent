"""Router — determines the routing decision from aggregated audit signals.

Pure function: no LLM calls, no side effects. Encodes:
- Policy override for compliance docs
- Halt-on-conflict (no autonomous re-loop)
- Per-claim confidence aggregation
"""

from __future__ import annotations

from audit_agent.config import doc_type_policies
from audit_agent.schemas import (
    AuditTag,
    CheckResult,
    CheckStatus,
    Claim,
    ClaimVerdict,
    Criticality,
    DocType,
    RoutingDecision,
    ScrutinyLevel,
)

# Tags that indicate mechanical failures fixable by re-run
REPROCESS_TAGS = {
    AuditTag.SCHEMA_INVALID,
    AuditTag.MISSING_REQUIRED_FIELD,
    AuditTag.FOOTING_ERROR,
    AuditTag.CITATION_NOT_FOUND,
    AuditTag.QUOTE_VALUE_MISMATCH,
    AuditTag.TYPE_MISMATCH,
}

# Tags that indicate semantic issues requiring human judgement
HUMAN_REVIEW_TAGS = {
    AuditTag.QAG_MISMATCH,
    AuditTag.MISSED_FLAG,
    AuditTag.SUMMARY_OVERREACH,
    AuditTag.CROSS_FIELD_VIOLATION,
    AuditTag.SELF_CONSISTENCY_FAIL,
}


def route(
    doc_type: DocType,
    scrutiny: ScrutinyLevel,
    claims: list[Claim],
    all_checks: list[CheckResult],
) -> tuple[RoutingDecision, list[ClaimVerdict], list[str]]:
    # Build per-claim verdicts
    verdicts = _build_claim_verdicts(claims, all_checks)

    # Collect all tags and failure info
    all_tags: set[AuditTag] = set()
    critical_failures: list[str] = []
    high_failures: list[str] = []
    has_abstain_high_stakes = False
    has_tier2_disagreement = False
    has_conflicting_signals = False

    for v in verdicts:
        all_tags.update(v.tags)
        if v.confidence == CheckStatus.FAIL:
            if v.criticality in (Criticality.CRITICAL, Criticality.HIGH):
                critical_failures.append(f"{v.field_name or 'summary'}: {_first_fail_reason(v)}")
            else:
                high_failures.append(f"{v.field_name or 'summary'}: {_first_fail_reason(v)}")
        elif v.confidence == CheckStatus.ABSTAIN:
            if v.criticality in (Criticality.CRITICAL, Criticality.HIGH):
                has_abstain_high_stakes = True

        # Check for Tier-2 disagreements
        for c in v.checks:
            if c.check_name.startswith("tier2_") and c.status == CheckStatus.FAIL:
                has_tier2_disagreement = True

        # Check for conflicting signals (Tier 1 pass but Tier 2 disagree)
        tier1_status = [c.status for c in v.checks if not c.check_name.startswith("tier2_")]
        tier2_status = [c.status for c in v.checks if c.check_name.startswith("tier2_")]
        if (all(s == CheckStatus.PASS for s in tier1_status) and tier1_status
                and any(s == CheckStatus.FAIL for s in tier2_status)):
            has_conflicting_signals = True

    # Policy override: compliance docs always go to human review
    policies = doc_type_policies()
    policy = policies.get(doc_type.value, {})
    policy_override = policy.get("policy_override")

    headline_reasons: list[str] = []

    # 1. Policy override
    if policy_override == "human_review":
        headline_reasons.append(f"Policy override: {doc_type.value} always requires human review")
        # Still check for reprocess-worthy failures
        if _has_reprocess_tags(all_tags) and not critical_failures:
            return RoutingDecision.REPROCESS, verdicts, [
                *headline_reasons,
                "Mechanical failure detected — reprocessing before human review",
            ]
        return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 2. Halt on conflict — escalate to human, never auto re-loop
    if has_conflicting_signals:
        headline_reasons.append("Conflicting signals: Tier-1 passed but Tier-2 disagreed — halting")
        return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 3. Critical failures → decide between reprocess and human review
    if critical_failures:
        reprocess_worthy = all_tags & REPROCESS_TAGS
        human_worthy = all_tags & HUMAN_REVIEW_TAGS

        if human_worthy or has_tier2_disagreement:
            headline_reasons.extend([f"Critical: {r}" for r in critical_failures])
            return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons
        elif reprocess_worthy:
            headline_reasons.extend([f"Reprocessable: {r}" for r in critical_failures])
            return RoutingDecision.REPROCESS, verdicts, headline_reasons
        else:
            headline_reasons.extend([f"Critical: {r}" for r in critical_failures])
            return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 4. High-stakes abstention → human review
    if has_abstain_high_stakes:
        headline_reasons.append("Abstained on high-stakes claim — cannot verify, escalating")
        return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 5. Non-critical failures
    if high_failures:
        if _has_reprocess_tags(all_tags) and not (all_tags & HUMAN_REVIEW_TAGS):
            headline_reasons.extend([f"Minor: {r}" for r in high_failures])
            return RoutingDecision.REPROCESS, verdicts, headline_reasons
        headline_reasons.extend([f"Review: {r}" for r in high_failures])
        return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 6. Scrutiny-based gate
    if scrutiny in (ScrutinyLevel.HIGH, ScrutinyLevel.CRITICAL):
        # High scrutiny with no failures still gets human review if Tier-2 didn't run
        tier2_ran = any(c.check_name.startswith("tier2_") for v in verdicts for c in v.checks)
        if not tier2_ran:
            headline_reasons.append(
                f"Scrutiny={scrutiny.value} but Tier-2 did not run — flagging for review"
            )
            return RoutingDecision.HUMAN_REVIEW, verdicts, headline_reasons

    # 7. All clear → trust
    headline_reasons.append("All checks passed")
    return RoutingDecision.TRUST, verdicts, headline_reasons


def _build_claim_verdicts(
    claims: list[Claim],
    all_checks: list[CheckResult],
) -> list[ClaimVerdict]:
    """Aggregate check results into per-claim verdicts."""
    # Index checks by claim_id
    checks_by_claim: dict[str, list[CheckResult]] = {}
    for c in all_checks:
        checks_by_claim.setdefault(c.claim_id, []).append(c)

    verdicts = []
    for claim in claims:
        checks = checks_by_claim.get(claim.claim_id, [])
        tags = [c.tag for c in checks if c.tag is not None]

        # Determine confidence: fail > abstain > pass
        has_fail = any(c.status == CheckStatus.FAIL for c in checks)
        has_abstain = any(c.status == CheckStatus.ABSTAIN for c in checks)

        if has_fail:
            confidence = CheckStatus.FAIL
        elif has_abstain:
            confidence = CheckStatus.ABSTAIN
        else:
            confidence = CheckStatus.PASS

        verdicts.append(ClaimVerdict(
            claim_id=claim.claim_id,
            field_name=claim.field_name,
            statement=claim.statement,
            raw_quote=claim.raw_quote,
            char_span=claim.char_span,
            criticality=claim.criticality,
            confidence=confidence,
            checks=checks,
            tags=tags,
        ))

    return verdicts


def _has_reprocess_tags(tags: set[AuditTag]) -> bool:
    return bool(tags & REPROCESS_TAGS)


def _first_fail_reason(verdict: ClaimVerdict) -> str:
    for c in verdict.checks:
        if c.status == CheckStatus.FAIL:
            return c.reason
    return "unknown"
