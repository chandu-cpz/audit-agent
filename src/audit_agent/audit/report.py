"""Report builder — assembles AuditReport and renders analyst-facing views.

Three report elements (from the blueprint):
1. Discrepancy highlight: side-by-side claimed value vs. source snippet
2. Categorical tags: color-coded badges
3. Causal trace: one-sentence rationale linking check → source span
Plus: human-review focus list (1–3 bullets)
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from audit_agent.schemas import (
    AuditFinding,
    AuditReport,
    AuditTag,
    CheckResult,
    CheckStatus,
    ClaimVerdict,
    Criticality,
    DocType,
    RoutingDecision,
    ScrutinyLevel,
)


TAG_COLORS = {
    AuditTag.FOOTING_ERROR: "red",
    AuditTag.CITATION_NOT_FOUND: "red",
    AuditTag.QUOTE_VALUE_MISMATCH: "red",
    AuditTag.MISSING_REQUIRED_FIELD: "red",
    AuditTag.SCHEMA_INVALID: "red",
    AuditTag.TYPE_MISMATCH: "red",
    AuditTag.NUMERIC_RANGE_VIOLATION: "red",
    AuditTag.QAG_MISMATCH: "yellow",
    AuditTag.MISSED_FLAG: "bright_red",
    AuditTag.SUMMARY_OVERREACH: "yellow",
    AuditTag.CROSS_FIELD_VIOLATION: "yellow",
    AuditTag.LOW_TOKEN_CONFIDENCE: "cyan",
    AuditTag.SELF_CONSISTENCY_FAIL: "yellow",
    AuditTag.ABSTAIN: "magenta",
}

ROUTING_COLORS = {
    RoutingDecision.TRUST: "green",
    RoutingDecision.HUMAN_REVIEW: "yellow",
    RoutingDecision.REPROCESS: "red",
}


def build_report(
    doc_id: str,
    doc_type: DocType,
    scrutiny: ScrutinyLevel,
    routing: RoutingDecision,
    verdicts: list[ClaimVerdict],
    headline_reasons: list[str],
    source_text: str,
    tier2_ran: bool = False,
) -> AuditReport:
    """Assemble a full AuditReport from audit results."""

    findings = _build_findings(verdicts, source_text)
    focus = _build_focus_list(verdicts, routing, source_text)
    confidence = _compute_confidence(verdicts)

    return AuditReport(
        doc_id=doc_id,
        doc_type=doc_type,
        scrutiny_level=scrutiny,
        routing_decision=routing,
        overall_confidence=confidence,
        headline_reasons=headline_reasons,
        human_review_focus=focus,
        findings=findings,
        claim_verdicts=verdicts,
        tier1_pass=all(
            v.confidence != CheckStatus.FAIL
            for v in verdicts
            if not any(c.check_name.startswith("tier2_") for c in v.checks)
        ),
        tier2_ran=tier2_ran,
    )


def _build_findings(
    verdicts: list[ClaimVerdict],
    source_text: str,
) -> list[AuditFinding]:
    """Convert failed check results into structured findings."""
    findings = []
    for v in verdicts:
        for check in v.checks:
            if check.status != CheckStatus.FAIL:
                continue
            evidence = check.evidence_span or _best_source_evidence(v, source_text)
            findings.append(AuditFinding(
                claim_id=v.claim_id,
                tag=check.tag or AuditTag.SCHEMA_INVALID,
                severity=v.criticality,
                claimed_value=v.statement,
                source_evidence=evidence,
                char_span=v.char_span,
                reason=check.reason,
                causal_trace=_make_causal_trace(check, v),
            ))
    return findings


def _make_causal_trace(check: CheckResult, verdict: ClaimVerdict) -> str:
    """One-sentence explanation linking the check to the conflict."""
    field = verdict.field_name or "summary claim"
    if check.evidence_span:
        return (
            f"Check '{check.check_name}' on '{field}' found: {check.reason}. "
            f"Source evidence: \"{check.evidence_span[:100]}\"."
        )
    return f"Check '{check.check_name}' on '{field}' found: {check.reason}."


def _build_focus_list(
    verdicts: list[ClaimVerdict],
    routing: RoutingDecision,
    source_text: str,
) -> list[str]:
    """1–3 actionable bullets telling the analyst where to look."""
    if routing == RoutingDecision.TRUST:
        return []

    focus = []
    # Prioritize critical/high failures
    for v in sorted(verdicts, key=lambda x: _crit_order(x.criticality)):
        if v.confidence == CheckStatus.FAIL:
            for c in v.checks:
                if c.status == CheckStatus.FAIL and len(focus) < 3:
                    focus.append(_focus_entry(v, c, source_text))
        elif v.confidence == CheckStatus.ABSTAIN and len(focus) < 3:
            focus.append(_focus_entry(v, None, source_text, abstain=True))

    return focus[:3]


def _crit_order(c: Criticality) -> int:
    return {Criticality.CRITICAL: 0, Criticality.HIGH: 1, Criticality.MEDIUM: 2, Criticality.LOW: 3}[c]


def _compute_confidence(verdicts: list[ClaimVerdict]) -> float:
    """Compute overall confidence as fraction of passing claims, weighted by criticality."""
    if not verdicts:
        return 1.0

    weights = {Criticality.CRITICAL: 4, Criticality.HIGH: 3, Criticality.MEDIUM: 2, Criticality.LOW: 1}
    total_weight = sum(weights[v.criticality] for v in verdicts)
    pass_weight = sum(
        weights[v.criticality] for v in verdicts if v.confidence == CheckStatus.PASS
    )
    return round(pass_weight / max(total_weight, 1), 3)


def _best_source_evidence(verdict: ClaimVerdict, source_text: str) -> str | None:
    if verdict.raw_quote:
        return verdict.raw_quote
    if verdict.char_span:
        return _snippet_for_span(source_text, verdict.char_span)
    return None


def _focus_entry(
    verdict: ClaimVerdict,
    check: CheckResult | None,
    source_text: str,
    abstain: bool = False,
) -> str:
    field = verdict.field_name or "summary"
    where = _format_location(verdict.char_span)
    evidence = _best_source_evidence(verdict, source_text)
    if abstain:
        prefix = f"Verify {field}{where}: could not determine correctness"
    else:
        prefix = f"Check {field}{where}: {check.reason[:100]}" if check else f"Check {field}{where}"
    if evidence:
        return f"{prefix}. Evidence: {evidence[:90]}"
    return prefix


def _format_location(char_span: tuple[int, int] | None) -> str:
    if not char_span or char_span == (0, 0):
        return ""
    start, end = char_span
    return f" at chars {start}:{end}"


def _snippet_for_span(source_text: str, char_span: tuple[int, int], context: int = 50) -> str:
    start, end = char_span
    left = max(0, start - context)
    right = min(len(source_text), end + context)
    snippet = source_text[left:right].replace("\n", " ").strip()
    if left > 0:
        snippet = f"...{snippet}"
    if right < len(source_text):
        snippet = f"{snippet}..."
    return snippet


def render_report(report: AuditReport, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    # Header
    routing_color = ROUTING_COLORS.get(report.routing_decision, "white")
    header = Text()
    header.append(f"[{report.doc_id}] ", style="bold")
    header.append(f"{report.doc_type.value} ", style="dim")
    header.append("→ ", style="dim")
    header.append(f"{report.routing_decision.value.upper()}", style=f"bold {routing_color}")
    header.append(f"  confidence={report.overall_confidence:.0%}", style="dim")
    header.append(f"  scrutiny={report.scrutiny_level.value}", style="dim")

    console.print(header)

    # Headline reasons
    if report.headline_reasons:
        for r in report.headline_reasons:
            console.print(f"  ├─ {r}", style="dim")

    # Findings table
    if report.findings:
        table = Table(show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("Tag", width=18)
        table.add_column("Severity", width=10)
        table.add_column("Field", width=20)
        table.add_column("Evidence", width=42)
        table.add_column("Reason", width=44)

        for f in report.findings:
            tag_color = TAG_COLORS.get(f.tag, "white")
            field_name = f.claim_id.split("_field_", 1)[1] if "_field_" in f.claim_id else f.claim_id
            table.add_row(
                Text(f.tag.value, style=tag_color),
                Text(f.severity.value, style="bold" if f.severity in (Criticality.CRITICAL, Criticality.HIGH) else ""),
                field_name[:20],
                (f.source_evidence or "n/a")[:42],
                f.reason[:44],
            )
        console.print(table)

    # Focus list
    if report.human_review_focus:
        console.print("  [bold yellow]Human Review Focus:[/bold yellow]")
        for item in report.human_review_focus:
            console.print(f"    → {item}")

    console.print()


def render_batch_summary(reports: list[AuditReport], console: Console | None = None) -> None:
    """Print a summary table for a batch of audit reports."""
    if console is None:
        console = Console()

    trust = sum(1 for r in reports if r.routing_decision == RoutingDecision.TRUST)
    review = sum(1 for r in reports if r.routing_decision == RoutingDecision.HUMAN_REVIEW)
    reprocess = sum(1 for r in reports if r.routing_decision == RoutingDecision.REPROCESS)
    total = len(reports)

    console.print()
    console.print("[bold]═══ Batch Audit Summary ═══[/bold]")
    console.print(f"  Total documents:  {total}")
    console.print(f"  [green]Trusted:[/green]          {trust} ({trust/total*100:.0f}%)")
    console.print(f"  [yellow]Human review:[/yellow]     {review} ({review/total*100:.0f}%)")
    console.print(f"  [red]Reprocess:[/red]        {reprocess} ({reprocess/total*100:.0f}%)")

    # Confidence distribution
    avg_conf = sum(r.overall_confidence for r in reports) / max(total, 1)
    console.print(f"  Avg confidence:   {avg_conf:.0%}")

    # Finding counts by tag
    tag_counts: dict[str, int] = {}
    for r in reports:
        for f in r.findings:
            tag_counts[f.tag.value] = tag_counts.get(f.tag.value, 0) + 1

    if tag_counts:
        console.print("\n  [bold]Finding distribution:[/bold]")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            console.print(f"    {tag}: {count}")
    console.print()
