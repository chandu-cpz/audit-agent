"""Evaluation module — the ONLY place that sees ground-truth labels.

Computes:
- Detection metrics per error type and per doc type
- Routing quality (false-trust rate on critical claims)
- Business-weighted impact score
- Random-10%-sampling baseline comparison
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from audit_agent.schemas import (
    AuditReport,
    GroundTruth,
    RoutingDecision,
)
from audit_agent.config import field_criticality

CRITICALITY_WEIGHTS = {"critical": 10, "high": 5, "medium": 2, "low": 1}


def evaluate(
    reports: list[AuditReport],
    ground_truths: list[GroundTruth],
    console: Console | None = None,
) -> dict[str, Any]:
    """Run full evaluation and print results."""
    if console is None:
        console = Console()

    gt_map = {gt.doc_id: gt for gt in ground_truths}
    results: dict[str, Any] = {}

    tp = fp = fn = tn = 0
    per_error_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0})
    per_doc_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    missed_critical = 0
    total_critical = 0
    business_impact_caught = 0.0
    business_impact_missed = 0.0

    for report in reports:
        gt = gt_map.get(report.doc_id)
        if not gt:
            continue

        flagged = report.routing_decision != RoutingDecision.TRUST
        corrupted = gt.is_corrupted
        doc_type = report.doc_type.value

        if corrupted and flagged:
            tp += 1
            per_doc_type[doc_type]["tp"] += 1
            for err in gt.injected_errors:
                per_error_type[err.error_type]["tp"] += 1
                impact = _error_impact(err.error_type, err.affected_fields, doc_type)
                business_impact_caught += impact
        elif corrupted and not flagged:
            fn += 1  # FALSE TRUST — worst outcome
            per_doc_type[doc_type]["fn"] += 1
            for err in gt.injected_errors:
                per_error_type[err.error_type]["fn"] += 1
                impact = _error_impact(err.error_type, err.affected_fields, doc_type)
                business_impact_missed += impact
                # Check if this was a critical error
                if _is_critical_error(err.affected_fields, doc_type):
                    missed_critical += 1
            if any(_is_critical_error(e.affected_fields, doc_type) for e in gt.injected_errors):
                total_critical += 1
        elif not corrupted and flagged:
            fp += 1
            per_doc_type[doc_type]["fp"] += 1
        else:
            tn += 1
            per_doc_type[doc_type]["tn"] += 1

        if corrupted:
            total_critical += sum(
                1 for e in gt.injected_errors
                if _is_critical_error(e.affected_fields, doc_type)
            )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    false_trust_rate = fn / max(tp + fn, 1)

    results["precision"] = round(precision, 3)
    results["recall"] = round(recall, 3)
    results["f1"] = round(f1, 3)
    results["false_trust_rate"] = round(false_trust_rate, 3)
    results["tp"] = tp
    results["fp"] = fp
    results["fn"] = fn
    results["tn"] = tn
    results["business_impact_caught"] = round(business_impact_caught, 1)
    results["business_impact_missed"] = round(business_impact_missed, 1)

    total_impact = business_impact_caught + business_impact_missed
    results["business_impact_catch_rate"] = (
        round(business_impact_caught / total_impact, 3) if total_impact > 0 else 1.0
    )

    baseline = _random_baseline(reports, ground_truths, n_trials=1000)
    results["baseline_recall"] = baseline["recall"]
    results["baseline_business_catch_rate"] = baseline["business_catch_rate"]

    console.print("\n[bold]═══ Evaluation Results ═══[/bold]\n")

    # Overall metrics
    main_table = Table(title="Overall Detection Performance", show_header=True)
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value")
    main_table.add_row("Precision", f"{precision:.1%}")
    main_table.add_row("Recall", f"{recall:.1%}")
    main_table.add_row("F1", f"{f1:.1%}")
    main_table.add_row("False Trust Rate", f"[{'red' if false_trust_rate > 0.1 else 'green'}]{false_trust_rate:.1%}[/]")
    main_table.add_row("Business Impact Caught", f"{results['business_impact_catch_rate']:.1%}")
    main_table.add_row("", "")
    main_table.add_row("[dim]Baseline (random 10%)[/dim]", "")
    main_table.add_row("  Recall", f"{baseline['recall']:.1%}")
    main_table.add_row("  Business Catch Rate", f"{baseline['business_catch_rate']:.1%}")
    console.print(main_table)

    # Per error type
    if per_error_type:
        err_table = Table(title="\nPer Error Type", show_header=True)
        err_table.add_column("Error Type")
        err_table.add_column("Caught (TP)")
        err_table.add_column("Missed (FN)")
        err_table.add_column("Recall")
        for etype, counts in sorted(per_error_type.items()):
            etp, efn = counts["tp"], counts["fn"]
            erecall = etp / max(etp + efn, 1)
            err_table.add_row(
                etype, str(etp), str(efn),
                f"[{'green' if erecall >= 0.8 else 'red'}]{erecall:.0%}[/]",
            )
        console.print(err_table)

    # Per doc type
    if per_doc_type:
        doc_table = Table(title="\nPer Document Type", show_header=True)
        doc_table.add_column("Doc Type")
        doc_table.add_column("TP")
        doc_table.add_column("FP")
        doc_table.add_column("FN")
        doc_table.add_column("TN")
        for dtype, counts in sorted(per_doc_type.items()):
            doc_table.add_row(
                dtype, str(counts["tp"]), str(counts["fp"]),
                str(counts["fn"]), str(counts["tn"]),
            )
        console.print(doc_table)

    # Routing distribution
    trust = sum(1 for r in reports if r.routing_decision == RoutingDecision.TRUST)
    review = sum(1 for r in reports if r.routing_decision == RoutingDecision.HUMAN_REVIEW)
    reprocess = sum(1 for r in reports if r.routing_decision == RoutingDecision.REPROCESS)
    console.print(f"\n[bold]Routing:[/bold] Trust={trust}  Review={review}  Reprocess={reprocess}")
    console.print()

    return results


def _error_impact(error_type: str, affected_fields: list[str], doc_type: str) -> float:
    """Compute business impact of an error based on field criticality."""
    fc = field_criticality()
    doc_fc = fc.get(doc_type, {})
    total = 0.0
    for field in affected_fields:
        crit = doc_fc.get(field, "medium")
        total += CRITICALITY_WEIGHTS.get(crit, 2)
    # Minimum impact even if no fields matched
    return max(total, CRITICALITY_WEIGHTS.get("medium", 2))


def _is_critical_error(affected_fields: list[str], doc_type: str) -> bool:
    fc = field_criticality()
    doc_fc = fc.get(doc_type, {})
    return any(doc_fc.get(f, "medium") == "critical" for f in affected_fields)


def _random_baseline(
    reports: list[AuditReport],
    ground_truths: list[GroundTruth],
    sample_rate: float = 0.10,
    n_trials: int = 1000,
) -> dict[str, float]:
    """Simulate the legacy random-10% spot-check across many trials."""
    gt_map = {gt.doc_id: gt for gt in ground_truths}
    corrupted_ids = {gt.doc_id for gt in ground_truths if gt.is_corrupted}
    all_ids = [r.doc_id for r in reports]
    n_sample = max(1, int(len(all_ids) * sample_rate))

    rng = random.Random(123)
    total_caught = 0
    total_corrupted = len(corrupted_ids) * n_trials
    total_impact_caught = 0.0
    total_impact = 0.0

    # Compute total possible impact
    for gt in ground_truths:
        if gt.is_corrupted:
            for r in reports:
                if r.doc_id == gt.doc_id:
                    doc_type = r.doc_type.value
                    for err in gt.injected_errors:
                        total_impact += _error_impact(err.error_type, err.affected_fields, doc_type)

    total_impact *= n_trials

    for _ in range(n_trials):
        sampled = set(rng.sample(all_ids, min(n_sample, len(all_ids))))
        caught = sampled & corrupted_ids
        total_caught += len(caught)

        for doc_id in caught:
            gt = gt_map[doc_id]
            for r in reports:
                if r.doc_id == doc_id:
                    for err in gt.injected_errors:
                        total_impact_caught += _error_impact(
                            err.error_type, err.affected_fields, r.doc_type.value
                        )

    return {
        "recall": total_caught / max(total_corrupted, 1),
        "business_catch_rate": total_impact_caught / max(total_impact, 1),
    }
