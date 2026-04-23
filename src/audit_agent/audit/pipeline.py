"""Pipeline orchestrator — runs the full audit on a single document or a batch.

Ties together: claim parser → Tier-1 checks → scrutiny → (optional) Tier-2 → router → report.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from audit_agent.audit.claim_parser import parse_claims
from audit_agent.audit.report import build_report
from audit_agent.audit.router import route
from audit_agent.audit.scrutiny import compute_scrutiny
from audit_agent.audit.tier1_checks import run_all_tier1
from audit_agent.audit.tier2_probes import run_tier2
from audit_agent.nim_client import NIMClient
from audit_agent.runtime import FIXTURES_DIR
from audit_agent.schemas import (
    AuditReport,
    ExtractionOutput,
    ScrutinyLevel,
)

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_MODELS = "google/gemma-4-31b-it,google/gemma-3n-e2b-it,google/gemma-3n-e4b-it"


def audit_single(
    source_text: str,
    extraction: ExtractionOutput,
    client: NIMClient | None = None,
    audit_model: str | None = None,
    run_tier2_probes: bool = True,
) -> AuditReport:
    """Run the full audit pipeline on a single document.

    If client is None, only Tier-1 (deterministic) checks run.
    """
    doc_type = extraction.predicted_doc_type
    audit_model = audit_model or os.environ.get("AUDIT_MODEL", DEFAULT_AUDIT_MODELS)
    if audit_model.strip() == "minimaxai/minimax-m2.7":
        audit_model = DEFAULT_AUDIT_MODELS

    # Step 1: Parse claims
    claims = parse_claims(extraction)
    logger.info("[%s] Parsed %d claims (%d field, %d summary)",
                extraction.doc_id, len(claims),
                sum(1 for c in claims if c.source.value == "structured_field"),
                sum(1 for c in claims if c.source.value == "summary"))

    # Step 2: Tier-1 deterministic checks
    tier1_results = run_all_tier1(source_text, extraction, claims)
    tier1_fails = sum(1 for r in tier1_results if r.status.value == "fail")
    logger.info("[%s] Tier-1: %d checks, %d failures",
                extraction.doc_id, len(tier1_results), tier1_fails)

    # Step 3: Compute scrutiny
    scrutiny = compute_scrutiny(source_text, extraction, claims, tier1_results)
    logger.info("[%s] Scrutiny: %s", extraction.doc_id, scrutiny.value)

    # Step 4: Tier-2 probes (if client available and warranted)
    tier2_results = []
    tier2_ran = False
    if client and run_tier2_probes:
        should_run_tier2 = (
            scrutiny in (ScrutinyLevel.MEDIUM, ScrutinyLevel.HIGH, ScrutinyLevel.CRITICAL)
            or tier1_fails > 0
        )
        if should_run_tier2:
            tier2_results = run_tier2(
                client, audit_model, source_text, claims,
                scrutiny, doc_type.value,
            )
            tier2_ran = len(tier2_results) > 0
            logger.info("[%s] Tier-2: %d probe results", extraction.doc_id, len(tier2_results))

    # Step 5: Route
    all_checks = tier1_results + tier2_results
    routing, verdicts, headline_reasons = route(doc_type, scrutiny, claims, all_checks)
    logger.info("[%s] Routing: %s", extraction.doc_id, routing.value)

    # Step 6: Build report
    report = build_report(
        doc_id=extraction.doc_id,
        doc_type=doc_type,
        scrutiny=scrutiny,
        routing=routing,
        verdicts=verdicts,
        headline_reasons=headline_reasons,
        source_text=source_text,
        tier2_ran=tier2_ran,
    )
    return report


def audit_batch(
    sources: dict[str, str],
    extractions: list[ExtractionOutput],
    client: NIMClient | None = None,
    audit_model: str | None = None,
    run_tier2_probes: bool = True,
) -> list[AuditReport]:
    """Run the audit pipeline on a batch of documents."""
    reports = []
    for ext in extractions:
        source_text = sources.get(ext.doc_id, "")
        if not source_text:
            logger.warning("[%s] No source text found, skipping", ext.doc_id)
            continue
        report = audit_single(
            source_text, ext, client, audit_model, run_tier2_probes,
        )
        reports.append(report)
    return reports


def save_fixtures(
    docs: list[dict[str, Any]],
    extractions: list[ExtractionOutput],
    ground_truths: list[Any],
) -> None:
    sources_dir = FIXTURES_DIR / "sources"
    outputs_dir = FIXTURES_DIR / "primary_outputs"
    sources_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        path = sources_dir / f"{doc['doc_id']}.md"
        path.write_text(doc["text"])

    for ext in extractions:
        path = outputs_dir / f"{ext.doc_id}.json"
        path.write_text(ext.model_dump_json(indent=2))

    for gt in ground_truths:
        path = outputs_dir / f"{gt.doc_id}.labels.json"
        path.write_text(gt.model_dump_json(indent=2))


def load_fixtures() -> tuple[dict[str, str], list[ExtractionOutput], list[Any]]:
    from audit_agent.schemas import GroundTruth

    sources_dir = FIXTURES_DIR / "sources"
    outputs_dir = FIXTURES_DIR / "primary_outputs"

    sources = {}
    for p in sorted(sources_dir.glob("*.md")):
        sources[p.stem] = p.read_text()

    extractions = []
    for p in sorted(outputs_dir.glob("*.json")):
        if p.name.endswith(".labels.json"):
            continue
        extractions.append(ExtractionOutput.model_validate_json(p.read_text()))

    ground_truths = []
    for p in sorted(outputs_dir.glob("*.labels.json")):
        ground_truths.append(GroundTruth.model_validate_json(p.read_text()))

    return sources, extractions, ground_truths
