"""CLI — Typer-based interface for the audit agent.

Commands:
  gen-docs         Generate synthetic documents
  run-primary      Run primary extraction (offline deterministic)
  inject-errors    Inject errors into primary outputs
  audit            Run the audit pipeline on primary outputs
  evaluate         Evaluate audit results against ground truth
  demo             Chain all steps end-to-end
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from audit_agent.schemas import ExtractionOutput, GroundTruth
from audit_agent.runtime import FIXTURES_DIR

app = typer.Typer(
    name="audit-agent",
    help="AI agent that audits the outputs of a primary document-processing LLM.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def _clear_glob(directory: Path, pattern: str) -> None:
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


@app.command()
def gen_docs(
    n: int = typer.Option(50, help="Number of documents to generate"),
    seed: int = typer.Option(42, help="Random seed"),
    out_dir: Optional[str] = typer.Option(None, help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate synthetic financial documents."""
    _setup_logging(verbose)
    from audit_agent.docgen.templates import generate_batch

    docs = generate_batch(n=n, seed=seed)
    target = Path(out_dir) if out_dir else FIXTURES_DIR / "sources"
    target.mkdir(parents=True, exist_ok=True)
    _clear_glob(target, "*.md")

    for doc in docs:
        (target / f"{doc['doc_id']}.md").write_text(doc["text"])

    console.print(f"[green]Generated {len(docs)} documents → {target}[/green]")

    # Also save metadata for primary step
    meta_path = target.parent / "doc_metadata.json"
    meta = [
        {
            "doc_id": d["doc_id"],
            "doc_type": d["doc_type"],
            "slots": d["slots"].to_dict(),
        }
        for d in docs
    ]
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"[dim]Metadata → {meta_path}[/dim]")


@app.command()
def run_primary(
    source_dir: Optional[str] = typer.Option(None, help="Source documents directory"),
    out_dir: Optional[str] = typer.Option(None, help="Output directory"),
    offline: bool = typer.Option(True, help="Use deterministic offline extraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the primary LLM extraction (offline or via NIM)."""
    _setup_logging(verbose)
    from audit_agent.primary.extractor import extract_document, extract_document_offline

    src = Path(source_dir) if source_dir else FIXTURES_DIR / "sources"
    out = Path(out_dir) if out_dir else FIXTURES_DIR / "primary_outputs"
    out.mkdir(parents=True, exist_ok=True)
    _clear_glob(out, "*.json")

    meta_path = src.parent / "doc_metadata.json"
    if not meta_path.exists():
        console.print("[red]No doc_metadata.json found. Run gen-docs first.[/red]")
        raise typer.Exit(1)

    metadata = json.loads(meta_path.read_text())

    for doc_meta in metadata:
        doc_id = doc_meta["doc_id"]
        source_path = src / f"{doc_id}.md"
        if not source_path.exists():
            continue
        source_text = source_path.read_text()

        if offline:
            extraction = extract_document_offline(
                doc_id=doc_id,
                doc_type=doc_meta["doc_type"],
                text=source_text,
                slots=doc_meta.get("slots", {}),
            )
        else:
            from audit_agent.nim_client import NIMClient
            client = NIMClient()
            model = os.environ.get("PRIMARY_MODEL", "stepfun-ai/step-3.5-flash")
            extraction = extract_document(client, model, doc_id, doc_meta["doc_type"], source_text)

        (out / f"{doc_id}.json").write_text(extraction.model_dump_json(indent=2))

    console.print(f"[green]Extracted {len(metadata)} documents → {out}[/green]")


@app.command()
def inject_errors(
    in_dir: Optional[str] = typer.Option(None, help="Primary outputs directory"),
    corruption_rate: float = typer.Option(0.30, help="Fraction of docs to corrupt"),
    max_per_doc: int = typer.Option(2, help="Max errors per document"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Inject controlled errors into primary LLM outputs."""
    _setup_logging(verbose)
    import random as _random
    from audit_agent.primary.error_injector import inject_errors as _inject

    target = Path(in_dir) if in_dir else FIXTURES_DIR / "primary_outputs"
    source_dir = FIXTURES_DIR / "sources"
    rng = _random.Random(seed)

    injected_count = 0
    label_count = 0

    for path in sorted(target.glob("*.json")):
        if path.name.endswith(".labels.json"):
            continue
        extraction = ExtractionOutput.model_validate_json(path.read_text())
        source_path = source_dir / f"{extraction.doc_id}.md"
        source_text = source_path.read_text() if source_path.exists() else ""

        mutated, ground_truth = _inject(
            extraction, source_text, rng,
            corruption_rate=corruption_rate,
            max_errors_per_doc=max_per_doc,
        )

        # Overwrite with mutated version
        path.write_text(mutated.model_dump_json(indent=2))
        # Save ground truth labels
        labels_path = target / f"{extraction.doc_id}.labels.json"
        labels_path.write_text(ground_truth.model_dump_json(indent=2))

        if ground_truth.is_corrupted:
            injected_count += 1
        label_count += 1

    console.print(
        f"[green]Injected errors in {injected_count}/{label_count} documents[/green]"
    )


@app.command()
def audit(
    source_dir: Optional[str] = typer.Option(None, help="Source documents directory"),
    primary_dir: Optional[str] = typer.Option(None, help="Primary outputs directory"),
    out: Optional[str] = typer.Option(None, help="Output report JSON path"),
    tier2: bool = typer.Option(False, help="Enable Tier-2 LLM probes (requires NIM)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the audit pipeline on primary extraction outputs."""
    _setup_logging(verbose)
    from audit_agent.audit.pipeline import audit_batch
    from audit_agent.audit.report import render_report, render_batch_summary

    src = Path(source_dir) if source_dir else FIXTURES_DIR / "sources"
    prim = Path(primary_dir) if primary_dir else FIXTURES_DIR / "primary_outputs"

    # Load sources
    sources = {}
    for p in sorted(src.glob("*.md")):
        sources[p.stem] = p.read_text()

    # Load extractions
    extractions = []
    for p in sorted(prim.glob("*.json")):
        if p.name.endswith(".labels.json"):
            continue
        extractions.append(ExtractionOutput.model_validate_json(p.read_text()))

    # Setup NIM client for Tier-2
    client = None
    if tier2:
        from audit_agent.nim_client import NIMClient
        client = NIMClient()

    reports = audit_batch(sources, extractions, client, run_tier2_probes=tier2)

    for report in reports:
        render_report(report, console)
    render_batch_summary(reports, console)

    # Save reports
    out_path = Path(out) if out else FIXTURES_DIR / "audit_reports.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        [r.model_dump(mode="json") for r in reports], indent=2,
    ))
    console.print(f"[dim]Reports → {out_path}[/dim]")


@app.command()
def evaluate_cmd(
    primary_dir: Optional[str] = typer.Option(None, help="Directory with labels.json"),
    reports_path: Optional[str] = typer.Option(None, help="Audit reports JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate audit results against ground truth labels."""
    _setup_logging(verbose)
    from audit_agent.evaluate import evaluate as _evaluate
    from audit_agent.schemas import AuditReport

    prim = Path(primary_dir) if primary_dir else FIXTURES_DIR / "primary_outputs"
    rpath = Path(reports_path) if reports_path else FIXTURES_DIR / "audit_reports.json"

    # Load ground truths
    ground_truths = []
    for p in sorted(prim.glob("*.labels.json")):
        ground_truths.append(GroundTruth.model_validate_json(p.read_text()))

    # Load reports
    raw = json.loads(rpath.read_text())
    reports = [AuditReport.model_validate(r) for r in raw]

    results = _evaluate(reports, ground_truths, console)
    console.print(f"\n[bold green]Business Impact Catch Rate: {results['business_impact_catch_rate']:.1%}[/bold green]")


@app.command()
def demo(
    n: int = typer.Option(50, help="Number of documents"),
    seed: int = typer.Option(42, help="Random seed"),
    corruption_rate: float = typer.Option(0.30, help="Error injection rate"),
    tier2: bool = typer.Option(False, help="Enable Tier-2 probes"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full demo: generate → extract → inject → audit → evaluate."""
    _setup_logging(verbose)

    console.print("[bold]═══ Step 1: Generate synthetic documents ═══[/bold]")
    gen_docs(n=n, seed=seed, out_dir=None, verbose=verbose)

    console.print("\n[bold]═══ Step 2: Run primary extraction (offline) ═══[/bold]")
    run_primary(source_dir=None, out_dir=None, offline=True, verbose=verbose)

    console.print("\n[bold]═══ Step 3: Inject errors ═══[/bold]")
    inject_errors(in_dir=None, corruption_rate=corruption_rate, max_per_doc=2, seed=seed, verbose=verbose)

    console.print("\n[bold]═══ Step 4: Audit pipeline ═══[/bold]")
    audit(source_dir=None, primary_dir=None, out=None, tier2=tier2, verbose=verbose)

    console.print("\n[bold]═══ Step 5: Evaluate ═══[/bold]")
    evaluate_cmd(primary_dir=None, reports_path=None, verbose=verbose)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
