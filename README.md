# Audit Agent

Risk-aware, non-replicative auditing for LLM-generated financial document outputs.

The prototype simulates a primary document-processing system, injects realistic failure modes, and then audits the primary outputs without fully re-running extraction. For each document it produces:

- a confidence level
- a routing decision: trust, human_review, or reprocess
- a structured report that points the reviewer to the exact field, quote, and reason

The repository is designed to answer the assignment directly:

- how the primary LLM outputs are simulated
- what audit signals are used without re-doing extraction
- how scrutiny differs by document type and complexity
- how uncertainty is handled
- which design decision was reversed after observing real batch behavior

## 0. Assignment Coverage At A Glance

| Assignment ask | Where this repository answers it |
|---|---|
| Simulate primary outputs and injected failures | Synthetic sources in `src/audit_agent/docgen/templates.py`, deterministic primary outputs in `src/audit_agent/primary/extractor.py`, controlled corruption in `src/audit_agent/primary/error_injector.py` |
| Audit quality without fully re-doing extraction | Claim decomposition in `src/audit_agent/audit/claim_parser.py`, deterministic checks in `src/audit_agent/audit/tier1_checks.py`, selective probes in `src/audit_agent/audit/tier2_probes.py` |
| Apply different scrutiny to simple versus complex documents | Complexity scoring in `src/audit_agent/audit/scrutiny.py`, thresholds in `configs/scoring_weights.yaml`, doc-type policies in `configs/doc_type_policies.yaml` |
| Handle uncertainty honestly | Typed abstention in `src/audit_agent/audit/tier2_probes.py`, high-stakes escalation in `src/audit_agent/audit/router.py` |
| Explain one reversed design decision | Section 8 below, tied to the quote-value consistency check now implemented in Tier-1 |

## 1. Architecture

The core design is a tiered audit, not a second extraction pass.

```
Source Document
    │
    ├──► Primary System
    │       └──► ExtractionOutput
    │             ├── structured fields
    │             ├── summary
    │             └── citation contract (raw_quote + char_span)
    │
    └──► Audit Agent
            ├── parse claims
            ├── Tier-1 deterministic checks
            ├── scrutiny scoring
            ├── optional Tier-2 probes
            ├── route: trust / human_review / reprocess
            └── produce analyst-facing report
```

Claims are the unit of audit. Before any checks run, `parse_claims()` converts the primary output into atomic auditable statements:

- each structured field becomes one field claim with its `value`, `raw_quote`, and `char_span`
- the prose summary is split into sentence-level claims

That decomposition matters because the audit agent is not trying to regenerate the whole extraction. It is trying to verify or falsify specific claims cheaply.

The default demo path is offline and deterministic. That matters for the assignment because it gives a fully reproducible prototype with no API dependency. When NIM is available, the same pipeline can optionally run Tier-2 probes with a different audit model.

Current model configuration for live NIM runs:

- primary: stepfun-ai/step-3.5-flash
- audit fallback chain: google/gemma-4-31b-it, google/gemma-3n-e2b-it, google/gemma-3n-e4b-it

That pairing is intentional: the primary path stays cheap, while the audit path uses a slower but stronger fallback chain for selective verification. The live Tier-2 path was validated with real NIM responses, but the recommended assignment demo remains the offline seeded run because it is fully reproducible.

## 2. How Primary Outputs Are Simulated

There is no real production source system in this assignment, so the repository creates one.

### Synthetic source documents

The generator in src/audit_agent/docgen/templates.py creates four document families:

- account opening forms
- investment policy statements
- quarterly portfolio reviews
- compliance disclosures

Each generated document is built from structured template slots such as:

- client identity
- risk tolerance
- investment objective
- time horizon
- asset allocation
- compliance flags
- bespoke constraints

The templates deliberately vary complexity. For example, quarterly reviews and IPS documents can include bespoke clauses and table-heavy structures, while account opening forms are usually simpler.

### Offline primary path

The offline primary path in src/audit_agent/primary/extractor.py reconstructs the primary extraction deterministically from the same template slots used to generate the source document. This is not meant to mimic production exactly. It exists so the assignment prototype can be run, tested, and evaluated end to end with stable seeded behavior.

That is an intentional design tradeoff. The offline extractor is not trying to be a realistic first-pass LLM. Realistic failure patterns are introduced one stage later by the error injector, which lets the repository stay reproducible while still exercising the audit logic on high-cost error classes.

### Citation contract

Every extracted field carries:

- value
- raw_quote
- char_span

That contract is the foundation of the audit layer. It allows the agent to ask: is the value grounded, is the quote present, and does the quote actually support the value?

## 3. Injected Error Types And Why These Ones

The error injector in src/audit_agent/primary/error_injector.py corrupts a configurable subset of primary outputs. The chosen error types are the ones that are both plausible for LLM systems and consequential for financial-services workflows.

| Error type | Why it matters | Expected audit signal |
|---|---|---|
| hallucinated_allocation | Invented portfolio content can directly mislead relationship managers | citation grounding, footing |
| numeric_drift | Small numeric mistakes are easy to miss and high-cost in finance | quote-value consistency, footing |
| logical_footing | Allocations or totals no longer reconcile | footing |
| temporal_misattribution | Dates or horizons move between fields | cross-field checks, optional Tier-2 |
| dropped_compliance_flag | Missing compliance content is high-risk | required fields, compliance policy |
| stale_template_bleed | Wrong-client or wrong-template text appears | type mismatch, optional Tier-2 |
| doc_misclassification | Wrong type changes the required checks and fields | type-schema match |
| schema_valid_garbage | Output is structurally valid but semantically wrong | mostly Tier-2; deterministic path is weaker |

These were chosen over broader “random corruption” because they map to recognizable failure modes a reviewer can understand and a routing policy can act on. The injector applies one or two weighted errors within the corrupted subset so the batch contains both isolated failures and multi-signal failures.

## 4. Audit Signals Without Re-Doing Extraction

The audit agent does not fully re-extract the document. Instead, it uses cheaper, narrower signals.

### Implemented deterministic checks

The table below is the core answer to the assignment's non-replication constraint.

| Signal | What it inspects | Failure indicator | Why it is non-replicative |
|---|---|---|---|
| required fields | output schema against doc-type policy | mandatory field missing | checks presence only; does not infer the missing value |
| citation grounding | raw_quote and char_span against source text | quote absent or span inconsistent | validates cited evidence instead of re-extracting content |
| quote-value consistency | numeric field value versus numeric literals in its own quote | quote and value disagree | parses the claimed evidence rather than solving the full document |
| footing | allocation percentages and dollar totals | weights or totals do not reconcile | uses arithmetic consistency on extracted structure |
| numeric plausibility | scalar ranges such as returns, values, horizons | implausible magnitude or sign | sanity-checks output values only |
| cross-field consistency | related fields such as risk tolerance and allocations | business rule contradiction | tests local coherence across extracted fields |
| type-schema match | source keywords versus predicted doc type | document looks like a different form | validates type plausibility, not full extraction |
| summary-extraction consistency | summary numbers versus extracted fields and source numbers | unsupported summary claim | checks for overreach without regenerating the summary |

These checks are reliable because each one targets a local failure mode directly. They are not trying to solve the whole document again; they are trying to falsify the primary output cheaply.

### Error type to audit signal map

This is the explicit bridge between the injected failure taxonomy and the audit behavior.

| Injected error type | Primary audit signals | Expected routing tendency |
|---|---|---|
| hallucinated_allocation | citation grounding, footing, quote-value consistency | usually `reprocess` |
| numeric_drift | quote-value consistency, sometimes footing | usually `reprocess` |
| logical_footing | footing | `reprocess` |
| temporal_misattribution | cross-field consistency, Tier-2 reverification | often `human_review` |
| dropped_compliance_flag | required fields, missed-compliance Tier-2 probe, policy override | `human_review` |
| stale_template_bleed | type-schema mismatch, Tier-2 reverification | often `human_review`, sometimes missed |
| doc_misclassification | type-schema match plus wrong required-field surface | usually `reprocess` |
| schema_valid_garbage | weak Tier-1 signal, stronger Tier-2 semantic disagreement | usually `human_review` when caught |

### Optional Tier-2 probes

When NIM is enabled, the audit pipeline can also run targeted probes:

- reverify a specific high-stakes claim
- check for missed compliance flags
- test narrative entailment for summary claims

These probes are selective. They are meant to close the gap on semantic failures that deterministic checks cannot resolve well. The pipeline only enters Tier-2 when `scrutiny >= medium` or Tier-1 already found failures. Inside Tier-2, low-scrutiny documents are still allowed to reverify high-stakes claims, but routine low-stakes claims are skipped.

## 5. How Scrutiny Is Decided

The audit does not treat all documents the same.

Scrutiny is based on:

- document-type prior
- structural density: long, table-heavy documents are harder
- regulatory density: AML, KYC, sanctions, suitability, disclosure language
- financial-network density: derivatives, trusts, multi-asset structures, SPVs, and similar complexity terms
- escalation if Tier-1 already found failures

Current scoring formula:

```text
raw_score = 0.30 * doc_type_prior
          + 0.25 * structural_density
          + 0.25 * regulatory_density
          + 0.20 * financial_network_density
```

Then two escalation rules apply:

- any critical claim lifts a `low` document to at least `medium`
- any Tier-1 failure bumps the scrutiny level up by one step

Current thresholds are heuristic rather than learned:

- low: score <= 0.30
- medium: score <= 0.50
- high: score <= 0.70
- critical: score > 0.70

Practical effect:

- simple account-opening documents often remain low or medium scrutiny
- quarterly reviews and IPS documents are more likely to become medium or high scrutiny because they carry allocations, performance numbers, and bespoke constraints
- compliance disclosures are treated as high-stakes and always route to human review by policy

This is deliberately tunable. The values in configs/scoring_weights.yaml are meant to express risk appetite, not claim to be statistically optimal.

## 6. Routing Decisions

The router in src/audit_agent/audit/router.py chooses among three outcomes.

- trust: no meaningful problems found, automation-safe
- human_review: semantic risk, uncertainty, compliance override, or conflicting signals
- reprocess: mechanical failure that is plausibly fixable by rerunning the primary system

Current routing philosophy:

- missing fields, broken citations, and quote/value mismatches are mechanical and reprocessable
- compliance disclosures always require human review
- if Tier-2 disagrees with Tier-1, the system halts and escalates rather than looping automatically

The router applies those ideas in a fixed order:

1. policy override first: compliance disclosures default to `human_review`
2. halt on conflict: if Tier-1 passes but Tier-2 disagrees, escalate to `human_review`
3. critical failures next:
    - reprocessable tags such as missing fields, citation failures, footing errors, quote-value mismatches, and type mismatches go to `reprocess`
    - semantic tags such as missed flags, summary overreach, cross-field violations, and Tier-2 disagreement go to `human_review`
4. high-stakes abstention goes to `human_review`
5. non-critical failures use the same split between mechanical `reprocess` and semantic `human_review`
6. if scrutiny is `high` or `critical` but Tier-2 did not run, the document is still flagged for `human_review`
7. only then does the router return `trust`

This matches the assignment’s requirement that the audit layer should not just say “bad” or “good”; it should decide what to do next.

## 7. How Uncertainty Is Handled

When the agent genuinely cannot determine whether a claim is correct, it can abstain rather than pretending certainty.

Typed abstention is implemented in the Tier-2 probe layer with reasons such as:

- ambiguous_source
- multiple_candidates
- insufficient_evidence
- indirect_support
- unsupported_inference
- source_defect

Routing rule:

- high-stakes abstention routes to human review

Reviewer communication rule:

- the report tells the analyst which claim could not be verified and why the system was uncertain

That behavior is intentional rather than a fallback. In this domain, uncertain-but-high-stakes claims should consume human attention; they should not be silently trusted and they should not always be sent back for blind reprocessing.

The offline Tier-1 demo path is mostly deterministic, so the abstention behavior is more visible when Tier-2 is enabled. That is an honest limitation of the current prototype.

### Worked example: Tier-2 abstention on schema-valid garbage

Source document contains a quarterly portfolio review with 8.5% YTD return and a $1.2M total value. The primary system returns a syntactically valid summary:

> "The client's portfolio demonstrates excellent risk-adjusted returns with a Sharpe ratio exceeding industry benchmarks across all time periods analyzed."

Tier-1 does not find any numbers in the summary, so the number-matching check has nothing to evaluate. The summary length (more than 30 words) with zero specific values on a numeric-heavy document type does trigger a `SUMMARY_OVERREACH` signal, but it is not conclusive on its own.

When Tier-2 runs the narrative entailment probe it asks: is this claim directly supported by the source? The source contains performance numbers and allocation tables but no Sharpe ratio and no benchmark comparison. The probe returns:

```json
{"verdict": "abstain", "reason": "unsupported_inference", "evidence_span": null}
```

Routing: the abstention on a high-stakes claim escalates to `human_review`.

Report output to analyst:

```
Claim: summary sentence 1
Verdict: ABSTAIN
Reason: unsupported_inference — Tier-2 could not find source evidence for this claim.
Action: manually verify whether summary reflects the actual document content.
```

This is the honest answer. The system does not silently trust the output and does not blindly reprocess it — it tells the analyst exactly where the uncertainty is.

## 8. Reversed Design Decision

This was the most important design change made after observing actual batch behavior.

### Original decision

Initially, the deterministic audit relied heavily on citation grounding plus footing. The assumption was that if the quote appeared in the source and the numbers reconciled globally, numeric drift would be caught often enough.

### What happened in practice

That assumption was wrong.

In an earlier run, numeric drift could survive when:

- the extracted scalar value changed
- the raw quote was left untouched
- the changed number did not break any global reconciliation rule

In other words, the quote was real, but the value no longer matched the quote.

### Reversed decision

I added a deterministic quote-value consistency check that parses numbers from raw_quote and compares them to the extracted numeric value.

That materially improved the audit layer because it catches a high-value class of failures without invoking a second LLM. It is exactly the kind of non-replicative signal the assignment is asking for.

Concrete example from the current fixtures:

- source evidence: `$47,882,047.07`
- extracted value: `45,952,353.43`
- citation grounding still passes because the quote is real
- the new quote-value consistency check fails because the extracted scalar no longer matches its own evidence

That is the failure pattern that changed my mind. Citation grounding plus footing looked sufficient on paper, but this specific class of error can survive both when the quote remains genuine and the surrounding structure still looks plausible. In the seeded `n=50` offline run documented below, numeric_drift recall is 4/4 after this reversal.

## 9. Reproducible Results

Seeded run used for the numbers below:

- command: uv run audit-agent demo --n 50 --seed 42
- primary path: offline deterministic extraction
- corruption rate: 30%
- Tier-2: disabled

### Batch metrics

| Metric | Value |
|---|---|
| Precision | 80.0% |
| Recall | 88.9% |
| F1 | 84.2% |
| False Trust Rate | 11.1% |
| Business Impact Catch Rate | 83.8% |
| Baseline random 10% recall | 10.1% |
| Baseline random 10% business catch rate | 10.0% |

Headline result: the audit layer catches 83.8% of business-weighted error impact versus 10.0% for the legacy random 10% spot-check baseline.

These are single-seed prototype metrics, not a production benchmark. They are useful because they are reproducible, not because they estimate a confidence interval.

### Routing distribution on that run

| Route | Count |
|---|---|
| trust | 30 |
| human_review | 11 |
| reprocess | 9 |

### Per-error-type recall

| Error type | Caught | Missed | Recall |
|---|---|---|---|
| doc_misclassification | 5 | 0 | 100% |
| hallucinated_allocation | 4 | 0 | 100% |
| numeric_drift | 4 | 0 | 100% |
| stale_template_bleed | 2 | 0 | 100% |
| schema_valid_garbage | 4 | 1 | 80% |
| temporal_misattribution | 2 | 1 | 67% |

### What the prototype is good at

- mechanical failures with strong local evidence
- hallucinated or altered allocations
- numeric drift that breaks quote-value agreement or reconciliation
- misclassification and missing mandatory structure
- stale template bleed via summary client-name consistency check

### What it is still weak at

- temporal misattribution when the changed value still looks plausible
- anything that requires genuine interpretation rather than contradiction checking

That weakness is the main reason Tier-2 probes exist.

## 10. Running The Project

### Setup

```bash
pip install uv
cd audit-agent
uv sync --group dev
```

### Offline demo

```bash
uv run audit-agent demo --n 50 --seed 42
```

This is the recommended evaluation path for the assignment because it is reproducible and requires no external API.

### Step-by-step offline run

```bash
uv run audit-agent gen-docs --n 50 --seed 42
uv run audit-agent run-primary --offline
uv run audit-agent inject-errors --corruption-rate 0.30
uv run audit-agent audit
uv run audit-agent evaluate
```

### Optional Tier-2 with NIM

```bash
cp .env.example .env
# Fill in NIM_API_KEY if you want live probe runs

uv run audit-agent demo --n 50 --seed 42 --tier2
```

Tier-2 is now configured for slow NIM backends: the client uses a larger timeout, bounded retries, and comma-separated model fallback. The default audit chain is `google/gemma-4-31b-it,google/gemma-3n-e2b-it,google/gemma-3n-e4b-it`.

If NIM is still unreachable after those attempts, the command still completes: probe calls abstain, the report records that the agent could not verify those claims, and routing escalates rather than hanging indefinitely.

For a quick live smoke test, this smaller command is enough:

```bash
uv run audit-agent demo --n 2 --seed 42 --tier2
```

### Tests

```bash
uv run python -m pytest tests/ -v
```

The repository now includes:

- deterministic Tier-1 tests for footing, required fields, citation grounding, numeric ranges, quote-value consistency, summary consistency, and claim parsing
- Tier-2 tests for narrative entailment, claim reverification, missed compliance flags, and abstention on probe failure
- scrutiny and router tests for threshold behavior, critical-claim escalation, Tier-1-failure escalation, conflict handling, and high-scrutiny review gating
- an offline CLI smoke test for the end-to-end demo flow

That coverage is still not the same thing as production calibration, but it does mean the key assignment claims are now exercised directly in tests rather than only described in prose.

## 11. Analyst-Facing Report Design

Each finding tries to answer the reviewer’s actual question: where should I look, and why?

The report includes:

- the affected field or claim
- the failed audit tag
- the supporting quote or source snippet
- the character span when available
- a one-sentence reason
- a short human-review focus list

Example of the intended experience:

- Check total_portfolio_value at chars 132:146: Extracted value does not match numeric evidence in quote. Evidence: $47,882,047.07

Example report excerpt from `fixtures/audit_reports.json`:

```text
DOC_0001 -> HUMAN_REVIEW
Field: total_portfolio_value
Tag: QUOTE_VALUE_MISMATCH
Evidence: $47,882,047.07
Reason: Extracted value 45952353.43 does not match numeric evidence in quote

Field: asset_allocations
Tag: FOOTING_ERROR
Evidence: Equities | $19,344,347.02 | Hedge_Funds 8.1%
Reason: Allocation weights sum to 108.1%, expected ~100% (±2.0%)
```

That is much more useful than “possible error in financial field.”

## 12. Limitations And Calibration

The prototype is intentionally narrow in a few places.

- The offline primary path is deterministic reconstruction, not an attempt to simulate every first-pass LLM behavior.
- Tier-1 is strongest on local contradictions and weakest on temporal misattribution when the changed value still looks plausible.
- Tier-2 improves semantic coverage, but it introduces latency and model-specific judgment noise, especially around rounding and narrative phrasing.
- Scrutiny thresholds and citation fuzz thresholds are heuristics. They are easy to tune, but they are not learned from production data.

Those limitations are acceptable for this assignment because the goal is to demonstrate a credible audit design, not claim production calibration.

### Precision versus recall

The seeded run shows precision of 80.0%, which means roughly 20% of flagged documents are false alarms (most of those are compliance disclosures blanket-escalated by policy). In this domain, a missed real error — a relationship manager walking into a client meeting with a hallucinated allocation or a dropped compliance flag — is materially more costly than a false flag that an analyst clears in thirty seconds. The current thresholds in `configs/scoring_weights.yaml` are set to prefer recall over precision. That tradeoff is explicit and tunable. A firm with a higher analyst cost or a lower tolerance for false-alarm fatigue would shift the thresholds; the scoring formula and routing logic make that adjustment straightforward without changing any code.

## 13. Future Work

The next improvements that matter most are:

1. stronger semantic checks for stale_template_bleed and schema_valid_garbage
2. better report rendering for side-by-side source versus claim comparison
3. reduce Tier-2 false positives on harmless rounding and summary paraphrase cases

## 14. Project Structure

```
audit-agent/
├── LICENSE
├── pyproject.toml
├── configs/
├── fixtures/
├── src/audit_agent/
│   ├── __init__.py
│   ├── runtime.py
│   ├── cli.py
│   ├── config.py
│   ├── evaluate.py
│   ├── nim_client.py
│   ├── schemas.py
│   ├── audit/
│   ├── docgen/
│   └── primary/
└── tests/
    ├── test_cli_demo.py
    ├── test_nim_client.py
    ├── test_report.py
    ├── test_scrutiny_router.py
    ├── test_tier1.py
    └── test_tier2_probes.py
```