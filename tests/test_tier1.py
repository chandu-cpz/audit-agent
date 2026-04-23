"""Tests for Tier-1 deterministic checks.

Hand-crafted inputs with known answers — no LLM, no NIM API needed.
"""

from __future__ import annotations

from audit_agent.schemas import (
    AuditTag,
    CheckResult,
    CheckStatus,
    Claim,
    ClaimSource,
    Criticality,
    DocType,
    RoutingDecision,
    ScrutinyLevel,
)
from audit_agent.audit.claim_parser import parse_claims
from audit_agent.audit.tier1_checks import (
    check_citation_grounding,
    check_cross_field_consistency,
    check_footing,
    check_numeric_ranges,
    check_quote_value_consistency,
    check_required_fields,
    check_summary_client_name_consistency,
    check_summary_extraction_consistency,
    run_all_tier1,
)
from audit_agent.testing import make_extraction as _make_extraction
from audit_agent.testing import make_field as _make_field


class TestFooting:
    def test_allocation_sums_to_100_passes(self):
        ext = _make_extraction(fields=[
            _make_field("asset_allocations", {"stocks": 60, "bonds": 30, "cash": 10}),
        ])
        results = check_footing(ext)
        assert any(r.status == CheckStatus.PASS and r.check_name == "footing_pct" for r in results)

    def test_allocation_sums_to_95_fails(self):
        ext = _make_extraction(fields=[
            _make_field("asset_allocations", {"stocks": 60, "bonds": 25, "cash": 10}),
        ])
        results = check_footing(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL and r.check_name == "footing_pct"]
        assert len(fails) == 1
        assert fails[0].tag == AuditTag.FOOTING_ERROR

    def test_allocation_within_tolerance(self):
        """101.5% is within 2% tolerance."""
        ext = _make_extraction(fields=[
            _make_field("asset_allocations", {"stocks": 60, "bonds": 31.5, "cash": 10}),
        ])
        results = check_footing(ext)
        pct_results = [r for r in results if r.check_name == "footing_pct"]
        assert all(r.status == CheckStatus.PASS for r in pct_results)

    def test_dollar_footing_matches(self):
        ext = _make_extraction(fields=[
            _make_field("total_portfolio_value", 1_000_000),
            _make_field("asset_allocations", {
                "stocks": {"weight": 60, "value": 600_000},
                "bonds": {"weight": 30, "value": 300_000},
                "cash": {"weight": 10, "value": 100_000},
            }),
        ])
        results = check_footing(ext)
        dollar = [r for r in results if r.check_name == "footing_dollar"]
        assert len(dollar) == 1
        assert dollar[0].status == CheckStatus.PASS

    def test_dollar_footing_mismatch(self):
        ext = _make_extraction(fields=[
            _make_field("total_portfolio_value", 1_000_000),
            _make_field("asset_allocations", {
                "stocks": {"weight": 60, "value": 600_000},
                "bonds": {"weight": 30, "value": 200_000},  # wrong — should be 300k
                "cash": {"weight": 10, "value": 100_000},
            }),
        ])
        results = check_footing(ext)
        dollar = [r for r in results if r.check_name == "footing_dollar"]
        assert len(dollar) == 1
        assert dollar[0].status == CheckStatus.FAIL
        assert dollar[0].tag == AuditTag.FOOTING_ERROR

    def test_no_allocation_field_no_error(self):
        """If there's no allocation field, footing check shouldn't fail."""
        ext = _make_extraction(fields=[
            _make_field("client_name", "Jane Doe"),
        ])
        results = check_footing(ext)
        assert not any(r.status == CheckStatus.FAIL for r in results)


class TestRequiredFields:
    def test_missing_field_detected(self):
        ext = _make_extraction(
            doc_type=DocType.ACCOUNT_OPENING,
            fields=[_make_field("client_name", "John Doe")],
        )
        results = check_required_fields(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        # account_opening requires many fields — some must be missing
        assert len(fails) > 0
        assert all(r.tag == AuditTag.MISSING_REQUIRED_FIELD for r in fails)

    def test_all_required_present(self):
        fields = [
            _make_field("total_portfolio_value", 1_000_000),
            _make_field("asset_allocations", {"stocks": 60, "bonds": 30, "cash": 10}),
            _make_field("performance_ytd", 8.5),
            _make_field("performance_qtd", 2.1),
            _make_field("benchmark_comparison", "vs S&P 500"),
            _make_field("risk_metrics", "Sharpe 1.2"),
            _make_field("strategy_changes", "none"),
            _make_field("fees_charged", 15000),
            _make_field("commentary", "Strong quarter"),
        ]
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=fields,
        )
        results = check_required_fields(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 0

    def test_investment_policy_statement_missing_fields_detected(self):
        ext = _make_extraction(
            doc_type=DocType.INVESTMENT_POLICY_STATEMENT,
            fields=[
                _make_field("client_name", "John Doe"),
                _make_field("risk_tolerance", "moderate"),
            ],
        )
        results = check_required_fields(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) >= 1
        assert all(r.tag == AuditTag.MISSING_REQUIRED_FIELD for r in fails)

    def test_compliance_disclosure_required_fields_all_present(self):
        ext = _make_extraction(
            doc_type=DocType.COMPLIANCE_DISCLOSURE,
            fields=[
                _make_field("pep_flag", False),
                _make_field("sanctions_screening", "clear"),
                _make_field("aml_status", "cleared"),
                _make_field("kyc_status", "complete"),
                _make_field("suitability_confirmed", True),
                _make_field("risk_disclosure_acknowledged", True),
            ],
        )
        results = check_required_fields(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 0


class TestCitationGrounding:
    def test_exact_match_passes(self):
        source = "The client's name is Jane Doe and she has $1,000,000 in assets."
        claim = Claim(
            claim_id="test_field_client_name",
            doc_id="test",
            source=ClaimSource.STRUCTURED_FIELD,
            field_name="client_name",
            statement="client_name = Jane Doe",
            value="Jane Doe",
            raw_quote="The client's name is Jane Doe",
            char_span=(0, 30),
            criticality=Criticality.MEDIUM,
        )
        results = check_citation_grounding(source, [claim])
        passes = [r for r in results if r.status == CheckStatus.PASS]
        assert len(passes) >= 1

    def test_fabricated_quote_fails(self):
        source = "The client's name is Jane Doe."
        claim = Claim(
            claim_id="test_field_client_name",
            doc_id="test",
            source=ClaimSource.STRUCTURED_FIELD,
            field_name="client_name",
            statement="client_name = Jane Doe",
            value="Jane Doe",
            raw_quote="John Smith has an account with us",
            char_span=(0, 33),
            criticality=Criticality.MEDIUM,
        )
        results = check_citation_grounding(source, [claim])
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) >= 1
        assert any(r.tag == AuditTag.CITATION_NOT_FOUND for r in fails)

    def test_summary_claims_skipped(self):
        source = "Some text."
        claim = Claim(
            claim_id="test_summary_0",
            doc_id="test",
            source=ClaimSource.SUMMARY,
            statement="The portfolio performed well.",
            criticality=Criticality.MEDIUM,
        )
        results = check_citation_grounding(source, [claim])
        # Summary claims don't have raw_quote, should produce nothing or abstain
        assert not any(r.status == CheckStatus.FAIL for r in results)

    def test_no_liquidity_field_with_grounded_quote_passes(self):
        """IPS documents with liquidity_needs='none' should not false-positive."""
        source = "The primary investment objective is long-term capital appreciation."
        claim = Claim(
            claim_id="test_field_liquidity_needs",
            doc_id="test",
            source=ClaimSource.STRUCTURED_FIELD,
            field_name="liquidity_needs",
            statement="liquidity_needs = none",
            value="none",
            raw_quote="long-term capital appreciation",
            char_span=(42, 72),
            criticality=Criticality.HIGH,
        )
        results = check_citation_grounding(source, [claim])
        assert all(r.status == CheckStatus.PASS for r in results)


class TestNumericRanges:
    def test_negative_portfolio_value_fails(self):
        ext = _make_extraction(fields=[
            _make_field("total_portfolio_value", -500_000),
        ])
        results = check_numeric_ranges(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 1
        assert fails[0].tag == AuditTag.NUMERIC_RANGE_VIOLATION

    def test_normal_values_pass(self):
        ext = _make_extraction(fields=[
            _make_field("total_portfolio_value", 1_000_000),
            _make_field("annual_income", 150_000),
        ])
        results = check_numeric_ranges(ext)
        assert not any(r.status == CheckStatus.FAIL for r in results)


class TestQuoteValueConsistency:
    def test_numeric_drift_against_quote_fails(self):
        claim = Claim(
            claim_id="test_field_total_portfolio_value",
            doc_id="test",
            source=ClaimSource.STRUCTURED_FIELD,
            field_name="total_portfolio_value",
            statement="total_portfolio_value = 1050000",
            value=1_050_000,
            raw_quote="Total Portfolio Value: $1,000,000.00",
            char_span=(0, 36),
            criticality=Criticality.HIGH,
        )
        results = check_quote_value_consistency([claim])
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 1
        assert fails[0].tag == AuditTag.QUOTE_VALUE_MISMATCH

    def test_matching_quote_passes(self):
        claim = Claim(
            claim_id="test_field_time_horizon",
            doc_id="test",
            source=ClaimSource.STRUCTURED_FIELD,
            field_name="time_horizon",
            statement="time_horizon = 10",
            value=10,
            raw_quote="Time Horizon: 10 years",
            char_span=(0, 22),
            criticality=Criticality.HIGH,
        )
        results = check_quote_value_consistency([claim])
        assert any(r.status == CheckStatus.PASS for r in results)


class TestSummaryConsistency:
    def test_summary_numbers_supported_by_source_do_not_fail(self):
        source = "Q1 2026 portfolio review. YTD Return: -10.91%."
        ext = _make_extraction(
            fields=[
                _make_field("performance_ytd", -10.91, "YTD Return: -10.91%", (25, 44)),
            ],
            summary="Q1 2026 portfolio review with YTD return of -10.91%.",
        )
        claims = parse_claims(ext)
        results = check_summary_extraction_consistency(source, ext, claims)
        assert not any(r.status == CheckStatus.FAIL for r in results)

    def test_summary_number_missing_from_fields_and_source_fails(self):
        source = "Portfolio review for Q1 2026."
        ext = _make_extraction(
            fields=[_make_field("performance_ytd", 8.5, "YTD Return: 8.5%", (0, 16))],
            summary="Portfolio review for Q1 2026 with YTD return of 11.2%.",
        )
        claims = parse_claims(ext)
        results = check_summary_extraction_consistency(source, ext, claims)
        assert any(r.status == CheckStatus.FAIL for r in results)


class TestRouter:
    def test_clean_low_scrutiny_trusts(self):
        from audit_agent.audit.router import route

        claims = [
            Claim(
                claim_id="test_field_client_name", doc_id="test",
                source=ClaimSource.STRUCTURED_FIELD, field_name="client_name",
                statement="client_name = Jane", criticality=Criticality.LOW,
            ),
        ]
        checks_typed = [CheckResult(
            check_name="required_fields", claim_id="test_field_client_name",
            status=CheckStatus.PASS, reason="OK",
        )]
        decision, verdicts, reasons = route(
            DocType.ACCOUNT_OPENING, ScrutinyLevel.LOW, claims, checks_typed,
        )
        assert decision == RoutingDecision.TRUST

    def test_compliance_always_human_review(self):
        from audit_agent.audit.router import route

        claims = [
            Claim(
                claim_id="test_field_pep_flag", doc_id="test",
                source=ClaimSource.STRUCTURED_FIELD, field_name="pep_flag",
                statement="pep_flag = false", criticality=Criticality.CRITICAL,
            ),
        ]
        checks = [CheckResult(
            check_name="required_fields", claim_id="test_field_pep_flag",
            status=CheckStatus.PASS, reason="OK",
        )]
        decision, verdicts, reasons = route(
            DocType.COMPLIANCE_DISCLOSURE, ScrutinyLevel.LOW, claims, checks,
        )
        assert decision == RoutingDecision.HUMAN_REVIEW

    def test_critical_failure_escalates(self):
        from audit_agent.audit.router import route

        claims = [
            Claim(
                claim_id="test_field_total_portfolio_value", doc_id="test",
                source=ClaimSource.STRUCTURED_FIELD, field_name="total_portfolio_value",
                statement="total_portfolio_value = 1000000", criticality=Criticality.CRITICAL,
            ),
        ]
        checks = [CheckResult(
            check_name="footing_dollar", claim_id="test_field_total_portfolio_value",
            status=CheckStatus.FAIL,
            tag=AuditTag.FOOTING_ERROR,
            reason="Dollar sum doesn't match",
        )]
        decision, verdicts, reasons = route(
            DocType.QUARTERLY_PORTFOLIO_REVIEW, ScrutinyLevel.MEDIUM, claims, checks,
        )
        assert decision in (RoutingDecision.HUMAN_REVIEW, RoutingDecision.REPROCESS)


class TestClaimParser:
    def test_field_claims_created(self):
        ext = _make_extraction(
            doc_type=DocType.ACCOUNT_OPENING,
            fields=[
                _make_field("client_name", "Jane Doe"),
                _make_field("risk_tolerance", "aggressive"),
            ],
            summary="Jane Doe opened an aggressive account.",
        )
        claims = parse_claims(ext)
        field_claims = [c for c in claims if c.source == ClaimSource.STRUCTURED_FIELD]
        summary_claims = [c for c in claims if c.source == ClaimSource.SUMMARY]
        assert len(field_claims) == 2
        assert len(summary_claims) >= 1

    def test_claim_ids_unique(self):
        ext = _make_extraction(
            doc_type=DocType.ACCOUNT_OPENING,
            fields=[
                _make_field("client_name", "Jane Doe"),
                _make_field("risk_tolerance", "aggressive"),
            ],
            summary="Jane Doe opened an aggressive growth account. She is a high-net-worth investor.",
        )
        claims = parse_claims(ext)
        ids = [c.claim_id for c in claims]
        assert len(ids) == len(set(ids)), "Claim IDs must be unique"


class TestTier1Integration:
    def test_clean_doc_mostly_passes(self):
        """A well-formed doc should have mostly pass results."""
        source = (
            "Client: Jane Doe | Risk: moderate | Objective: growth | Account Type: individual\n"
            "DOB: 1980-01-15 | Citizenship: US | Employment: employed\n"
            "Annual Income: $150,000 | Net Worth: $2,000,000 | Source: salary\n"
            "PEP Status: No"
        )
        ext = _make_extraction(
            doc_type=DocType.ACCOUNT_OPENING,
            fields=[
                _make_field("client_name", "Jane Doe", "Client: Jane Doe", (0, 18)),
                _make_field("risk_tolerance", "moderate", "Risk: moderate", (21, 35)),
                _make_field("investment_objective", "growth", "Objective: growth", (38, 55)),
                _make_field("account_type", "individual", "Account Type: individual", (58, 82)),
                _make_field("date_of_birth", "1980-01-15", "DOB: 1980-01-15", (83, 98)),
                _make_field("citizenship", "US", "Citizenship: US", (101, 116)),
                _make_field("employment_status", "employed", "Employment: employed", (119, 138)),
                _make_field("annual_income", 150_000, "Annual Income: $150,000", (139, 162)),
                _make_field("net_worth", 2_000_000, "Net Worth: $2,000,000", (165, 186)),
                _make_field("source_of_funds", "salary", "Source: salary", (189, 203)),
                _make_field("pep_status", "No", "PEP Status: No", (204, 219)),
            ],
            summary="Jane Doe opened an individual account with moderate risk tolerance.",
        )
        claims = parse_claims(ext)
        results = run_all_tier1(source, ext, claims)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        # Should be mostly clean — a few minor issues possible
        assert len(fails) <= 3, f"Too many failures for a clean doc: {[f.reason for f in fails]}"


class TestSummaryClientName:
    def test_stale_name_in_summary_fails(self):
        """Bleed name in summary that doesn't match extracted client_name."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("client_name", "Margaret Chen")],
            summary="Portfolio review for Alexander Thornberry shows strong performance.",
        )
        claims = parse_claims(ext)
        results = check_summary_client_name_consistency("Client: Margaret Chen", ext, claims)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 1
        assert fails[0].tag == AuditTag.SUMMARY_CLIENT_MISMATCH
        assert "Alexander Thornberry" in fails[0].reason

    def test_correct_client_name_in_summary_passes(self):
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("client_name", "Margaret Chen")],
            summary="Margaret Chen's portfolio delivered 8.5% YTD.",
        )
        claims = parse_claims(ext)
        results = check_summary_client_name_consistency("Client: Margaret Chen", ext, claims)
        assert not any(r.status == CheckStatus.FAIL for r in results)

    def test_no_client_name_field_uses_source_path(self):
        """When no client_name field, bleed name not in source should fail."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("total_portfolio_value", 1_000_000)],
            summary="Portfolio review for Alexander Thornberry shows strong performance.",
        )
        claims = parse_claims(ext)
        # Source does not contain Alexander Thornberry — flag it
        results = check_summary_client_name_consistency(
            "Total Portfolio Value: $1,000,000", ext, claims
        )
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) == 1
        assert fails[0].tag == AuditTag.SUMMARY_CLIENT_MISMATCH

    def test_name_in_summary_also_in_source_passes(self):
        """Name in summary that also appears in source is legitimate."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("total_portfolio_value", 1_000_000)],
            summary="Portfolio review for Margaret Chen shows strong performance.",
        )
        claims = parse_claims(ext)
        results = check_summary_client_name_consistency(
            "Client: Margaret Chen | Total Portfolio Value: $1,000,000", ext, claims
        )
        assert not any(r.status == CheckStatus.FAIL for r in results)

    def test_document_term_words_not_flagged(self):
        """'Quarterly Portfolio' in summary should not be treated as a person name."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("client_name", "Margaret Chen")],
            summary="Quarterly Portfolio Review for Q1 2026.",
        )
        claims = parse_claims(ext)
        results = check_summary_client_name_consistency(
            "Client: Margaret Chen", ext, claims
        )
        assert not any(r.status == CheckStatus.FAIL for r in results)


class TestCrossFieldTimeHorizon:
    def test_aggressive_risk_very_short_horizon_fails(self):
        ext = _make_extraction(
            doc_type=DocType.INVESTMENT_POLICY_STATEMENT,
            fields=[
                _make_field("risk_tolerance", "aggressive"),
                _make_field("time_horizon", 2),
            ],
        )
        results = check_cross_field_consistency(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert any(r.tag == AuditTag.CROSS_FIELD_VIOLATION for r in fails)
        assert any("time_horizon" in r.claim_id for r in fails)

    def test_conservative_risk_very_long_horizon_fails(self):
        ext = _make_extraction(
            doc_type=DocType.INVESTMENT_POLICY_STATEMENT,
            fields=[
                _make_field("risk_tolerance", "conservative"),
                _make_field("time_horizon", 25),
            ],
        )
        results = check_cross_field_consistency(ext)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert any(r.tag == AuditTag.CROSS_FIELD_VIOLATION for r in fails)
        assert any("time_horizon" in r.claim_id for r in fails)

    def test_moderate_risk_any_horizon_passes(self):
        ext = _make_extraction(
            doc_type=DocType.INVESTMENT_POLICY_STATEMENT,
            fields=[
                _make_field("risk_tolerance", "moderate"),
                _make_field("time_horizon", 10),
            ],
        )
        results = check_cross_field_consistency(ext)
        assert not any(r.status == CheckStatus.FAIL for r in results)

    def test_aggressive_risk_normal_horizon_passes(self):
        ext = _make_extraction(
            doc_type=DocType.INVESTMENT_POLICY_STATEMENT,
            fields=[
                _make_field("risk_tolerance", "aggressive"),
                _make_field("time_horizon", 15),
            ],
        )
        results = check_cross_field_consistency(ext)
        assert not any(r.status == CheckStatus.FAIL for r in results)


class TestSummaryZeroGrounding:
    def test_long_numberless_summary_on_financial_doc_fails(self):
        """Schema-valid garbage: long summary with no numbers on a QPR."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("performance_ytd", 8.5, "YTD Return: 8.5%", (0, 16))],
            summary=(
                "The client's portfolio demonstrates excellent risk-adjusted returns with a "
                "Sharpe ratio exceeding industry benchmarks across all time periods analyzed. "
                "The diversification strategy has effectively minimized drawdown risk while "
                "capturing upside potential in both domestic and international markets."
            ),
        )
        claims = parse_claims(ext)
        source = "YTD Return: 8.5%"
        results = check_summary_extraction_consistency(source, ext, claims)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        assert len(fails) >= 1
        assert any(r.tag == AuditTag.SUMMARY_OVERREACH for r in fails)

    def test_short_numberless_summary_not_flagged(self):
        """Short summary with no numbers should not trip the heuristic."""
        ext = _make_extraction(
            doc_type=DocType.QUARTERLY_PORTFOLIO_REVIEW,
            fields=[_make_field("commentary", "Strong quarter")],
            summary="Strong quarterly performance with no major changes.",
        )
        claims = parse_claims(ext)
        results = check_summary_extraction_consistency("Strong quarterly performance.", ext, claims)
        overreach = [
            r for r in results
            if r.status == CheckStatus.FAIL and r.tag == AuditTag.SUMMARY_OVERREACH
            and "no specific numeric" in r.reason
        ]
        assert len(overreach) == 0

    def test_account_opening_long_numberless_summary_not_flagged(self):
        """Heuristic only applies to numeric-heavy doc types."""
        ext = _make_extraction(
            doc_type=DocType.ACCOUNT_OPENING,
            fields=[_make_field("client_name", "Jane Doe")],
            summary=(
                "The client has expressed satisfaction with portfolio performance and has "
                "confirmed that their investment objectives, risk tolerance, and financial "
                "circumstances remain unchanged since the last review period and is fully "
                "compliant with all regulatory requirements currently in effect."
            ),
        )
        claims = parse_claims(ext)
        results = check_summary_extraction_consistency("Client: Jane Doe", ext, claims)
        overreach = [
            r for r in results
            if r.status == CheckStatus.FAIL and r.tag == AuditTag.SUMMARY_OVERREACH
            and "no specific numeric" in r.reason
        ]
        assert len(overreach) == 0
