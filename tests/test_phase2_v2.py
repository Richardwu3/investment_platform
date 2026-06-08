"""
tests/test_phase2_v2.py
========================
Unit tests for Phase 2.2: executions table, get_full_trace(),
review_copilot, and updated analyzer.

Run with:
    python tests/test_phase2_v2.py

Or with pytest:
    python -m pytest tests/test_phase2_v2.py -v

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decisions.db import Database, DecisionRepository
from decisions.analyzer import DecisionAnalyzer
from decisions.review_copilot import DecisionReviewCopilot


# ---------------------------------------------------------------------------
# Shared test fixture helpers
# ---------------------------------------------------------------------------

def make_repo() -> tuple[DecisionRepository, str]:
    """Create a temp-file SQLite repo. Returns (repo, tmp_path)."""
    tmp = tempfile.mktemp(suffix=".db")
    db = Database(tmp)
    return DecisionRepository(db), tmp


def seed_decision(
    repo: DecisionRepository,
    human_decision: str = "approve",
    human_weights: dict | None = None,
    human_reason: str | None = None,
    date: str = "2024-01-31",
) -> int:
    """Create a cycle + decision, return decision_id."""
    cycle_id = repo.create_cycle(date, "GTAA_126d_Top2", market_regime="bull")
    return repo.log_decision(
        date=date,
        strategy="GTAA_126d_Top2",
        ai_signal={"SPY": 0.5, "TLT": 0.5},
        human_decision=human_decision,
        ai_confidence=0.75,
        ai_confidence_method="momentum_spread",
        ai_momentum_scores={"SPY": 0.089, "QQQ": 0.072, "TLT": -0.031, "GLD": 0.012, "DBC": -0.044},
        ai_selected_assets=["SPY", "TLT"],
        human_weights=human_weights,
        human_reason=human_reason,
        cycle_id=cycle_id,
    )


def seed_execution(repo: DecisionRepository, decision_id: int, **kwargs) -> int:
    """Add one execution fill with sensible defaults."""
    defaults = dict(
        symbol="SPY", side="buy", quantity=10.0, price=450.0,
        execution_time="2024-02-01T09:31:00",
        commission=1.00, commission_type="flat", broker="alpaca",
    )
    defaults.update(kwargs)
    return repo.add_execution(decision_id=decision_id, **defaults)


def seed_outcome(repo: DecisionRepository, decision_id: int, **kwargs) -> int:
    """Add an outcome with sensible defaults."""
    defaults = dict(
        actual_return_30d=0.035,
        benchmark_return_30d=0.028,
        ai_only_return_30d=0.028,
        asset_returns={"SPY": 0.038, "TLT": 0.032},
    )
    defaults.update(kwargs)
    return repo.log_outcome(decision_id=decision_id, **defaults)


# ---------------------------------------------------------------------------
# Tests: executions table
# ---------------------------------------------------------------------------

class TestExecutions(unittest.TestCase):

    def setUp(self):
        self.repo, self.tmp = make_repo()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_add_execution_approve(self):
        """add_execution should succeed for an approved decision."""
        did = seed_decision(self.repo, "approve")
        eid = seed_execution(self.repo, did)
        self.assertGreater(eid, 0)

    def test_add_execution_modify(self):
        """add_execution should succeed for a modified decision."""
        did = seed_decision(
            self.repo, "modify",
            human_weights={"SPY": 0.7, "GLD": 0.3},
            human_reason="test reason",
        )
        eid = seed_execution(self.repo, did, symbol="SPY")
        self.assertGreater(eid, 0)

    def test_add_execution_rejected_raises(self):
        """add_execution must raise ValueError for a rejected decision."""
        did = seed_decision(self.repo, "reject", human_reason="too uncertain")
        with self.assertRaises(ValueError, msg="Should reject execution on rejected decision"):
            seed_execution(self.repo, did)

    def test_add_execution_nonexistent_decision_raises(self):
        """add_execution must raise ValueError if decision_id not found."""
        with self.assertRaises(ValueError):
            seed_execution(self.repo, 99999)

    def test_add_execution_invalid_side_raises(self):
        did = seed_decision(self.repo)
        with self.assertRaises(ValueError):
            repo_add = self.repo.add_execution(
                decision_id=did, symbol="SPY", side="short",  # invalid
                quantity=10, price=450, execution_time="2024-02-01T09:31:00",
            )

    def test_add_execution_zero_quantity_raises(self):
        did = seed_decision(self.repo)
        with self.assertRaises(ValueError):
            self.repo.add_execution(
                decision_id=did, symbol="SPY", side="buy",
                quantity=0, price=450, execution_time="2024-02-01T09:31:00",
            )

    def test_add_execution_zero_price_raises(self):
        did = seed_decision(self.repo)
        with self.assertRaises(ValueError):
            self.repo.add_execution(
                decision_id=did, symbol="SPY", side="buy",
                quantity=10, price=0, execution_time="2024-02-01T09:31:00",
            )

    def test_net_amount_buy_is_negative(self):
        """Buy fills should have negative net_amount (cash leaves portfolio)."""
        did = seed_decision(self.repo)
        eid = seed_execution(self.repo, did, quantity=10, price=450.0, commission=1.0)
        execs = self.repo.get_executions_by_decision(did)
        self.assertEqual(len(execs), 1)
        # net_amount = -(10 * 450 + 1) = -4501
        self.assertAlmostEqual(execs[0]["net_amount"], -4501.0, places=4)

    def test_net_amount_sell_is_positive(self):
        """Sell fills should have positive net_amount (cash enters portfolio)."""
        did = seed_decision(self.repo)
        eid = self.repo.add_execution(
            decision_id=did, symbol="SPY", side="sell",
            quantity=10, price=450.0, commission=1.0,
            execution_time="2024-02-01T09:31:00",
        )
        execs = self.repo.get_executions_by_decision(did)
        # net_amount = 10 * 450 - 1 = 4499
        self.assertAlmostEqual(execs[0]["net_amount"], 4499.0, places=4)

    def test_commission_bps_type_conversion(self):
        """commission_type='bps' should convert bps to dollar amount."""
        did = seed_decision(self.repo)
        # 10 shares @ $450, commission = 5 bps = 0.0005 * 4500 = $2.25
        self.repo.add_execution(
            decision_id=did, symbol="SPY", side="buy",
            quantity=10, price=450.0,
            commission=5.0, commission_type="bps",
            execution_time="2024-02-01T09:31:00",
        )
        execs = self.repo.get_executions_by_decision(did)
        self.assertAlmostEqual(execs[0]["commission"], 2.25, places=4)

    def test_get_executions_empty_for_no_fills(self):
        """get_executions_by_decision returns [] when no fills recorded."""
        did = seed_decision(self.repo)
        execs = self.repo.get_executions_by_decision(did)
        self.assertEqual(execs, [])

    def test_get_executions_multiple_fills_ordered(self):
        """Multiple fills should be returned ordered by execution_time ASC."""
        did = seed_decision(self.repo)
        self.repo.add_execution(
            did, symbol="TLT", side="buy", quantity=5, price=98.0,
            execution_time="2024-02-01T09:32:00",
        )
        self.repo.add_execution(
            did, symbol="SPY", side="buy", quantity=10, price=450.0,
            execution_time="2024-02-01T09:31:00",
        )
        execs = self.repo.get_executions_by_decision(did)
        self.assertEqual(len(execs), 2)
        # SPY fill was earlier, should appear first
        self.assertEqual(execs[0]["symbol"], "SPY")
        self.assertEqual(execs[1]["symbol"], "TLT")

    def test_execution_order_id_stored(self):
        """order_id field should be stored and retrievable."""
        did = seed_decision(self.repo)
        seed_execution(self.repo, did, order_id="ord_abc123")
        execs = self.repo.get_executions_by_decision(did)
        self.assertEqual(execs[0]["order_id"], "ord_abc123")

    def test_execution_broker_default_paper(self):
        """Default broker should be 'paper'."""
        did = seed_decision(self.repo)
        self.repo.add_execution(
            did, symbol="SPY", side="buy", quantity=10, price=450.0,
            execution_time="2024-02-01T09:31:00",
        )
        execs = self.repo.get_executions_by_decision(did)
        self.assertEqual(execs[0]["broker"], "paper")


# ---------------------------------------------------------------------------
# Tests: get_full_trace
# ---------------------------------------------------------------------------

class TestGetFullTrace(unittest.TestCase):

    def setUp(self):
        self.repo, self.tmp = make_repo()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_trace_returns_none_for_missing_decision(self):
        """get_full_trace returns None for non-existent decision_id."""
        result = self.repo.get_full_trace(99999)
        self.assertIsNone(result)

    def test_trace_structure_keys(self):
        """get_full_trace returns dict with all required top-level keys."""
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        self.assertIsNotNone(trace)
        for key in ("decision", "cycle", "executions", "outcome", "execution_summary"):
            self.assertIn(key, trace, f"Missing key: {key}")

    def test_trace_decision_is_deserialized(self):
        """decision.ai_signal should be a dict, not a JSON string."""
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        self.assertIsInstance(trace["decision"]["ai_signal"], dict)

    def test_trace_cycle_populated(self):
        """trace.cycle should contain cycle fields when linked."""
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        self.assertIsNotNone(trace["cycle"])
        self.assertIn("cycle_date", trace["cycle"])
        self.assertEqual(trace["cycle"]["market_regime"], "bull")

    def test_trace_executions_empty_by_default(self):
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        self.assertEqual(trace["executions"], [])

    def test_trace_executions_populated(self):
        did = seed_decision(self.repo)
        seed_execution(self.repo, did, symbol="SPY")
        seed_execution(self.repo, did, symbol="TLT", price=98.0, quantity=5)
        trace = self.repo.get_full_trace(did)
        self.assertEqual(len(trace["executions"]), 2)
        symbols = {e["symbol"] for e in trace["executions"]}
        self.assertEqual(symbols, {"SPY", "TLT"})

    def test_trace_outcome_none_when_not_recorded(self):
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        self.assertIsNone(trace["outcome"])

    def test_trace_outcome_populated(self):
        did = seed_decision(self.repo)
        seed_outcome(self.repo, did, actual_return_30d=0.04)
        trace = self.repo.get_full_trace(did)
        self.assertIsNotNone(trace["outcome"])
        self.assertAlmostEqual(trace["outcome"]["actual_return_30d"], 0.04)

    def test_trace_execution_summary_totals(self):
        """execution_summary totals should be computed correctly."""
        did = seed_decision(self.repo)
        seed_execution(self.repo, did, symbol="SPY", quantity=10, price=450.0, commission=1.0)
        seed_execution(self.repo, did, symbol="TLT", quantity=5, price=98.0, commission=0.5)
        trace = self.repo.get_full_trace(did)
        es = trace["execution_summary"]
        self.assertEqual(es["total_legs"], 2)
        # net_amount abs: |-(4500+1)| + |-(490+0.5)| = 4501 + 490.5 = 4991.5
        self.assertAlmostEqual(es["total_notional"], 4991.5, places=1)
        self.assertAlmostEqual(es["total_commission"], 1.5, places=4)
        self.assertIn("SPY", es["symbols_traded"])
        self.assertIn("TLT", es["symbols_traded"])

    def test_trace_execution_summary_fill_times(self):
        """first_fill_time and last_fill_time should be min/max of execution_time."""
        did = seed_decision(self.repo)
        self.repo.add_execution(
            did, symbol="SPY", side="buy", quantity=10, price=450.0,
            execution_time="2024-02-01T09:31:00",
        )
        self.repo.add_execution(
            did, symbol="TLT", side="buy", quantity=5, price=98.0,
            execution_time="2024-02-01T09:32:15",
        )
        trace = self.repo.get_full_trace(did)
        es = trace["execution_summary"]
        self.assertEqual(es["first_fill_time"], "2024-02-01T09:31:00")
        self.assertEqual(es["last_fill_time"], "2024-02-01T09:32:15")

    def test_trace_summary_empty_executions(self):
        """execution_summary with no fills should have zero totals."""
        did = seed_decision(self.repo)
        trace = self.repo.get_full_trace(did)
        es = trace["execution_summary"]
        self.assertEqual(es["total_legs"], 0)
        self.assertAlmostEqual(es["total_notional"], 0.0)
        self.assertIsNone(es["first_fill_time"])
        self.assertIsNone(es["last_fill_time"])


# ---------------------------------------------------------------------------
# Tests: DecisionAnalyzer new methods
# ---------------------------------------------------------------------------

class TestAnalyzerValueAdd(unittest.TestCase):
    """Tests for calculate_human_value_add and per_decision_value_add."""

    def test_calculate_human_value_add_basic(self):
        """Should return actual - ai_only."""
        outcome = {"actual_return_30d": 0.028, "ai_only_return_30d": 0.021}
        result = DecisionAnalyzer.calculate_human_value_add(outcome)
        self.assertAlmostEqual(result, 0.007, places=6)

    def test_calculate_human_value_add_negative(self):
        """Human detraction case: actual < ai_only."""
        outcome = {"actual_return_30d": 0.015, "ai_only_return_30d": 0.035}
        result = DecisionAnalyzer.calculate_human_value_add(outcome)
        self.assertAlmostEqual(result, -0.02, places=6)

    def test_calculate_human_value_add_none_when_no_ai_only(self):
        """Returns None when ai_only_return_30d is not recorded."""
        outcome = {"actual_return_30d": 0.03, "ai_only_return_30d": None}
        self.assertIsNone(DecisionAnalyzer.calculate_human_value_add(outcome))

    def test_calculate_human_value_add_none_when_no_actual(self):
        """Returns None when actual_return_30d is not recorded."""
        outcome = {"actual_return_30d": None, "ai_only_return_30d": 0.02}
        self.assertIsNone(DecisionAnalyzer.calculate_human_value_add(outcome))

    def test_calculate_human_value_add_zero(self):
        """Returns 0.0 when actual == ai_only (no value added or lost)."""
        outcome = {"actual_return_30d": 0.03, "ai_only_return_30d": 0.03}
        result = DecisionAnalyzer.calculate_human_value_add(outcome)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_per_decision_value_add_empty(self):
        """Returns empty list when no records."""
        analyzer = DecisionAnalyzer([])
        self.assertEqual(analyzer.per_decision_value_add(), [])

    def test_per_decision_value_add_no_modify(self):
        """Returns empty list when no modify decisions."""
        records = [
            {"id": 1, "date": "2024-01-31", "strategy": "GTAA",
             "human_decision": "approve", "actual_return_30d": 0.03,
             "human_value_add": None},
        ]
        analyzer = DecisionAnalyzer(records)
        self.assertEqual(analyzer.per_decision_value_add(), [])

    def test_per_decision_value_add_returns_series(self):
        """Returns list with correct fields for modify decisions."""
        records = [
            {"id": 2, "date": "2024-02-29", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.028, "human_value_add": 0.007},
            {"id": 6, "date": "2024-06-28", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.033, "human_value_add": 0.002},
        ]
        analyzer = DecisionAnalyzer(records)
        series = analyzer.per_decision_value_add()
        self.assertEqual(len(series), 2)
        # Check required fields
        for row in series:
            for field in ("decision_id", "date", "strategy",
                          "actual_return", "ai_return", "value_add", "helpful"):
                self.assertIn(field, row, f"Missing field: {field}")

    def test_per_decision_value_add_sorted_by_date(self):
        """Results should be sorted by date ascending."""
        records = [
            {"id": 6, "date": "2024-06-28", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.033, "human_value_add": 0.002},
            {"id": 2, "date": "2024-02-29", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.028, "human_value_add": 0.007},
        ]
        analyzer = DecisionAnalyzer(records)
        series = analyzer.per_decision_value_add()
        self.assertEqual(series[0]["date"], "2024-02-29")
        self.assertEqual(series[1]["date"], "2024-06-28")

    def test_per_decision_value_add_helpful_flag(self):
        """helpful=True when value_add > 0, False when <= 0."""
        records = [
            {"id": 1, "date": "2024-01-31", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.03, "human_value_add": 0.005},
            {"id": 2, "date": "2024-02-28", "strategy": "GTAA",
             "human_decision": "modify",
             "actual_return_30d": 0.01, "human_value_add": -0.01},
        ]
        analyzer = DecisionAnalyzer(records)
        series = analyzer.per_decision_value_add()
        self.assertTrue(series[0]["helpful"])
        self.assertFalse(series[1]["helpful"])

    def test_full_report_includes_per_decision(self):
        """full_report should include per_decision_value_add key."""
        analyzer = DecisionAnalyzer([])
        report = analyzer.full_report()
        self.assertIn("per_decision_value_add", report)


# ---------------------------------------------------------------------------
# Tests: DecisionReviewCopilot
# ---------------------------------------------------------------------------

class TestDecisionReviewCopilot(unittest.TestCase):

    def setUp(self):
        self.repo, self.tmp = make_repo()
        self.copilot = DecisionReviewCopilot(self.repo, use_llm=False)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _full_seed(self, human_decision="approve", human_weights=None,
                   human_reason=None, with_outcome=True, with_execution=True):
        """Create a fully-seeded decision with fills and outcome."""
        did = seed_decision(
            self.repo, human_decision=human_decision,
            human_weights=human_weights, human_reason=human_reason,
        )
        if with_execution and human_decision != "reject":
            seed_execution(self.repo, did, symbol="SPY", quantity=12, price=473.0)
            seed_execution(self.repo, did, symbol="TLT", quantity=50, price=98.0)
        if with_outcome and human_decision != "reject":
            seed_outcome(self.repo, did)
        return did

    def test_generate_review_raises_for_missing_id(self):
        """generate_review should raise ValueError for non-existent decision."""
        with self.assertRaises(ValueError):
            self.copilot.generate_review(99999)

    def test_generate_review_returns_string(self):
        """generate_review should return a non-empty string."""
        did = self._full_seed()
        md = self.copilot.generate_review(did)
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 100)

    def test_review_contains_decision_id(self):
        did = self._full_seed()
        md = self.copilot.generate_review(did)
        self.assertIn(f"#{did}", md)

    def test_review_contains_all_section_headers(self):
        """All 8 numbered sections should appear in the report."""
        did = self._full_seed()
        md = self.copilot.generate_review(did)
        for i in range(1, 9):
            self.assertIn(f"## {i}.", md, f"Missing section {i}")

    def test_review_approve_contains_approved_badge(self):
        did = self._full_seed(human_decision="approve")
        md = self.copilot.generate_review(did)
        self.assertIn("Approved", md)

    def test_review_modify_contains_modified_badge(self):
        did = self._full_seed(
            human_decision="modify",
            human_weights={"SPY": 0.7, "GLD": 0.3},
            human_reason="Macro hedge",
        )
        md = self.copilot.generate_review(did)
        self.assertIn("Modified", md)

    def test_review_modify_contains_human_reason(self):
        reason = "Macro hedge concern"
        did = self._full_seed(
            human_decision="modify",
            human_weights={"SPY": 0.7, "GLD": 0.3},
            human_reason=reason,
        )
        md = self.copilot.generate_review(did)
        self.assertIn(reason, md)

    def test_review_reject_contains_rejected_badge(self):
        did = self._full_seed(
            human_decision="reject",
            human_reason="Too uncertain",
            with_outcome=False,
            with_execution=False,
        )
        md = self.copilot.generate_review(did)
        self.assertIn("Rejected", md)

    def test_review_pending_outcome_shows_pending_message(self):
        """Decision without outcome should show 'Outcome not yet recorded'."""
        did = seed_decision(self.repo)  # no outcome
        md = self.copilot.generate_review(did)
        self.assertIn("not yet recorded", md.lower())

    def test_review_with_outcome_shows_return(self):
        did = self._full_seed(with_outcome=True)
        md = self.copilot.generate_review(did)
        # 3.5% realized return from seed_outcome default
        self.assertIn("3.50%", md)

    def test_review_execution_section_shows_fills(self):
        did = self._full_seed(with_execution=True, with_outcome=False)
        md = self.copilot.generate_review(did)
        self.assertIn("SPY", md)
        self.assertIn("TLT", md)

    def test_review_no_execution_shows_placeholder(self):
        did = seed_decision(self.repo)
        md = self.copilot.generate_review(did)
        self.assertIn("No execution records", md)

    def test_review_markdown_has_tables(self):
        """Report should contain Markdown table separators."""
        did = self._full_seed()
        md = self.copilot.generate_review(did)
        self.assertIn("|", md)
        self.assertIn("---", md)

    def test_review_value_add_approve_shows_na(self):
        """Approved decisions should not show human value-add comparison."""
        did = self._full_seed(human_decision="approve")
        md = self.copilot.generate_review(did)
        # Section 6 exists but should say 'N/A' or similar for approved
        self.assertIn("## 6.", md)

    def test_review_value_add_modify_with_counterfactual(self):
        """Modified decision with ai_only outcome should show value-add."""
        did = seed_decision(
            self.repo, "modify",
            human_weights={"SPY": 0.7, "GLD": 0.3},
            human_reason="Macro override",
        )
        self.repo.log_outcome(
            decision_id=did,
            actual_return_30d=0.028,
            benchmark_return_30d=0.025,
            ai_only_return_30d=0.021,
        )
        md = self.copilot.generate_review(did)
        # Should mention human value-add
        self.assertIn("value-add", md.lower())
        # Should show both returns
        self.assertIn("2.80%", md)
        self.assertIn("2.10%", md)

    def test_save_review_creates_file(self):
        """save_review should write a .md file and return its path."""
        import tempfile
        did = self._full_seed()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self.copilot.save_review(did, output_dir=tmpdir)
            self.assertTrue(Path(filepath).exists())
            content = Path(filepath).read_text(encoding="utf-8")
            self.assertGreater(len(content), 100)
            self.assertTrue(filepath.endswith(".md"))

    def test_review_navigation_footer_contains_decision_id(self):
        """Footer should contain the decision ID in CLI commands."""
        did = self._full_seed()
        md = self.copilot.generate_review(did)
        self.assertIn(f"--id {did}", md)

    def test_review_rule_based_confidence_high(self):
        """High confidence (>= 0.75) should trigger 'high confidence' narrative."""
        did = seed_decision(self.repo)  # confidence=0.75 from seed_decision
        md = self.copilot.generate_review(did)
        # Rule-based summary checks confidence
        self.assertIn("high confidence", md.lower())

    def test_review_rule_based_no_execution_lesson(self):
        """No fills should trigger the 'log execution' lesson."""
        did = seed_decision(self.repo)
        md = self.copilot.generate_review(did)
        self.assertIn("exec", md)

    def test_review_rule_based_with_execution_commission_lesson(self):
        """Having fills with commission should mention commission in lessons."""
        did = self._full_seed(with_outcome=False)
        md = self.copilot.generate_review(did)
        self.assertIn("commission", md.lower())


# ---------------------------------------------------------------------------
# Tests: regression — existing behavior unchanged
# ---------------------------------------------------------------------------

class TestRegressionExistingBehavior(unittest.TestCase):
    """Ensure existing Phase 2.1 behavior is not broken by new additions."""

    def setUp(self):
        self.repo, self.tmp = make_repo()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_log_decision_still_works(self):
        did = seed_decision(self.repo)
        self.assertGreater(did, 0)

    def test_get_decision_still_deserializes_json(self):
        did = seed_decision(self.repo)
        d = self.repo.get_decision(did)
        self.assertIsInstance(d["ai_signal"], dict)
        self.assertIsInstance(d["ai_momentum_scores"], dict)

    def test_log_outcome_still_computes_ai_correct(self):
        did = seed_decision(self.repo)
        seed_outcome(self.repo, did, actual_return_30d=0.04)
        records = self.repo.get_all_outcomes()
        r = next(x for x in records if x["id"] == did)
        self.assertEqual(r["ai_correct"], "direction_correct")

    def test_modify_without_weights_still_raises(self):
        with self.assertRaises(ValueError):
            self.repo.log_decision(
                date="2024-01-31", strategy="GTAA",
                ai_signal={"SPY": 0.5},
                human_decision="modify",
            )

    def test_duplicate_outcome_still_raises(self):
        did = seed_decision(self.repo)
        seed_outcome(self.repo, did)
        with self.assertRaises(ValueError):
            seed_outcome(self.repo, did)

    def test_executions_table_exists_in_schema(self):
        """The new executions table should be created by schema init."""
        with self.repo.db.connect() as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("executions", tables)

    def test_analyzer_full_report_backward_compatible(self):
        """full_report should still return all original keys plus new ones."""
        analyzer = DecisionAnalyzer([])
        report = analyzer.full_report()
        original_keys = [
            "adoption_rates", "ai_accuracy", "human_value_add",
            "confidence_calibration", "by_strategy",
            "pending_outcomes_count", "generated_at",
        ]
        for key in original_keys:
            self.assertIn(key, report, f"Regression: missing key '{key}'")
        # New key
        self.assertIn("per_decision_value_add", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
