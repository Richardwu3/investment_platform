"""
decisions/review_copilot.py
============================
AI-powered post-decision review report generator.

Generates structured Markdown retrospectives for each investment decision,
combining deterministic metrics (from Code = Truth layer) with optional
LLM-generated narrative insights (from AI Interface layer).

Architecture:
    Code = Truth  → All numbers computed by DecisionAnalyzer / DecisionRepository
    LLM = Interface → Reads the numbers, writes the narrative in plain English
    Human reviews the final Markdown before any action is taken

Two operating modes (selected by use_llm flag):
    Rule-based (default, always works, no API key needed):
        Generates structured Markdown with templated observations.
        Deterministic, reproducible, fast.

    LLM-enhanced (requires ANTHROPIC_API_KEY env variable):
        Calls Claude claude-sonnet-4-20250514 to generate genuinely insightful narrative
        for "Review Summary" and "Lessons Learned" sections.
        All factual inputs (metrics, weights, returns) are passed as structured
        context — LLM only writes prose, never computes numbers.
        Falls back to rule-based if API call fails.

Report Sections:
    1. Decision Snapshot    — date, strategy, market regime
    2. AI Recommendation    — signal weights, confidence, momentum scores
    3. Human Review         — decision type, modifications, reasoning
    4. Execution Record     — fills, quantities, prices, commissions
    5. 30-Day Outcome       — realized return vs. benchmark, AI accuracy verdict
    6. Human vs. AI         — value-add analysis (actual vs. counterfactual)
    7. Review Summary       — rule-based or LLM-generated synthesis
    8. Lessons Learned      — rule-based or LLM-generated improvement ideas

Usage::

    from decisions.db import Database, DecisionRepository
    from decisions.review_copilot import DecisionReviewCopilot

    db = Database("decisions/journal.db")
    repo = DecisionRepository(db)
    copilot = DecisionReviewCopilot(repo)

    # Rule-based (no API key needed)
    md = copilot.generate_review(decision_id=3)

    # LLM-enhanced
    copilot_llm = DecisionReviewCopilot(repo, use_llm=True)
    md = copilot_llm.generate_review(decision_id=3)

    print(md)
    # or save to file:
    Path("reports/review_3.md").write_text(md)

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime
from typing import Dict, List, Optional

from decisions.analyzer import DecisionAnalyzer
from decisions.db import DecisionRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers (private, no external dependencies)
# ---------------------------------------------------------------------------

def _pct(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format float as percentage string."""
    if value is None:
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def _ratio(value: Optional[float], decimals: int = 3) -> str:
    """Format float as ratio string."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _weights_table(weights: Dict[str, float]) -> str:
    """Render a weight dict as a Markdown table row."""
    if not weights:
        return "_No weights recorded_"
    rows = ["| Asset | Weight |", "|-------|--------|"]
    for symbol, w in sorted(weights.items()):
        rows.append(f"| {symbol} | {_pct(w)} |")
    return "\n".join(rows)


def _momentum_table(scores: Dict[str, float]) -> str:
    """Render momentum scores as a Markdown table, sorted descending."""
    if not scores:
        return "_No momentum scores recorded_"
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = ["| Asset | Momentum Score | Rank |", "|-------|---------------|------|"]
    for rank, (symbol, score) in enumerate(sorted_scores, 1):
        bar = "▓" * min(int(abs(score) * 100), 15)
        direction = "+" if score >= 0 else ""
        rows.append(f"| {symbol} | {direction}{score:.4f}  {bar} | #{rank} |")
    return "\n".join(rows)


def _executions_table(executions: List[Dict]) -> str:
    """Render execution records as a Markdown table."""
    if not executions:
        return "_No execution records logged._"
    rows = [
        "| # | Symbol | Side | Qty | Price | Net Amount | Commission | Broker | Time |",
        "|---|--------|------|-----|-------|-----------|------------|--------|------|",
    ]
    for i, ex in enumerate(executions, 1):
        net = ex.get("net_amount")
        net_str = f"${net:+.2f}" if net is not None else "N/A"
        rows.append(
            f"| {i} | **{ex['symbol']}** | {ex['side'].upper()} | "
            f"{ex['quantity']:.4f} | ${ex['price']:.2f} | {net_str} | "
            f"${ex.get('commission', 0):.2f} | {ex.get('broker', 'N/A')} | "
            f"{str(ex.get('execution_time', 'N/A'))[:16]} |"
        )
    return "\n".join(rows)


def _outcome_badge(ai_correct: Optional[str]) -> str:
    """Return an emoji badge for AI accuracy verdict."""
    return {
        "direction_correct": "✅ Direction Correct",
        "direction_wrong":   "❌ Direction Wrong",
        "inconclusive":      "⚪ Inconclusive",
    }.get(ai_correct or "", "⏳ Pending")


def _human_decision_badge(decision: str) -> str:
    """Return an emoji badge for human decision type."""
    return {
        "approve": "✅ Approved",
        "modify":  "✏️ Modified",
        "reject":  "🚫 Rejected",
    }.get(decision, decision)


def _regime_badge(regime: Optional[str]) -> str:
    """Return an emoji badge for market regime."""
    return {
        "bull":     "🐂 Bull Market",
        "bear":     "🐻 Bear Market",
        "sideways": "🦀 Sideways Market",
    }.get(regime or "", "❓ Unknown")


# ---------------------------------------------------------------------------
# Rule-based narrative generators
# ---------------------------------------------------------------------------

def _rule_based_summary(trace: Dict, value_add: Optional[float]) -> str:
    """
    Generate a templated review summary without LLM.

    Logic is transparent and deterministic — good for tests and offline use.

    Args:
        trace:     Full trace dict from DecisionRepository.get_full_trace().
        value_add: Human value-add float, or None.

    Returns:
        Markdown string for the Review Summary section.
    """
    decision = trace["decision"]
    outcome = trace.get("outcome")
    human_dec = decision.get("human_decision", "")
    conf = decision.get("ai_confidence")
    ai_correct = outcome.get("ai_correct") if outcome else None
    actual_ret = outcome.get("actual_return_30d") if outcome else None
    benchmark_ret = outcome.get("benchmark_return_30d") if outcome else None

    lines = []

    # Confidence commentary
    if conf is not None:
        if conf >= 0.75:
            lines.append(
                f"The AI model expressed **high confidence** ({_pct(conf)}) in this signal, "
                "indicating a strong momentum separation between selected and excluded assets."
            )
        elif conf >= 0.55:
            lines.append(
                f"The AI model expressed **moderate confidence** ({_pct(conf)}), "
                "suggesting the momentum signal was present but not definitive."
            )
        else:
            lines.append(
                f"The AI model expressed **low confidence** ({_pct(conf)}), "
                "which should have flagged this as a high-uncertainty setup."
            )

    # Human decision commentary
    if human_dec == "approve":
        lines.append(
            "The human reviewer **approved the signal without modification**, "
            "indicating alignment with the AI's assessment of market conditions."
        )
    elif human_dec == "modify":
        reason = decision.get("human_reason", "no reason recorded")
        lines.append(
            f"The human reviewer **modified the allocation** with the rationale: "
            f'_"{reason}"_'
        )
        if value_add is not None:
            if value_add > 0:
                lines.append(
                    f"This modification generated **+{_pct(value_add)} of human alpha** "
                    "over the pure AI signal — the override improved outcomes."
                )
            elif value_add < 0:
                lines.append(
                    f"This modification resulted in **{_pct(value_add)} vs. AI signal** — "
                    "the pure AI allocation would have performed better."
                )
    elif human_dec == "reject":
        lines.append(
            "The human reviewer **rejected the signal entirely**, "
            f"citing: _{decision.get('human_reason', 'no reason recorded')}_"
        )

    # Outcome commentary
    if outcome is None:
        lines.append("_No outcome data recorded yet — check back after 30 days._")
    else:
        if ai_correct == "direction_correct":
            lines.append(
                f"**The AI signal was directionally correct**: the portfolio returned "
                f"{_pct(actual_ret, signed=True)} over 30 days."
            )
        elif ai_correct == "direction_wrong":
            lines.append(
                f"**The AI signal was directionally wrong**: the portfolio returned "
                f"{_pct(actual_ret, signed=True)} over 30 days."
            )
        elif ai_correct == "inconclusive":
            lines.append(
                f"**The outcome is inconclusive** ({_pct(actual_ret, signed=True)}): "
                "the absolute return is below the 0.5% noise threshold."
            )

        if benchmark_ret is not None and actual_ret is not None:
            excess = actual_ret - benchmark_ret
            if excess > 0:
                lines.append(
                    f"The portfolio **outperformed SPY** by {_pct(excess, signed=True)} "
                    "over the same window."
                )
            else:
                lines.append(
                    f"The portfolio **underperformed SPY** by {_pct(excess, signed=True)} "
                    "over the same window."
                )

    return "\n\n".join(lines)


def _rule_based_lessons(trace: Dict, value_add: Optional[float]) -> str:
    """
    Generate templated lessons learned section without LLM.

    Args:
        trace:     Full trace dict.
        value_add: Human value-add float, or None.

    Returns:
        Markdown bullet list as string.
    """
    decision = trace["decision"]
    outcome = trace.get("outcome")
    conf = decision.get("ai_confidence")
    human_dec = decision.get("human_decision", "")
    ai_correct = outcome.get("ai_correct") if outcome else None

    lessons = []

    # Confidence-based lessons
    if conf is not None and conf < 0.55 and ai_correct == "direction_wrong":
        lessons.append(
            "**Low-confidence signals warrant smaller position sizes or outright rejection.** "
            f"This setup had {_pct(conf)} confidence and an incorrect direction call — "
            "consider setting a confidence floor (e.g. 60%) below which the strategy defaults to cash."
        )
    elif conf is not None and conf >= 0.75 and ai_correct == "direction_correct":
        lessons.append(
            "**High-confidence signals have been historically reliable.** "
            f"This {_pct(conf)} confidence setup delivered a correct directional call — "
            "reinforce the conviction to approve high-confidence signals promptly."
        )

    # Human override lessons
    if human_dec == "modify" and value_add is not None:
        if value_add > 0.005:
            lessons.append(
                f"**This human modification added measurable value ({_pct(value_add, signed=True)}).** "
                "Analyze what information informed this override — if it's a repeatable signal "
                "(macro view, earnings risk, correlation concern), consider formalizing it as a "
                "strategy rule or a secondary signal layer."
            )
        elif value_add < -0.005:
            lessons.append(
                f"**This human modification detracted {_pct(value_add, signed=True)} vs. pure AI.** "
                "Review whether the override rationale was based on genuine information edge "
                "or behavioral bias (recency, overconfidence, loss aversion). "
                "Track your override success rate in the analyzer."
            )

    # Execution lessons
    executions = trace.get("executions", [])
    if not executions:
        lessons.append(
            "**No execution records were logged for this decision.** "
            "Consider using `python decisions/cli.py exec` to track fills — "
            "execution data enables slippage analysis and P&L attribution."
        )
    else:
        total_commission = sum(e.get("commission") or 0.0 for e in executions)
        if total_commission > 0:
            lessons.append(
                f"**Total commission paid: ${total_commission:.2f}.** "
                "Track commission costs over time relative to strategy returns — "
                "at scale, execution costs meaningfully impact realized alpha."
            )

    # Generic fallback
    if not lessons:
        lessons.append(
            "Continue tracking decisions to build a statistically meaningful dataset. "
            "The feedback loop becomes most valuable after 20+ decisions — "
            "patterns in confidence calibration and override success rates will emerge."
        )

    return "\n\n".join(f"- {lesson}" for lesson in lessons)


# ---------------------------------------------------------------------------
# LLM-enhanced narrative (calls Claude API)
# ---------------------------------------------------------------------------

def _llm_generate_section(prompt: str, section_name: str) -> Optional[str]:
    """
    Call the Anthropic API to generate a narrative section.

    Reads ANTHROPIC_API_KEY from environment. Falls back gracefully to None
    if the API key is absent, the package is unavailable, or the call fails.

    Args:
        prompt:       Full prompt including structured context and instruction.
        section_name: Label for logging (e.g. "review_summary").

    Returns:
        Generated markdown string, or None on any failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set; skipping LLM for section '%s'.", section_name)
        return None

    try:
        import anthropic  # type: ignore
    except ImportError:
        logger.warning("anthropic package not installed; skipping LLM. pip install anthropic")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=(
                "You are an investment research analyst generating a post-decision review "
                "for a quantitative asset management platform. "
                "Write in a concise, professional tone. "
                "Use markdown formatting (bold, bullet points). "
                "Do NOT invent numbers — only reference the metrics provided in the prompt. "
                "Focus on actionable insights, not generic commentary."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        logger.info("LLM generated section '%s' (%d chars).", section_name, len(text))
        return text
    except Exception as e:
        logger.warning("LLM call failed for section '%s': %s. Falling back to rule-based.", section_name, e)
        return None


# ---------------------------------------------------------------------------
# Main copilot class
# ---------------------------------------------------------------------------

class DecisionReviewCopilot:
    """
    Generates structured Markdown post-decision review reports.

    Combines deterministic metrics (Code = Truth) with optional LLM narrative
    (LLM = Interface). The LLM reads the metrics but never computes them.

    Operating modes:
        use_llm=False (default): Fully deterministic, no API key required.
        use_llm=True:            LLM writes the Summary and Lessons sections.
                                 Falls back to rule-based if API call fails.

    Usage::

        copilot = DecisionReviewCopilot(repo)
        md = copilot.generate_review(decision_id=3)
        Path("reports/review_3.md").write_text(md)
    """

    def __init__(
        self,
        repo: DecisionRepository,
        use_llm: bool = False,
    ) -> None:
        """
        Args:
            repo:    DecisionRepository instance (wired to database).
            use_llm: If True, attempt to use Anthropic API for narrative sections.
                     Requires ANTHROPIC_API_KEY environment variable.
        """
        self.repo = repo
        self.use_llm = use_llm
        logger.info(
            "DecisionReviewCopilot initialized | use_llm=%s", use_llm
        )

    def generate_review(self, decision_id: int) -> str:
        """
        Generate a complete post-decision review report in Markdown format.

        Assembles all eight sections from database data. Sections 7 and 8
        (Review Summary, Lessons Learned) are either rule-based or LLM-generated
        depending on the use_llm flag.

        Args:
            decision_id: Primary key of the decision to review.

        Returns:
            Complete Markdown string ready to write to file or display in terminal.

        Raises:
            ValueError: If decision_id is not found in the database.

        Example::

            md = copilot.generate_review(3)
            print(md)
            Path("reports/decision_3_review.md").write_text(md)
        """
        trace = self.repo.get_full_trace(decision_id)
        if trace is None:
            raise ValueError(f"Decision id={decision_id} not found.")

        decision = trace["decision"]
        cycle = trace.get("cycle")
        executions = trace.get("executions", [])
        outcome = trace.get("outcome")
        exec_summary = trace.get("execution_summary", {})

        # Compute human value-add for this single decision
        value_add = DecisionAnalyzer.calculate_human_value_add(outcome) if outcome else None

        # Effective weights (what was actually executed)
        if decision["human_decision"] == "modify" and decision.get("human_weights"):
            effective_weights = decision["human_weights"]
        elif decision["human_decision"] == "approve":
            effective_weights = decision.get("ai_signal", {})
        else:
            effective_weights = {}  # rejected

        sections = [
            self._section_header(decision_id, decision, cycle),
            self._section_ai_recommendation(decision),
            self._section_human_review(decision, effective_weights),
            self._section_executions(executions, exec_summary),
            self._section_outcome(outcome),
            self._section_human_vs_ai(decision, outcome, value_add, effective_weights),
            self._section_review_summary(trace, value_add),
            self._section_lessons_learned(trace, value_add),
            self._section_footer(decision_id),
        ]
        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_header(
        self,
        decision_id: int,
        decision: Dict,
        cycle: Optional[Dict],
    ) -> str:
        """Section 1: Decision Snapshot."""
        strategy = decision.get("strategy", "Unknown")
        date = decision.get("date", "Unknown")
        regime = cycle.get("market_regime") if cycle else None
        created = decision.get("created_at", "")[:19]
        human_dec = decision.get("human_decision", "")

        return textwrap.dedent(f"""\
            # 📋 Decision Review — #{decision_id}

            > *Generated by AI Investment Research Copilot | Yuchuan Wu Phase 2*
            > *Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

            ## 1. Decision Snapshot

            | Field | Value |
            |-------|-------|
            | **Decision ID** | #{decision_id} |
            | **Signal Date** | {date} |
            | **Strategy** | `{strategy}` |
            | **Market Regime** | {_regime_badge(regime)} |
            | **Human Decision** | {_human_decision_badge(human_dec)} |
            | **Decision Logged** | {created} |
        """).rstrip()

    def _section_ai_recommendation(self, decision: Dict) -> str:
        """Section 2: AI Recommendation."""
        ai_signal = decision.get("ai_signal") or {}
        confidence = decision.get("ai_confidence")
        conf_method = decision.get("ai_confidence_method", "N/A")
        selected = decision.get("ai_selected_assets") or list(ai_signal.keys())
        momentum_scores = decision.get("ai_momentum_scores") or {}

        conf_bar = ""
        if confidence is not None:
            filled = int(confidence * 20)
            conf_bar = f" `{'█' * filled}{'░' * (20 - filled)}` {_pct(confidence)}"

        return textwrap.dedent(f"""\
            ## 2. AI Recommendation

            **Selected Assets:** {', '.join(f'`{a}`' for a in selected) or 'None'}

            **Confidence Score:** {conf_bar or _ratio(confidence)}
            **Confidence Method:** `{conf_method}`

            ### Recommended Weights

            {_weights_table(ai_signal)}

            ### Momentum Scores (all universe assets)

            {_momentum_table(momentum_scores)}
        """).rstrip()

    def _section_human_review(self, decision: Dict, effective_weights: Dict) -> str:
        """Section 3: Human Review."""
        human_dec = decision.get("human_decision", "")
        human_weights = decision.get("human_weights") or {}
        human_reason = decision.get("human_reason") or "_No reason recorded._"

        modification_block = ""
        if human_dec == "modify" and human_weights:
            modification_block = f"""
### Modified Weights

{_weights_table(human_weights)}

### Effective Allocation (what was executed)

{_weights_table(effective_weights)}
"""
        elif human_dec == "reject":
            modification_block = "\n> ⚠️ **Decision was REJECTED** — no allocation executed.\n"

        return textwrap.dedent(f"""\
            ## 3. Human Review

            **Decision:** {_human_decision_badge(human_dec)}

            **Reviewer's Rationale:**
            > {human_reason}
            {modification_block}
        """).rstrip()

    def _section_executions(
        self,
        executions: List[Dict],
        exec_summary: Dict,
    ) -> str:
        """Section 4: Execution Record."""
        summary_block = ""
        if executions:
            summary_block = textwrap.dedent(f"""
                **Execution Summary:**
                - Legs filled: {exec_summary.get('total_legs', 0)}
                - Total notional: ${exec_summary.get('total_notional', 0):,.2f}
                - Total commission: ${exec_summary.get('total_commission', 0):.4f}
                - Brokers used: {', '.join(exec_summary.get('brokers_used', []))}
                - First fill: {exec_summary.get('first_fill_time', 'N/A')}
                - Last fill: {exec_summary.get('last_fill_time', 'N/A')}
            """)

        return textwrap.dedent(f"""\
            ## 4. Execution Record
            {summary_block}
            ### Fill Detail

            {_executions_table(executions)}
        """).rstrip()

    def _section_outcome(self, outcome: Optional[Dict]) -> str:
        """Section 5: 30-Day Outcome."""
        if outcome is None:
            return textwrap.dedent("""\
                ## 5. 30-Day Outcome

                > ⏳ **Outcome not yet recorded.**
                > Run `python decisions/cli.py outcome --id <ID> --return <value>` after 30 days.
            """).rstrip()

        actual = outcome.get("actual_return_30d")
        benchmark = outcome.get("benchmark_return_30d")
        ai_correct = outcome.get("ai_correct")
        asset_returns = outcome.get("asset_returns") or {}
        outcome_date = outcome.get("outcome_date", "Unknown")

        excess = None
        if actual is not None and benchmark is not None:
            excess = actual - benchmark

        asset_block = ""
        if asset_returns:
            asset_rows = ["| Asset | 30d Return |", "|-------|-----------|"]
            for sym, ret in sorted(asset_returns.items()):
                direction = "📈" if ret > 0 else "📉"
                asset_rows.append(f"| {sym} | {direction} {_pct(ret, signed=True)} |")
            asset_block = "\n### Per-Asset Returns\n\n" + "\n".join(asset_rows)

        return textwrap.dedent(f"""\
            ## 5. 30-Day Outcome

            | Metric | Value |
            |--------|-------|
            | **Realized Return (30d)** | {_pct(actual, signed=True)} |
            | **SPY Benchmark (30d)** | {_pct(benchmark, signed=True)} |
            | **Excess vs. Benchmark** | {_pct(excess, signed=True)} |
            | **AI Direction Verdict** | {_outcome_badge(ai_correct)} |
            | **Outcome Recorded** | {outcome_date} |
            {asset_block}
        """).rstrip()

    def _section_human_vs_ai(
        self,
        decision: Dict,
        outcome: Optional[Dict],
        value_add: Optional[float],
        effective_weights: Dict,
    ) -> str:
        """Section 6: Human vs. AI Comparison."""
        human_dec = decision.get("human_decision", "")

        if human_dec == "reject":
            return textwrap.dedent("""\
                ## 6. Human vs. AI Comparison

                > 🚫 **Decision was rejected** — no allocation executed.
                > The AI-only counterfactual is not applicable.
            """).rstrip()

        if outcome is None:
            return textwrap.dedent("""\
                ## 6. Human vs. AI Comparison

                > ⏳ Awaiting outcome data to compute human value-add.
            """).rstrip()

        actual = outcome.get("actual_return_30d")
        ai_only = outcome.get("ai_only_return_30d")

        if human_dec == "approve":
            comparison_block = textwrap.dedent(f"""\
                **Decision: Approved without modification**

                The human reviewer adopted the AI signal as-is.
                Human value-add analysis is only applicable to modified decisions.

                | Metric | Value |
                |--------|-------|
                | **Realized Return** | {_pct(actual, signed=True)} |
                | **Human Modification** | None |
                | **Value-Add** | N/A (approved without change) |
            """)
        elif human_dec == "modify":
            if value_add is not None and ai_only is not None:
                verdict = "🟢 Override helped" if value_add > 0 else "🔴 Override hurt"
                comparison_block = textwrap.dedent(f"""\
                    **Decision: Modified by human reviewer**

                    | Metric | Value |
                    |--------|-------|
                    | **Realized Return (human weights)** | {_pct(actual, signed=True)} |
                    | **Counterfactual Return (AI weights)** | {_pct(ai_only, signed=True)} |
                    | **Human Value-Add** | {_pct(value_add, signed=True)} |
                    | **Verdict** | {verdict} |

                    > *Human value-add = realized return − counterfactual AI return.*
                    > *Positive = the human modification improved the outcome.*
                """)
            else:
                comparison_block = textwrap.dedent(f"""\
                    **Decision: Modified by human reviewer**

                    | Metric | Value |
                    |--------|-------|
                    | **Realized Return (human weights)** | {_pct(actual, signed=True)} |
                    | **Counterfactual (AI-only)** | Not recorded |
                    | **Human Value-Add** | Cannot compute without counterfactual |

                    > To enable value-add analysis, record `--ai-only` when logging outcomes.
                """)
        else:
            comparison_block = "_No comparison available._"

        return f"## 6. Human vs. AI Comparison\n\n{comparison_block}"

    def _section_review_summary(self, trace: Dict, value_add: Optional[float]) -> str:
        """Section 7: Review Summary (rule-based or LLM-generated)."""
        if self.use_llm:
            context = self._build_llm_context(trace, value_add)
            prompt = (
                f"{context}\n\n"
                "Write a 3–4 paragraph **Review Summary** for this investment decision. "
                "Structure: (1) What the AI recommended and why (based on momentum). "
                "(2) How and why the human reviewer responded. "
                "(3) What actually happened and what it tells us about the strategy. "
                "Be specific — reference the actual numbers provided above. "
                "Do not use generic phrases like 'it's important to note'. "
                "Output markdown only, no section headers."
            )
            generated = _llm_generate_section(prompt, "review_summary")
            if generated:
                return f"## 7. Review Summary\n\n{generated}"

        # Fallback to rule-based
        narrative = _rule_based_summary(trace, value_add)
        return f"## 7. Review Summary\n\n{narrative}"

    def _section_lessons_learned(self, trace: Dict, value_add: Optional[float]) -> str:
        """Section 8: Lessons Learned (rule-based or LLM-generated)."""
        if self.use_llm:
            context = self._build_llm_context(trace, value_add)
            prompt = (
                f"{context}\n\n"
                "Write **3 specific, actionable lessons** from this investment decision. "
                "Each lesson should be concrete — avoid platitudes. "
                "Focus on: confidence calibration, human override quality, execution friction, "
                "or market regime considerations. "
                "Format as a markdown bullet list. Each bullet should be 2–3 sentences."
            )
            generated = _llm_generate_section(prompt, "lessons_learned")
            if generated:
                return f"## 8. Lessons Learned\n\n{generated}"

        # Fallback to rule-based
        narrative = _rule_based_lessons(trace, value_add)
        return f"## 8. Lessons Learned\n\n{narrative}"

    def _section_footer(self, decision_id: int) -> str:
        """Footer with navigation links."""
        return textwrap.dedent(f"""\
            ## Navigation

            ```bash
            # View all decisions
            python decisions/cli.py list

            # Log outcome for this decision (if pending)
            python decisions/cli.py outcome --id {decision_id} --return <value> --benchmark <spy_return>

            # Log execution fills
            python decisions/cli.py exec --id {decision_id} --symbol SPY --side buy --qty 10 --price 450.00

            # Run full analytics report
            python decisions/cli.py analyze

            # View trace for this decision
            python decisions/cli.py trace --id {decision_id}
            ```

            ---
            *AI Investment Research Copilot | Yuchuan Wu | Phase 2*
            *Code = Truth. LLM = Interface. Human = Final Authority.*
        """).rstrip()

    # ------------------------------------------------------------------
    # LLM context builder
    # ------------------------------------------------------------------

    def _build_llm_context(self, trace: Dict, value_add: Optional[float]) -> str:
        """
        Serialize the full trace into a compact prompt context for LLM.

        Only passes structured data — LLM reads but never computes metrics.

        Args:
            trace:     Full trace dict from get_full_trace().
            value_add: Pre-computed human value-add float.

        Returns:
            Formatted string block to prepend to LLM prompts.
        """
        decision = trace["decision"]
        outcome = trace.get("outcome")
        exec_summary = trace.get("execution_summary", {})

        context_dict = {
            "decision_id": decision.get("id"),
            "date": decision.get("date"),
            "strategy": decision.get("strategy"),
            "market_regime": trace.get("cycle", {}).get("market_regime") if trace.get("cycle") else None,
            "ai_signal": decision.get("ai_signal"),
            "ai_confidence": decision.get("ai_confidence"),
            "ai_confidence_method": decision.get("ai_confidence_method"),
            "ai_momentum_scores": decision.get("ai_momentum_scores"),
            "ai_selected_assets": decision.get("ai_selected_assets"),
            "human_decision": decision.get("human_decision"),
            "human_weights": decision.get("human_weights"),
            "human_reason": decision.get("human_reason"),
            "execution_summary": exec_summary,
            "outcome": {
                "actual_return_30d": outcome.get("actual_return_30d") if outcome else None,
                "benchmark_return_30d": outcome.get("benchmark_return_30d") if outcome else None,
                "ai_only_return_30d": outcome.get("ai_only_return_30d") if outcome else None,
                "ai_correct": outcome.get("ai_correct") if outcome else None,
                "human_value_add": value_add,
                "asset_returns": outcome.get("asset_returns") if outcome else None,
            },
        }
        return (
            "## Structured Decision Data (Code = Truth, do not modify these numbers)\n\n"
            "```json\n"
            + json.dumps(context_dict, indent=2, default=str)
            + "\n```"
        )

    def save_review(
        self,
        decision_id: int,
        output_dir: str = "reports",
    ) -> str:
        """
        Generate review and save to a Markdown file.

        Args:
            decision_id: Primary key of the decision.
            output_dir:  Directory to save the report. Created if absent.

        Returns:
            Path to the saved file as string.

        Raises:
            ValueError: If decision_id not found.
        """
        from pathlib import Path

        md = self.generate_review(decision_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"review_decision_{decision_id}_{timestamp}.md"
        filepath = Path(output_dir) / filename
        filepath.write_text(md, encoding="utf-8")
        logger.info("Review saved to: %s", filepath)
        return str(filepath)
