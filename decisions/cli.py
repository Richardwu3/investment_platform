#!/usr/bin/env python3
"""
decisions/cli.py
================
Command-line interface for the AI Investment Decision Journal.

This is the Human-in-the-Loop interface: it records AI recommendations,
captures human decisions (approve / modify / reject), and later updates
realized outcomes to close the feedback loop.

Commands:
    log      — Record a new AI recommendation and human decision
    list     — View decision history with optional filters
    outcome  — Record realized return for a past decision
    pending  — List decisions awaiting outcome entry
    analyze  — Print full analytics report (adoption rate, AI accuracy, etc.)
    seed     — Load sample data for demo / testing purposes

Usage Examples:
    # Log a decision interactively
    python decisions/cli.py log

    # Log with pre-filled arguments (non-interactive)
    python decisions/cli.py log \\
        --date 2024-01-31 \\
        --strategy GTAA_126d_Top2 \\
        --ai-signal '{"SPY": 0.5, "TLT": 0.5}' \\
        --ai-confidence 0.72 \\
        --human-decision approve

    # List last 20 decisions
    python decisions/cli.py list --limit 20

    # List only modified decisions
    python decisions/cli.py list --filter modify

    # Record outcome for decision #3
    python decisions/cli.py outcome --id 3 --return 0.034 --benchmark 0.021

    # Show what decisions need outcomes
    python decisions/cli.py pending

    # Full analytics report
    python decisions/cli.py analyze

    # Load demo data
    python decisions/cli.py seed

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decisions.db import Database, DecisionRepository
from decisions.analyzer import DecisionAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "decisions/journal.db"

logging.basicConfig(
    level=logging.WARNING,  # CLI stays quiet; only errors surface
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# ANSI color codes — auto-disabled on Windows or non-TTY
USE_COLOR = sys.stdout.isatty() and sys.platform != "win32"

def _c(text: str, code: str) -> str:
    """Wrap text in ANSI color if terminal supports it."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def bold(t: str)   -> str: return _c(t, "1")
def green(t: str)  -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str)    -> str: return _c(t, "31")
def cyan(t: str)   -> str: return _c(t, "36")
def dim(t: str)    -> str: return _c(t, "2")

def pct(value: Optional[float], decimals: int = 2) -> str:
    """Format float as percentage string."""
    if value is None:
        return dim("N/A")
    return f"{value * 100:.{decimals}f}%"

def ratio(value: Optional[float], decimals: int = 3) -> str:
    """Format float as ratio string."""
    if value is None:
        return dim("N/A")
    return f"{value:.{decimals}f}"

def human_decision_color(decision: str) -> str:
    """Color-code human decision type."""
    mapping = {"approve": green, "modify": yellow, "reject": red}
    fn = mapping.get(decision, lambda x: x)
    return fn(decision.upper())

def ai_correct_color(ai_correct: Optional[str]) -> str:
    """Color-code AI accuracy result."""
    if ai_correct is None:
        return dim("pending")
    mapping = {
        "direction_correct": green("✓ correct"),
        "direction_wrong":   red("✗ wrong"),
        "inconclusive":      yellow("~ inconclusive"),
    }
    return mapping.get(ai_correct, dim(ai_correct))

def separator(char: str = "─", width: int = 72) -> str:
    """Return a horizontal separator line."""
    return dim(char * width)


# ---------------------------------------------------------------------------
# Interactive input helpers
# ---------------------------------------------------------------------------

def prompt(message: str, default: Optional[str] = None, required: bool = True) -> str:
    """
    Prompt user for text input with optional default.

    Args:
        message:  Prompt text shown to user.
        default:  Pre-filled value (shown in brackets). If user presses Enter, used.
        required: If True, re-prompt until non-empty input received.

    Returns:
        User-provided string (stripped), or default if Enter pressed.
    """
    display = f"{message} [{default}]: " if default else f"{message}: "
    while True:
        value = input(display).strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print(red("  ✗ This field is required."))

def prompt_choice(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Prompt user to select from a fixed list of choices.

    Args:
        message:  Prompt text.
        choices:  Valid options.
        default:  Default choice if user presses Enter.

    Returns:
        Selected choice string.
    """
    options_str = " / ".join(
        bold(c) if c == default else c for c in choices
    )
    display = f"{message} ({options_str}): "
    while True:
        value = input(display).strip().lower()
        if not value and default:
            return default
        if value in choices:
            return value
        print(red(f"  ✗ Invalid choice. Please enter one of: {', '.join(choices)}"))

def prompt_json(message: str, default: Optional[str] = None) -> Dict:
    """
    Prompt user for a JSON string and parse it.

    Args:
        message: Prompt text.
        default: Default JSON string.

    Returns:
        Parsed Python dict.

    Raises:
        SystemExit: If user provides invalid JSON after 3 attempts.
    """
    example = '  e.g. {"SPY": 0.5, "TLT": 0.5}'
    for attempt in range(3):
        raw = prompt(f"{message}\n{dim(example)}", default=default)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(red(f"  ✗ Invalid JSON: {e}"))
    print(red("  ✗ Failed 3 times. Aborting."))
    sys.exit(1)

def prompt_float(
    message: str,
    default: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    required: bool = True,
) -> Optional[float]:
    """
    Prompt user for a float value with optional range validation.

    Args:
        message:  Prompt text.
        default:  Default float value.
        min_val:  Minimum accepted value (inclusive).
        max_val:  Maximum accepted value (inclusive).
        required: If False, returns None on empty input.

    Returns:
        Parsed float, or None if not required and empty.
    """
    default_str = str(default) if default is not None else None
    while True:
        raw = prompt(message, default=default_str, required=required)
        if not raw and not required:
            return None
        try:
            value = float(raw)
            if min_val is not None and value < min_val:
                print(red(f"  ✗ Must be ≥ {min_val}"))
                continue
            if max_val is not None and value > max_val:
                print(red(f"  ✗ Must be ≤ {max_val}"))
                continue
            return value
        except ValueError:
            print(red("  ✗ Must be a number."))


# ---------------------------------------------------------------------------
# Command: log
# ---------------------------------------------------------------------------

def cmd_log(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Interactively (or via arguments) record a new AI recommendation and human decision.

    Interactive mode is triggered when key arguments are missing.
    All fields from the decisions schema are captured.

    Args:
        args: Parsed CLI arguments (may be partially filled).
        repo: DecisionRepository instance.
    """
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('LOG NEW DECISION')}")
    print(f"{bold('═' * 72)}\n")

    # --- Date ---
    today = date.today().strftime("%Y-%m-%d")
    d = args.date or prompt("Signal date (YYYY-MM-DD)", default=today)

    # --- Strategy ---
    strategy = args.strategy or prompt("Strategy name", default="GTAA_126d_Top2")

    # --- AI Signal ---
    if args.ai_signal:
        try:
            ai_signal = json.loads(args.ai_signal)
        except json.JSONDecodeError:
            print(red("  ✗ --ai-signal must be valid JSON."))
            sys.exit(1)
    else:
        print(f"\n  {cyan('AI Recommendation')}")
        ai_signal = prompt_json("AI-recommended weights (JSON)")

    # --- AI Confidence ---
    if args.ai_confidence is not None:
        ai_confidence = args.ai_confidence
        ai_confidence_method = "cli_argument"
    else:
        print(f"\n  {cyan('AI Confidence')} {dim('(0.0 = no confidence, 1.0 = max)')}")
        ai_confidence = prompt_float(
            "AI confidence score", min_val=0.0, max_val=1.0, required=False
        )
        ai_confidence_method = "momentum_spread" if ai_confidence is not None else None
        if ai_confidence is not None:
            print(dim("  Confidence method: momentum_spread (Top1 score / Top3 average)"))

    # --- AI Momentum Scores ---
    print(f"\n  {cyan('Momentum Scores')} {dim('(optional — press Enter to skip)')}")
    raw_scores = prompt("Momentum scores JSON", default="", required=False)
    ai_momentum_scores = None
    if raw_scores:
        try:
            ai_momentum_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print(yellow("  ⚠ Could not parse momentum scores; skipping."))

    ai_selected_assets = list(ai_signal.keys()) if ai_signal else None

    # --- Human Decision ---
    print(f"\n  {cyan('Human Decision')}")
    if args.human_decision:
        human_decision = args.human_decision
        if human_decision not in ("approve", "modify", "reject"):
            print(red("  ✗ --human-decision must be approve, modify, or reject."))
            sys.exit(1)
    else:
        print(f"  AI recommends: {bold(json.dumps(ai_signal))}")
        human_decision = prompt_choice(
            "  Your decision", choices=["approve", "modify", "reject"], default="approve"
        )

    # --- Human Weights (only if modify) ---
    human_weights = None
    human_reason = None

    if human_decision == "modify":
        print(f"\n  {cyan('Modified Weights')}")
        print(f"  {dim('Original: ' + json.dumps(ai_signal))}")
        human_weights = prompt_json("  Your modified weights (JSON)")
        human_reason = prompt("  Reason for modification", required=True)

    elif human_decision == "reject":
        human_reason = prompt("\n  Reason for rejection", required=True)

    # --- Market Regime (optional) ---
    print(f"\n  {cyan('Market Context')} {dim('(optional)')}")
    regime_raw = prompt("  Market regime (bull/bear/sideways)", default="", required=False)
    market_regime = regime_raw if regime_raw in ("bull", "bear", "sideways") else None

    # --- Create cycle and decision ---
    cycle_id = repo.create_cycle(
        cycle_date=d,
        strategy=strategy,
        market_regime=market_regime,
    )
    decision_id = repo.log_decision(
        date=d,
        strategy=strategy,
        ai_signal=ai_signal,
        human_decision=human_decision,
        ai_confidence=ai_confidence,
        ai_confidence_method=ai_confidence_method,
        ai_momentum_scores=ai_momentum_scores,
        ai_selected_assets=ai_selected_assets,
        human_weights=human_weights,
        human_reason=human_reason,
        cycle_id=cycle_id,
    )

    print(f"\n{separator()}")
    print(f"  {green('✓ Decision logged.')}  ID: {bold(str(decision_id))}")
    print(f"  Date: {d}  |  Strategy: {strategy}")
    print(f"  Human decision: {human_decision_color(human_decision)}")
    if ai_confidence is not None:
        print(f"  AI confidence: {pct(ai_confidence)}")
    print(f"\n  {dim('Record outcome later with:')}")
    print(f"  {cyan(f'python decisions/cli.py outcome --id {decision_id} --return <value> --benchmark <value>')}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Display decision history in a formatted table.

    Args:
        args: Parsed arguments (limit, filter, strategy).
        repo: DecisionRepository instance.
    """
    decisions = repo.list_decisions(
        strategy=args.strategy,
        human_decision=args.filter,
        limit=args.limit,
        offset=args.offset,
    )

    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('DECISION HISTORY')}  {dim(f'(showing {len(decisions)} records)')}")
    print(bold('═' * 72))

    if not decisions:
        print(f"\n  {dim('No decisions recorded yet.')}")
        print(f"  {dim('Run: python decisions/cli.py log')}\n")
        return

    # Table header
    print(f"\n  {'ID':<5} {'Date':<12} {'Strategy':<20} {'AI Signal':<24} "
          f"{'Human':<10} {'Conf':<7} {'30d Ret':<10} {'AI':<14}")
    print(f"  {separator('─', 68)}")

    for r in decisions:
        ai_sig = r.get("ai_signal") or {}
        # Format weights compactly: SPY:50% TLT:50%
        sig_str = " ".join(f"{k}:{v*100:.0f}%" for k, v in ai_sig.items())[:22]

        actual_ret = r.get("actual_return_30d")
        ret_str = pct(actual_ret) if actual_ret is not None else dim("pending")
        ai_acc = ai_correct_color(r.get("ai_correct"))

        conf = r.get("ai_confidence")
        conf_str = pct(conf, 0) if conf is not None else dim("—")

        print(
            f"  {str(r['id']):<5} {str(r['date']):<12} {str(r['strategy'])[:18]:<20} "
            f"{sig_str:<24} {human_decision_color(r['human_decision']):<20} "
            f"{conf_str:<7} {ret_str:<10} {ai_acc}"
        )

    print(f"\n  {dim(f'Total shown: {len(decisions)}')}")
    if len(decisions) == args.limit:
        print(f"  {dim(f'Use --offset {args.offset + args.limit} to see more.')}")
    print()


# ---------------------------------------------------------------------------
# Command: outcome
# ---------------------------------------------------------------------------

def cmd_outcome(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Record the realized return for a past decision.

    Supports both interactive and argument-driven modes.

    Args:
        args: Parsed arguments (id, return, benchmark, asset-returns).
        repo: DecisionRepository instance.
    """
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('RECORD OUTCOME')}")
    print(bold('═' * 72) + "\n")

    # --- Get decision ID ---
    if args.id:
        decision_id = args.id
    else:
        pending = repo.get_pending_outcomes()
        if not pending:
            print(f"  {green('✓ No pending outcomes.')} All recorded decisions have outcomes.\n")
            return
        print(f"  {yellow(f'{len(pending)} decisions awaiting outcomes:')}\n")
        for r in pending[:10]:
            sig_str = " ".join(f"{k}:{v*100:.0f}%" for k, v in (r.get("ai_signal") or {}).items())
            print(f"    [{r['id']}] {r['date']} | {r['strategy']} | {sig_str}")
        print()
        decision_id = int(prompt("Decision ID to update"))

    decision = repo.get_decision(decision_id)
    if decision is None:
        print(red(f"  ✗ Decision id={decision_id} not found."))
        sys.exit(1)

    print(f"  Decision #{decision_id}: {decision['date']} | {decision['strategy']}")
    print(f"  AI signal:       {json.dumps(decision.get('ai_signal', {}))}")
    print(f"  Human decision:  {human_decision_color(decision['human_decision'])}")
    if decision.get("human_weights"):
        print(f"  Final weights:   {json.dumps(decision['human_weights'])}")
    print()

    # --- Realized return ---
    if args.ret is not None:
        actual_return = args.ret
    else:
        actual_return = prompt_float(
            "  Realized portfolio return (e.g. 0.034 for +3.4%)",
            min_val=-1.0, max_val=10.0
        )

    # --- Benchmark return ---
    if args.benchmark is not None:
        benchmark_return = args.benchmark
    else:
        benchmark_return = prompt_float(
            "  SPY benchmark return over same period (optional)",
            required=False
        )

    # --- AI-only counterfactual ---
    ai_only_return = None
    if args.ai_only is not None:
        ai_only_return = args.ai_only
    elif decision.get("human_decision") == "modify":
        print(f"\n  {cyan('Counterfactual')} {dim('(what pure AI weights would have returned)')}")
        ai_only_return = prompt_float(
            "  AI-only return (optional — enables human value-add calculation)",
            required=False
        )

    # --- Per-asset returns ---
    asset_returns = None
    raw_assets = prompt("\n  Per-asset returns JSON (optional, press Enter to skip)", required=False)
    if raw_assets:
        try:
            asset_returns = json.loads(raw_assets)
        except json.JSONDecodeError:
            print(yellow("  ⚠ Could not parse; skipping per-asset returns."))

    notes = prompt("  Notes (optional)", required=False)

    outcome_id = repo.log_outcome(
        decision_id=decision_id,
        actual_return_30d=actual_return,
        benchmark_return_30d=benchmark_return,
        asset_returns=asset_returns,
        ai_only_return_30d=ai_only_return,
        notes=notes or None,
    )

    # Determine if this was a good call
    threshold = 0.005
    if abs(actual_return) < threshold:
        verdict = yellow("~ inconclusive (return < 0.5%)")
    elif actual_return > 0:
        verdict = green("✓ direction correct")
    else:
        verdict = red("✗ direction wrong")

    print(f"\n{separator()}")
    print(f"  {green('✓ Outcome recorded.')}  Outcome ID: {bold(str(outcome_id))}")
    print(f"  Realized return:   {bold(pct(actual_return))}")
    if benchmark_return is not None:
        print(f"  Benchmark (SPY):   {pct(benchmark_return)}  |  Excess: {pct(actual_return - benchmark_return)}")
    if ai_only_return is not None:
        human_va = actual_return - ai_only_return
        va_str = green(pct(human_va)) if human_va > 0 else red(pct(human_va))
        print(f"  AI-only return:    {pct(ai_only_return)}  |  Human value-add: {va_str}")
    print(f"  AI verdict:        {verdict}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: pending
# ---------------------------------------------------------------------------

def cmd_pending(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    List decisions that are old enough to record outcomes but haven't been updated.

    Args:
        args: Parsed arguments (days).
        repo: DecisionRepository instance.
    """
    pending = repo.get_pending_outcomes(days_threshold=args.days)

    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('PENDING OUTCOMES')}  {dim(f'(decisions ≥ {args.days} days old)')}")
    print(bold('═' * 72))

    if not pending:
        print(f"\n  {green('✓ No pending outcomes.')}\n")
        return

    print(f"\n  {yellow(f'{len(pending)} decisions need outcome data:')}\n")
    print(f"  {'ID':<6} {'Date':<12} {'Strategy':<22} {'AI Signal':<28} {'Human'}")
    print(f"  {separator('─', 66)}")

    for r in pending:
        sig_str = " ".join(
            f"{k}:{v*100:.0f}%" for k, v in (r.get("ai_signal") or {}).items()
        )[:26]
        print(
            f"  {str(r['id']):<6} {str(r['date']):<12} {str(r['strategy'])[:20]:<22} "
            f"{sig_str:<28} {human_decision_color(r['human_decision'])}"
        )

    print(f"\n  {dim('Record outcomes with:')}")
    print(f"  {cyan('python decisions/cli.py outcome --id <ID> --return <value>')}\n")


# ---------------------------------------------------------------------------
# Command: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Print a comprehensive analytics report of the decision journal.

    Covers: adoption rates, AI accuracy, human value-add, confidence
    calibration, and strategy breakdown.

    Args:
        args: Parsed arguments (currently unused, reserved for filters).
        repo: DecisionRepository instance.
    """
    records = repo.get_all_outcomes()
    analyzer = DecisionAnalyzer(records)
    report = analyzer.full_report()

    adoption = report["adoption_rates"]
    accuracy = report["ai_accuracy"]
    hva = report["human_value_add"]
    calib = report["confidence_calibration"]

    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('DECISION JOURNAL — ANALYTICS REPORT')}")
    gen_at = report["generated_at"][:19]
    print(f"  {dim(f'Generated: {gen_at}')}")
    print(bold('═' * 72))

    # --- Adoption Rates ---
    print(f"\n  {bold('[ HUMAN DECISION BREAKDOWN ]')}")
    print(f"  Total decisions:    {bold(str(adoption['total_decisions']))}")
    if adoption["total_decisions"] > 0:
        print(f"  Approved (adopt):   {green(str(adoption['approved']))}  "
              f"{dim(pct(adoption['adoption_rate']))}")
        print(f"  Modified:           {yellow(str(adoption['modified']))}  "
              f"{dim(pct(adoption['modification_rate']))}")
        print(f"  Rejected:           {red(str(adoption['rejected']))}  "
              f"{dim(pct(adoption['rejection_rate']))}")

    # --- AI Accuracy ---
    print(f"\n  {bold('[ AI ACCURACY ]')}")
    print(f"  Decisions with outcomes:  {accuracy['n_with_outcomes']}")
    print(f"  Conclusive outcomes:      {accuracy['n_conclusive']}")
    if accuracy["ai_accuracy_rate"] is not None:
        acc = accuracy["ai_accuracy_rate"]
        acc_display = green(pct(acc)) if acc >= 0.6 else yellow(pct(acc)) if acc >= 0.5 else red(pct(acc))
        print(f"  {bold('AI Accuracy Rate:')}        {acc_display}")
        print(f"    Correct calls:  {green(str(accuracy['n_correct']))}  "
              f"| Mean return: {pct(accuracy['mean_return_correct'])}")
        print(f"    Wrong calls:    {red(str(accuracy['n_wrong']))}  "
              f"| Mean return: {pct(accuracy['mean_return_wrong'])}")
        print(f"    Inconclusive:   {yellow(str(accuracy['n_inconclusive']))}")
    else:
        print(f"  {dim('No conclusive outcomes yet.')}")
    print(f"  Pending outcomes:         {report['pending_outcomes_count']}")

    # --- Human Value-Add ---
    print(f"\n  {bold('[ HUMAN VALUE-ADD (modification decisions) ]')}")
    if hva["n_modify_with_outcomes"] > 0:
        mean_va = hva["mean_human_value_add"]
        va_display = green(pct(mean_va)) if mean_va and mean_va > 0 else red(pct(mean_va))
        print(f"  Modifications with outcomes: {hva['n_modify_with_outcomes']}")
        print(f"  {bold('Mean value-add per trade:')} {va_display}")
        print(f"  % modifications helpful:   {pct(hva['pct_modifications_helpful'])}")
        print(f"  Cumulative human alpha:    {pct(hva['total_human_alpha'])}")
    else:
        print(f"  {dim('No modification outcomes recorded yet.')}")

    # --- Confidence Calibration ---
    print(f"\n  {bold('[ CONFIDENCE CALIBRATION ]')}")
    if calib["has_confidence_data"]:
        print(f"  Confidence ↔ Return correlation: {bold(ratio(calib['correlation']))}")
        print(f"  {dim(calib['interpretation'])}\n")
        print(f"  {'Tercile':<24} {'N':<6} {'Accuracy':<12} {'Mean Return'}")
        print(f"  {separator('─', 55)}")
        for t in calib["tercile_analysis"]:
            acc_str = pct(t["accuracy"]) if t["accuracy"] is not None else dim("N/A")
            ret_str = pct(t["mean_return"])
            print(f"  {t['label']:<24} {t['n']:<6} {acc_str:<12} {ret_str}")
    else:
        print(f"  {dim(calib['interpretation'])}")

    # --- Strategy Breakdown ---
    by_strat = report["by_strategy"]
    if len(by_strat) > 1:
        print(f"\n  {bold('[ BY STRATEGY ]')}")
        for strat, data in by_strat.items():
            acc_rate = data["ai_accuracy"]["ai_accuracy_rate"]
            adopt_rate = data["adoption_rates"]["adoption_rate"]
            print(f"  {bold(strat)}: {data['n_decisions']} decisions | "
                  f"Adopt: {pct(adopt_rate)} | Accuracy: {pct(acc_rate)}")

    print(f"\n{separator()}")
    print(f"  {dim('Pass report dict to AI Agent for narrative generation.')}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: seed
# ---------------------------------------------------------------------------

def cmd_seed(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Load sample data into the database for demonstration and testing.

    Creates 8 realistic decisions across several months with a mix of
    approve / modify / reject, and fills in outcomes for the older ones.
    Safe to run multiple times (checks for existing data first).

    Args:
        args: Parsed arguments (force flag).
        repo: DecisionRepository instance.
    """
    existing = repo.list_decisions(limit=1)
    if existing and not args.force:
        print(f"\n  {yellow('⚠ Database already contains decisions.')}")
        print(f"  Use {cyan('--force')} to add seed data anyway.\n")
        return

    print(f"\n  {cyan('Seeding demo data...')}\n")

    SAMPLE_DECISIONS = [
        {
            "date": "2024-01-31", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "QQQ": 0.5}, "ai_confidence": 0.81,
            "ai_momentum_scores": {"SPY": 0.089, "QQQ": 0.072, "TLT": -0.031, "GLD": 0.012, "DBC": -0.044},
            "ai_selected_assets": ["SPY", "QQQ"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "bull",
            "outcome": {"actual_return_30d": 0.041, "benchmark_return_30d": 0.038,
                        "ai_only_return_30d": 0.041, "asset_returns": {"SPY": 0.038, "QQQ": 0.044}},
        },
        {
            "date": "2024-02-29", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "QQQ": 0.5}, "ai_confidence": 0.76,
            "ai_momentum_scores": {"SPY": 0.071, "QQQ": 0.063, "TLT": -0.018, "GLD": 0.031, "DBC": -0.022},
            "ai_selected_assets": ["SPY", "QQQ"],
            "human_decision": "modify",
            "human_weights": {"SPY": 0.5, "GLD": 0.5},
            "human_reason": "Concerned about tech concentration; adding gold as hedge given macro uncertainty.",
            "market_regime": "bull",
            "outcome": {"actual_return_30d": 0.028, "benchmark_return_30d": 0.032,
                        "ai_only_return_30d": 0.021, "asset_returns": {"SPY": 0.032, "GLD": 0.024}},
        },
        {
            "date": "2024-03-28", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"QQQ": 0.5, "GLD": 0.5}, "ai_confidence": 0.62,
            "ai_momentum_scores": {"SPY": 0.055, "QQQ": 0.068, "TLT": -0.042, "GLD": 0.057, "DBC": 0.011},
            "ai_selected_assets": ["QQQ", "GLD"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "bull",
            "outcome": {"actual_return_30d": 0.019, "benchmark_return_30d": 0.024,
                        "ai_only_return_30d": 0.019, "asset_returns": {"QQQ": 0.023, "GLD": 0.015}},
        },
        {
            "date": "2024-04-30", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"TLT": 0.5, "GLD": 0.5}, "ai_confidence": 0.44,
            "ai_momentum_scores": {"SPY": -0.012, "QQQ": -0.021, "TLT": 0.018, "GLD": 0.034, "DBC": -0.007},
            "ai_selected_assets": ["TLT", "GLD"],
            "human_decision": "reject",
            "human_reason": "Low confidence (44%). Fed meeting next week creates bond uncertainty; going to cash.",
            "market_regime": "sideways",
            "outcome": None,  # rejected — no outcome to track
        },
        {
            "date": "2024-05-31", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "GLD": 0.5}, "ai_confidence": 0.69,
            "ai_momentum_scores": {"SPY": 0.061, "QQQ": 0.042, "TLT": -0.028, "GLD": 0.073, "DBC": 0.019},
            "ai_selected_assets": ["SPY", "GLD"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "bull",
            "outcome": {"actual_return_30d": -0.022, "benchmark_return_30d": -0.019,
                        "ai_only_return_30d": -0.022, "asset_returns": {"SPY": -0.018, "GLD": -0.026}},
        },
        {
            "date": "2024-06-28", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "QQQ": 0.5}, "ai_confidence": 0.83,
            "ai_momentum_scores": {"SPY": 0.094, "QQQ": 0.088, "TLT": -0.019, "GLD": 0.041, "DBC": -0.033},
            "ai_selected_assets": ["SPY", "QQQ"],
            "human_decision": "modify",
            "human_weights": {"SPY": 0.7, "QQQ": 0.3},
            "human_reason": "Overweight SPY for broader exposure; QQQ showing some mean-reversion signals.",
            "market_regime": "bull",
            "outcome": {"actual_return_30d": 0.033, "benchmark_return_30d": 0.028,
                        "ai_only_return_30d": 0.031, "asset_returns": {"SPY": 0.034, "QQQ": 0.029}},
        },
        {
            "date": "2024-07-31", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "TLT": 0.5}, "ai_confidence": 0.58,
            "ai_momentum_scores": {"SPY": 0.048, "QQQ": 0.031, "TLT": 0.044, "GLD": 0.039, "DBC": -0.011},
            "ai_selected_assets": ["SPY", "TLT"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "sideways",
            "outcome": {"actual_return_30d": 0.003, "benchmark_return_30d": 0.011,
                        "ai_only_return_30d": 0.003, "asset_returns": {"SPY": 0.009, "TLT": -0.003}},
        },
        {
            "date": "2024-08-30", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"GLD": 0.5, "DBC": 0.5}, "ai_confidence": 0.52,
            "ai_momentum_scores": {"SPY": 0.011, "QQQ": -0.008, "TLT": 0.022, "GLD": 0.059, "DBC": 0.048},
            "ai_selected_assets": ["GLD", "DBC"],
            "human_decision": "modify",
            "human_weights": {"GLD": 0.5, "TLT": 0.5},
            "human_reason": "Replacing DBC with TLT; DBC momentum driven by oil spike, prefer bonds for risk-off.",
            "market_regime": "sideways",
            "outcome": None,  # recent — no outcome yet
        },
    ]

    ids_created = []
    for record in SAMPLE_DECISIONS:
        cycle_id = repo.create_cycle(
            cycle_date=record["date"],
            strategy=record["strategy"],
            market_regime=record.get("market_regime"),
        )
        decision_id = repo.log_decision(
            date=record["date"],
            strategy=record["strategy"],
            ai_signal=record["ai_signal"],
            human_decision=record["human_decision"],
            ai_confidence=record.get("ai_confidence"),
            ai_confidence_method="momentum_spread",
            ai_momentum_scores=record.get("ai_momentum_scores"),
            ai_selected_assets=record.get("ai_selected_assets"),
            human_weights=record.get("human_weights"),
            human_reason=record.get("human_reason"),
            cycle_id=cycle_id,
        )
        ids_created.append(decision_id)

        if record.get("outcome") and record["human_decision"] != "reject":
            o = record["outcome"]
            repo.log_outcome(
                decision_id=decision_id,
                actual_return_30d=o["actual_return_30d"],
                benchmark_return_30d=o.get("benchmark_return_30d"),
                asset_returns=o.get("asset_returns"),
                ai_only_return_30d=o.get("ai_only_return_30d"),
            )
        print(f"  {green('✓')} [{decision_id}] {record['date']} | {record['human_decision']}", end="")
        if record.get("outcome"):
            ret = record["outcome"]["actual_return_30d"]
            ret_str = green(pct(ret)) if ret > 0 else red(pct(ret))
            print(f" | outcome: {ret_str}", end="")
        print()

    print(f"\n  {green(f'✓ Seeded {len(ids_created)} decisions.')}")
    print(f"  {dim('Run: python decisions/cli.py analyze')}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="python decisions/cli.py",
        description="AI Investment Decision Journal — Columbia MAFN Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decisions/cli.py log                              # interactive log
  python decisions/cli.py log --date 2024-01-31 \\
      --ai-signal '{"SPY": 0.5, "TLT": 0.5}' \\
      --human-decision approve
  python decisions/cli.py list --limit 20
  python decisions/cli.py list --filter modify
  python decisions/cli.py outcome --id 3 --return 0.034 --benchmark 0.021
  python decisions/cli.py pending
  python decisions/cli.py analyze
  python decisions/cli.py seed
        """,
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # log
    p_log = sub.add_parser("log", help="Record a new AI recommendation and human decision")
    p_log.add_argument("--date", help="Signal date YYYY-MM-DD (default: today)")
    p_log.add_argument("--strategy", help="Strategy name (default: GTAA_126d_Top2)")
    p_log.add_argument("--ai-signal", dest="ai_signal", help='JSON weights e.g. \'{"SPY":0.5,"TLT":0.5}\'')
    p_log.add_argument("--ai-confidence", dest="ai_confidence", type=float, help="Confidence score 0.0–1.0")
    p_log.add_argument("--human-decision", dest="human_decision",
                       choices=["approve", "modify", "reject"], help="Human decision type")

    # list
    p_list = sub.add_parser("list", help="View decision history")
    p_list.add_argument("--limit", type=int, default=20, help="Max decisions to show")
    p_list.add_argument("--offset", type=int, default=0, help="Skip first N decisions")
    p_list.add_argument("--filter", choices=["approve", "modify", "reject"], help="Filter by decision type")
    p_list.add_argument("--strategy", help="Filter by strategy name")

    # outcome
    p_out = sub.add_parser("outcome", help="Record realized return for a past decision")
    p_out.add_argument("--id", type=int, help="Decision ID to update")
    p_out.add_argument("--return", dest="ret", type=float, help="Realized return (e.g. 0.034 for +3.4%%)")
    p_out.add_argument("--benchmark", type=float, help="SPY benchmark return over same window")
    p_out.add_argument("--ai-only", dest="ai_only", type=float,
                       help="Counterfactual return with pure AI weights (enables value-add calc)")

    # pending
    p_pend = sub.add_parser("pending", help="List decisions awaiting outcome data")
    p_pend.add_argument("--days", type=int, default=30, help="Minimum age in days before outcome is due")

    # analyze
    sub.add_parser("analyze", help="Print full analytics report")

    # seed
    p_seed = sub.add_parser("seed", help="Load sample data for demo/testing")
    p_seed.add_argument("--force", action="store_true", help="Add seed data even if DB is non-empty")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, initialize database, and dispatch to the correct command handler.

    Entry point for the CLI. All command handlers receive:
        - args: populated Namespace from argparse
        - repo: DecisionRepository wired to the configured database
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db = Database(args.db)
    repo = DecisionRepository(db)

    command_map = {
        "log":     cmd_log,
        "list":    cmd_list,
        "outcome": cmd_outcome,
        "pending": cmd_pending,
        "analyze": cmd_analyze,
        "seed":    cmd_seed,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args, repo)
    except KeyboardInterrupt:
        print(f"\n\n  {dim('Aborted.')}\n")
        sys.exit(0)
    except Exception as e:
        if args.verbose:
            raise
        print(red(f"\n  ✗ Error: {e}\n"))
        sys.exit(1)


if __name__ == "__main__":
    main()
