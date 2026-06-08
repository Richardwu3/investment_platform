#!/usr/bin/env python3
"""
decisions/cli.py
================
Command-line interface for the AI Investment Decision Journal.

This is the Human-in-the-Loop interface: it records AI recommendations,
captures human decisions (approve / modify / reject), tracks broker-level
fills, and generates post-decision review reports.

Commands:
    log      — Record a new AI recommendation and human decision
    list     — View decision history with optional filters
    outcome  — Record realized return for a past decision
    pending  — List decisions awaiting outcome entry
    analyze  — Print full analytics report
    exec     — [NEW] Record a broker fill for an approved/modified decision
    trace    — [NEW] Display the complete decision lifecycle as a timeline
    review   — [NEW] Generate a Markdown post-decision review report
    seed     — Load sample data for demo / testing

Usage Examples:
    python decisions/cli.py log
    python decisions/cli.py list --limit 20
    python decisions/cli.py list --filter modify
    python decisions/cli.py outcome --id 3 --return 0.034 --benchmark 0.021
    python decisions/cli.py pending
    python decisions/cli.py analyze

    # New in Phase 2.2:
    python decisions/cli.py exec --id 2 --symbol SPY --side buy --qty 10.5 --price 450.25
    python decisions/cli.py exec --id 2 --symbol TLT --side buy --qty 8 --price 98.10 --commission 1.00
    python decisions/cli.py trace --id 2
    python decisions/cli.py review --id 2
    python decisions/cli.py review --id 2 --llm          # LLM-enhanced (needs ANTHROPIC_API_KEY)
    python decisions/cli.py review --id 2 --save         # save to reports/ directory
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decisions.db import Database, DecisionRepository
from decisions.analyzer import DecisionAnalyzer
from decisions.review_copilot import DecisionReviewCopilot

DEFAULT_DB_PATH = "decisions/journal.db"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and sys.platform != "win32"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def bold(t: str)   -> str: return _c(t, "1")
def green(t: str)  -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str)    -> str: return _c(t, "31")
def cyan(t: str)   -> str: return _c(t, "36")
def dim(t: str)    -> str: return _c(t, "2")
def blue(t: str)   -> str: return _c(t, "34")
def magenta(t: str) -> str: return _c(t, "35")

def pct(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    if value is None:
        return dim("N/A")
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"

def ratio(value: Optional[float], decimals: int = 3) -> str:
    if value is None:
        return dim("N/A")
    return f"{value:.{decimals}f}"

def human_decision_color(decision: str) -> str:
    mapping = {"approve": green, "modify": yellow, "reject": red}
    return mapping.get(decision, lambda x: x)(decision.upper())

def ai_correct_color(ai_correct: Optional[str]) -> str:
    if ai_correct is None:
        return dim("pending")
    return {
        "direction_correct": green("✓ correct"),
        "direction_wrong":   red("✗ wrong"),
        "inconclusive":      yellow("~ inconclusive"),
    }.get(ai_correct, dim(ai_correct))

def separator(char: str = "─", width: int = 72) -> str:
    return dim(char * width)

def side_color(side: str) -> str:
    return green(side.upper()) if side == "buy" else red(side.upper())


# ---------------------------------------------------------------------------
# Interactive input helpers
# ---------------------------------------------------------------------------

def prompt(message: str, default: Optional[str] = None, required: bool = True) -> str:
    display = f"{message} [{default}]: " if default else f"{message}: "
    while True:
        value = input(display).strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print(red("  ✗ This field is required."))

def prompt_choice(message: str, choices: List[str], default: Optional[str] = None) -> str:
    options_str = " / ".join(bold(c) if c == default else c for c in choices)
    display = f"{message} ({options_str}): "
    while True:
        value = input(display).strip().lower()
        if not value and default:
            return default
        if value in choices:
            return value
        print(red(f"  ✗ Invalid choice. Must be one of: {', '.join(choices)}"))

def prompt_json(message: str, default: Optional[str] = None) -> Dict:
    example = "  e.g. {\"SPY\": 0.5, \"TLT\": 0.5}"
    for _ in range(3):
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
    Interactively record a new AI recommendation and human decision.

    Args:
        args: Parsed CLI arguments (may be partially pre-filled).
        repo: DecisionRepository instance.
    """
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('LOG NEW DECISION')}")
    print(f"{bold('═' * 72)}\n")

    today = date.today().strftime("%Y-%m-%d")
    d = args.date or prompt("Signal date (YYYY-MM-DD)", default=today)
    strategy = args.strategy or prompt("Strategy name", default="GTAA_126d_Top2")

    if args.ai_signal:
        try:
            ai_signal = json.loads(args.ai_signal)
        except json.JSONDecodeError:
            print(red("  ✗ --ai-signal must be valid JSON."))
            sys.exit(1)
    else:
        print(f"\n  {cyan('AI Recommendation')}")
        ai_signal = prompt_json("AI-recommended weights (JSON)")

    if args.ai_confidence is not None:
        ai_confidence = args.ai_confidence
        ai_confidence_method = "cli_argument"
    else:
        print(f"\n  {cyan('AI Confidence')} {dim('(0.0–1.0, press Enter to skip)')}")
        ai_confidence = prompt_float("AI confidence score", min_val=0.0, max_val=1.0, required=False)
        ai_confidence_method = "momentum_spread" if ai_confidence is not None else None

    print(f"\n  {cyan('Momentum Scores')} {dim('(optional, press Enter to skip)')}")
    raw_scores = prompt("Momentum scores JSON", default="", required=False)
    ai_momentum_scores = None
    if raw_scores:
        try:
            ai_momentum_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print(yellow("  ⚠ Could not parse momentum scores; skipping."))

    ai_selected_assets = list(ai_signal.keys()) if ai_signal else None

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

    human_weights = None
    human_reason = None
    if human_decision == "modify":
        print(f"\n  {cyan('Modified Weights')}")
        print(f"  {dim('Original: ' + json.dumps(ai_signal))}")
        human_weights = prompt_json("  Your modified weights (JSON)")
        human_reason = prompt("  Reason for modification", required=True)
    elif human_decision == "reject":
        human_reason = prompt("\n  Reason for rejection", required=True)

    print(f"\n  {cyan('Market Context')} {dim('(optional)')}")
    regime_raw = prompt("  Market regime (bull/bear/sideways)", default="", required=False)
    market_regime = regime_raw if regime_raw in ("bull", "bear", "sideways") else None

    cycle_id = repo.create_cycle(
        cycle_date=d, strategy=strategy, market_regime=market_regime,
    )
    decision_id = repo.log_decision(
        date=d, strategy=strategy, ai_signal=ai_signal,
        human_decision=human_decision,
        ai_confidence=ai_confidence, ai_confidence_method=ai_confidence_method,
        ai_momentum_scores=ai_momentum_scores, ai_selected_assets=ai_selected_assets,
        human_weights=human_weights, human_reason=human_reason,
        cycle_id=cycle_id,
    )

    print(f"\n{separator()}")
    print(f"  {green('✓ Decision logged.')}  ID: {bold(str(decision_id))}")
    print(f"  Date: {d}  |  Strategy: {strategy}")
    print(f"  Human decision: {human_decision_color(human_decision)}")
    if ai_confidence is not None:
        print(f"  AI confidence: {pct(ai_confidence)}")
    print(f"\n  {dim('Next steps:')}")
    if human_decision != "reject":
        print(f"  {cyan(f'python decisions/cli.py exec --id {decision_id} --symbol <TICKER> --side buy --qty <N> --price <P>')}")
    print(f"  {cyan(f'python decisions/cli.py outcome --id {decision_id} --return <R> --benchmark <B>')}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Display decision history in a formatted table.

    Args:
        args: Parsed arguments (limit, filter, strategy, offset).
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

    print(f"\n  {'ID':<5} {'Date':<12} {'Strategy':<20} {'AI Signal':<24} "
          f"{'Human':<10} {'Conf':<7} {'30d Ret':<10} {'AI':<14}")
    print(f"  {separator('─', 68)}")

    for r in decisions:
        ai_sig = r.get("ai_signal") or {}
        sig_str = " ".join(f"{k}:{v*100:.0f}%" for k, v in ai_sig.items())[:22]
        actual_ret = r.get("actual_return_30d")
        ret_str = pct(actual_ret, signed=True) if actual_ret is not None else dim("pending")
        conf = r.get("ai_confidence")
        conf_str = pct(conf, 0) if conf is not None else dim("—")

        print(
            f"  {str(r['id']):<5} {str(r['date']):<12} {str(r['strategy'])[:18]:<20} "
            f"{sig_str:<24} {human_decision_color(r['human_decision']):<20} "
            f"{conf_str:<7} {ret_str:<10} {ai_correct_color(r.get('ai_correct'))}"
        )

    print(f"\n  {dim(f'Total shown: {len(decisions)}')}")
    if len(decisions) == args.limit:
        print(f"  {dim(f'--offset {args.offset + args.limit} to see more')}")
    print()


# ---------------------------------------------------------------------------
# Command: outcome
# ---------------------------------------------------------------------------

def cmd_outcome(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Record the realized return for a past decision.

    Args:
        args: Parsed arguments (id, ret, benchmark, ai_only).
        repo: DecisionRepository instance.
    """
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('RECORD OUTCOME')}")
    print(bold('═' * 72) + "\n")

    if args.id:
        decision_id = args.id
    else:
        pending = repo.get_pending_outcomes()
        if not pending:
            print(f"  {green('✓ No pending outcomes.')}\n")
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
    print(f"  AI signal:      {json.dumps(decision.get('ai_signal', {}))}")
    print(f"  Human decision: {human_decision_color(decision['human_decision'])}")
    if decision.get("human_weights"):
        print(f"  Final weights:  {json.dumps(decision['human_weights'])}")
    print()

    actual_return = args.ret if args.ret is not None else prompt_float(
        "  Realized portfolio return (e.g. 0.034 for +3.4%)", min_val=-1.0, max_val=10.0
    )
    benchmark_return = args.benchmark if args.benchmark is not None else prompt_float(
        "  SPY benchmark return (optional)", required=False
    )
    ai_only_return = None
    if args.ai_only is not None:
        ai_only_return = args.ai_only
    elif decision.get("human_decision") == "modify":
        print(f"\n  {cyan('Counterfactual')} {dim('(AI-only return for value-add calc)')}")
        ai_only_return = prompt_float(
            "  AI-only return (optional)", required=False
        )

    raw_assets = prompt("\n  Per-asset returns JSON (optional, press Enter to skip)", required=False)
    asset_returns = None
    if raw_assets:
        try:
            asset_returns = json.loads(raw_assets)
        except json.JSONDecodeError:
            print(yellow("  ⚠ Could not parse; skipping per-asset returns."))

    notes = prompt("  Notes (optional)", required=False)
    outcome_id = repo.log_outcome(
        decision_id=decision_id, actual_return_30d=actual_return,
        benchmark_return_30d=benchmark_return, asset_returns=asset_returns,
        ai_only_return_30d=ai_only_return, notes=notes or None,
    )

    threshold = 0.005
    if abs(actual_return) < threshold:
        verdict = yellow("~ inconclusive (return < 0.5%)")
    elif actual_return > 0:
        verdict = green("✓ direction correct")
    else:
        verdict = red("✗ direction wrong")

    print(f"\n{separator()}")
    print(f"  {green('✓ Outcome recorded.')}  Outcome ID: {bold(str(outcome_id))}")
    print(f"  Realized return:  {bold(pct(actual_return, signed=True))}")
    if benchmark_return is not None:
        print(f"  Benchmark (SPY):  {pct(benchmark_return, signed=True)}  "
              f"| Excess: {pct(actual_return - benchmark_return, signed=True)}")
    if ai_only_return is not None:
        va = actual_return - ai_only_return
        va_str = green(pct(va, signed=True)) if va > 0 else red(pct(va, signed=True))
        print(f"  AI-only return:   {pct(ai_only_return, signed=True)}  | Human value-add: {va_str}")
    print(f"  AI verdict:       {verdict}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: pending
# ---------------------------------------------------------------------------

def cmd_pending(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """List decisions old enough for outcome recording but not yet updated."""
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

    print(f"\n  {dim('Record outcomes:')}")
    print(f"  {cyan('python decisions/cli.py outcome --id <ID> --return <value>')}\n")


# ---------------------------------------------------------------------------
# Command: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """Print comprehensive analytics report."""
    records = repo.get_all_outcomes()
    analyzer = DecisionAnalyzer(records)
    report = analyzer.full_report()

    adoption = report["adoption_rates"]
    accuracy = report["ai_accuracy"]
    hva = report["human_value_add"]
    calib = report["confidence_calibration"]
    per_dec = report["per_decision_value_add"]

    gen_at = report["generated_at"][:19]
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('DECISION JOURNAL — ANALYTICS REPORT')}")
    print(f"  {dim(f'Generated: {gen_at}')}")
    print(bold('═' * 72))

    print(f"\n  {bold('[ HUMAN DECISION BREAKDOWN ]')}")
    print(f"  Total decisions:    {bold(str(adoption['total_decisions']))}")
    if adoption["total_decisions"] > 0:
        print(f"  Approved (adopt):   {green(str(adoption['approved']))}  {dim(pct(adoption['adoption_rate']))}")
        print(f"  Modified:           {yellow(str(adoption['modified']))}  {dim(pct(adoption['modification_rate']))}")
        print(f"  Rejected:           {red(str(adoption['rejected']))}  {dim(pct(adoption['rejection_rate']))}")

    print(f"\n  {bold('[ AI ACCURACY ]')}")
    print(f"  Decisions with outcomes:  {accuracy['n_with_outcomes']}")
    print(f"  Conclusive outcomes:      {accuracy['n_conclusive']}")
    if accuracy["ai_accuracy_rate"] is not None:
        acc = accuracy["ai_accuracy_rate"]
        acc_display = (green if acc >= 0.6 else yellow if acc >= 0.5 else red)(pct(acc))
        print(f"  {bold('AI Accuracy Rate:')}        {acc_display}")
        print(f"    Correct: {green(str(accuracy['n_correct']))} | Mean ret: {pct(accuracy['mean_return_correct'], signed=True)}")
        print(f"    Wrong:   {red(str(accuracy['n_wrong']))} | Mean ret: {pct(accuracy['mean_return_wrong'], signed=True)}")
        print(f"    Inconclusive: {yellow(str(accuracy['n_inconclusive']))}")
    else:
        print(f"  {dim('No conclusive outcomes yet.')}")
    print(f"  Pending outcomes:         {report['pending_outcomes_count']}")

    print(f"\n  {bold('[ HUMAN VALUE-ADD ]')}")
    if hva["n_modify_with_outcomes"] > 0:
        mean_va = hva["mean_human_value_add"]
        va_display = green(pct(mean_va, signed=True)) if mean_va and mean_va > 0 else red(pct(mean_va, signed=True))
        print(f"  Modifications with outcomes: {hva['n_modify_with_outcomes']}")
        print(f"  {bold('Mean value-add per trade:')} {va_display}")
        print(f"  % modifications helpful:   {pct(hva['pct_modifications_helpful'])}")
        print(f"  Cumulative human alpha:    {pct(hva['total_human_alpha'], signed=True)}")
        if per_dec:
            print(f"\n  {dim('Per-decision breakdown:')}")
            for row in per_dec:
                helpful_str = green("↑") if row["helpful"] else red("↓")
                print(
                    f"    {helpful_str} [{row['decision_id']}] {row['date']} | "
                    f"actual: {pct(row['actual_return'], signed=True)} "
                    f"AI-only: {pct(row['ai_return'], signed=True)} "
                    f"value-add: {pct(row['value_add'], signed=True)}"
                )
    else:
        print(f"  {dim('No modification outcomes recorded yet.')}")

    print(f"\n  {bold('[ CONFIDENCE CALIBRATION ]')}")
    if calib["has_confidence_data"]:
        print(f"  Confidence ↔ Return correlation: {bold(ratio(calib['correlation']))}")
        print(f"  {dim(calib['interpretation'])}\n")
        print(f"  {'Tercile':<24} {'N':<6} {'Accuracy':<12} {'Mean Return'}")
        print(f"  {separator('─', 55)}")
        for t in calib["tercile_analysis"]:
            acc_str = pct(t["accuracy"]) if t["accuracy"] is not None else dim("N/A")
            print(f"  {t['label']:<24} {t['n']:<6} {acc_str:<12} {pct(t['mean_return'], signed=True)}")
    else:
        print(f"  {dim(calib['interpretation'])}")

    by_strat = report["by_strategy"]
    if len(by_strat) > 1:
        print(f"\n  {bold('[ BY STRATEGY ]')}")
        for strat, data in by_strat.items():
            acc_rate = data["ai_accuracy"]["ai_accuracy_rate"]
            adopt_rate = data["adoption_rates"]["adoption_rate"]
            print(f"  {bold(strat)}: {data['n_decisions']} decisions | "
                  f"Adopt: {pct(adopt_rate)} | Accuracy: {pct(acc_rate)}")

    print(f"\n{separator()}\n")


# ---------------------------------------------------------------------------
# Command: exec  (NEW)
# ---------------------------------------------------------------------------

def cmd_exec(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Record a broker fill for an approved or modified decision.

    Supports both interactive and argument-driven modes. Validates that the
    target decision exists and was not rejected before writing.

    Args:
        args: Parsed arguments (id, symbol, side, qty, price, commission,
              commission_type, broker, order_id, time).
        repo: DecisionRepository instance.
    """
    print(f"\n{bold('═' * 72)}")
    print(f"  {bold('LOG EXECUTION FILL')}")
    print(bold('═' * 72) + "\n")

    # Get decision
    decision_id = args.id or int(prompt("Decision ID"))
    decision = repo.get_decision(decision_id)
    if decision is None:
        print(red(f"  ✗ Decision id={decision_id} not found."))
        sys.exit(1)
    if decision["human_decision"] == "reject":
        print(red(f"  ✗ Decision #{decision_id} was rejected — cannot log fills."))
        sys.exit(1)

    effective_weights = decision.get("human_weights") or decision.get("ai_signal") or {}
    print(f"  Decision #{decision_id}: {decision['date']} | {decision['strategy']}")
    print(f"  Human decision: {human_decision_color(decision['human_decision'])}")
    print(f"  Effective weights: {json.dumps(effective_weights)}\n")

    # Symbol
    symbol = (args.symbol or prompt("  Ticker symbol (e.g. SPY)")).upper().strip()

    # Side
    if args.side:
        side = args.side.lower()
        if side not in ("buy", "sell", "sell_short"):
            print(red("  ✗ --side must be buy, sell, or sell_short."))
            sys.exit(1)
    else:
        side = prompt_choice("  Trade side", choices=["buy", "sell", "sell_short"], default="buy")

    # Quantity
    qty = args.qty if args.qty is not None else prompt_float(
        "  Quantity (shares/units)", min_val=0.0001
    )

    # Price
    price = args.price if args.price is not None else prompt_float(
        "  Fill price per unit ($)", min_val=0.01
    )

    # Commission
    commission = args.commission if args.commission is not None else prompt_float(
        "  Commission ($, press Enter for 0)", default=0.0, min_val=0.0, required=False
    ) or 0.0

    commission_type = args.commission_type or "flat"

    # Broker
    broker = args.broker or prompt("  Broker (e.g. alpaca, ibkr, paper)", default="paper")

    # Order ID
    order_id = args.order_id or prompt("  Broker order ID (optional)", default="", required=False) or None

    # Execution time
    default_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    exec_time = args.time or prompt("  Execution time (ISO)", default=default_time)

    notes = prompt("  Notes (optional)", default="", required=False) or None

    execution_id = repo.add_execution(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        execution_time=exec_time,
        commission=commission,
        commission_type=commission_type,
        broker=broker,
        order_id=order_id,
        notes=notes,
    )

    notional = qty * price
    net = -(notional + commission) if side == "buy" else (notional - commission)

    print(f"\n{separator()}")
    print(f"  {green('✓ Execution logged.')}  ID: {bold(str(execution_id))}")
    print(f"  {side_color(side)} {bold(symbol)} × {qty:.4f} @ ${price:.2f}")
    print(f"  Notional:   ${notional:,.2f}")
    print(f"  Commission: ${commission:.4f} ({commission_type})")
    print(f"  Net amount: {green(f'+${net:,.2f}') if net > 0 else red(f'${net:,.2f}')}")
    print(f"  Broker: {broker}  |  Time: {exec_time[:19]}")
    if order_id:
        print(f"  Order ID: {order_id}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: trace  (NEW)
# ---------------------------------------------------------------------------

def cmd_trace(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Display the complete decision lifecycle as a timeline.

    Shows: decision → executions → outcome in chronological order,
    with a summary of execution statistics.

    Args:
        args: Parsed arguments (id).
        repo: DecisionRepository instance.
    """
    decision_id = args.id or int(prompt("Decision ID to trace"))
    trace = repo.get_full_trace(decision_id)
    if trace is None:
        print(red(f"\n  ✗ Decision id={decision_id} not found.\n"))
        sys.exit(1)

    decision = trace["decision"]
    cycle = trace.get("cycle")
    executions = trace.get("executions", [])
    outcome = trace.get("outcome")
    exec_summary = trace.get("execution_summary", {})

    print(f"\n{bold('═' * 72)}")
    print(f"  {bold(f'FULL DECISION TRACE — #{decision_id}')}")
    print(bold('═' * 72))

    # ── Node 1: Decision ──────────────────────────────────────────────
    print(f"\n  {blue('●')} {bold('DECISION')}  {dim(decision.get('created_at', '')[:19])}")
    print(f"  │")
    print(f"  ├─ Date:      {decision.get('date')}")
    print(f"  ├─ Strategy:  {decision.get('strategy')}")
    if cycle and cycle.get("market_regime"):
        print(f"  ├─ Regime:    {cycle['market_regime']}")

    ai_sig = decision.get("ai_signal") or {}
    sig_str = "  ".join(f"{k}: {v*100:.0f}%" for k, v in ai_sig.items())
    print(f"  ├─ AI signal: {sig_str}")

    conf = decision.get("ai_confidence")
    if conf is not None:
        print(f"  ├─ AI conf:   {pct(conf)}")

    human_dec = decision.get("human_decision", "")
    print(f"  ├─ Human:     {human_decision_color(human_dec)}")
    if decision.get("human_reason"):
        print(f"  │  Reason:   {dim(str(decision['human_reason'])[:60])}")
    if decision.get("human_weights"):
        hw = decision["human_weights"]
        hw_str = "  ".join(f"{k}: {v*100:.0f}%" for k, v in hw.items())
        print(f"  └─ Modified: {hw_str}")
    else:
        print(f"  └─ ({'No modification' if human_dec == 'approve' else 'Rejected'})")

    # ── Node 2: Executions ────────────────────────────────────────────
    print(f"\n  {cyan('●')} {bold('EXECUTIONS')}  {dim(f'({len(executions)} fills)')}")
    print(f"  │")
    if executions:
        for i, ex in enumerate(executions):
            is_last = i == len(executions) - 1
            branch = "└─" if is_last else "├─"
            net = ex.get("net_amount")
            net_str = (green(f"+${net:,.2f}") if net and net > 0 else red(f"${net:,.2f}")) if net else dim("N/A")
            print(
                f"  {branch} [{ex['id']}] {side_color(ex['side'])} "
                f"{bold(ex['symbol'])} × {ex['quantity']:.4f} @ ${ex['price']:.2f} "
                f"→ net {net_str}  {dim(ex.get('broker',''))} {dim(str(ex.get('execution_time',''))[:16])}"
            )
        print(f"  │")
        print(f"  ├─ Total notional: ${exec_summary.get('total_notional', 0):,.2f}")
        print(f"  └─ Total commission: ${exec_summary.get('total_commission', 0):.4f}")
    else:
        print(f"  └─ {dim('No fills recorded yet.')}")
        print(f"     {dim(f'python decisions/cli.py exec --id {decision_id} ...')}")

    # ── Node 3: Outcome ───────────────────────────────────────────────
    print(f"\n  {green('●') if outcome else dim('○')} {bold('OUTCOME')}  "
          f"{dim(outcome.get('outcome_date', 'pending') if outcome else 'not recorded')}")
    print(f"  │")
    if outcome:
        actual = outcome.get("actual_return_30d")
        benchmark = outcome.get("benchmark_return_30d")
        ai_only = outcome.get("ai_only_return_30d")
        ai_correct = outcome.get("ai_correct")
        hva = outcome.get("human_value_add")

        ret_color = green if actual and actual > 0 else red
        print(f"  ├─ Realized return (30d):  {ret_color(pct(actual, signed=True))}")
        if benchmark is not None:
            bm_color = green if benchmark > 0 else red
            excess = (actual or 0) - benchmark
            exc_str = green(pct(excess, signed=True)) if excess > 0 else red(pct(excess, signed=True))
            print(f"  ├─ SPY benchmark (30d):    {bm_color(pct(benchmark, signed=True))} (excess: {exc_str})")
        print(f"  ├─ AI verdict:             {ai_correct_color(ai_correct)}")
        if ai_only is not None:
            print(f"  ├─ AI-only return (30d):   {pct(ai_only, signed=True)}")
        if hva is not None:
            hva_str = green(pct(hva, signed=True)) if hva > 0 else red(pct(hva, signed=True))
            print(f"  └─ Human value-add:        {hva_str}")
        else:
            print(f"  └─ Human value-add:        {dim('N/A (no counterfactual recorded)')}")
    else:
        print(f"  └─ {dim('Pending. Record with:')}")
        print(f"     {cyan(f'python decisions/cli.py outcome --id {decision_id} --return <R>')}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{separator()}")
    print(f"  {dim('Generate full review report:')}")
    print(f"  {cyan(f'python decisions/cli.py review --id {decision_id}')}")
    print(separator() + "\n")


# ---------------------------------------------------------------------------
# Command: review  (NEW)
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Generate a Markdown post-decision review report.

    Supports rule-based (default) and LLM-enhanced (--llm flag) modes.
    Output can be printed to stdout or saved to a file (--save flag).

    Args:
        args: Parsed arguments (id, llm, save, output_dir).
        repo: DecisionRepository instance.
    """
    decision_id = args.id or int(prompt("Decision ID to review"))

    # Verify decision exists before initializing copilot
    if repo.get_decision(decision_id) is None:
        print(red(f"\n  ✗ Decision id={decision_id} not found.\n"))
        sys.exit(1)

    use_llm = getattr(args, "llm", False)
    if use_llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(yellow(
                "\n  ⚠ ANTHROPIC_API_KEY not set. "
                "Set it with: export ANTHROPIC_API_KEY=sk-...\n"
                "  Falling back to rule-based mode.\n"
            ))
            use_llm = False

    copilot = DecisionReviewCopilot(repo, use_llm=use_llm)

    mode_label = "LLM-enhanced" if use_llm else "rule-based"
    print(f"\n  {cyan(f'Generating {mode_label} review for decision #{decision_id}...')}\n")

    try:
        md = copilot.generate_review(decision_id)
    except ValueError as e:
        print(red(f"\n  ✗ {e}\n"))
        sys.exit(1)

    save = getattr(args, "save", False)
    if save:
        output_dir = getattr(args, "output_dir", "reports") or "reports"
        filepath = copilot.save_review(decision_id, output_dir=output_dir)
        print(f"  {green('✓ Report saved:')} {bold(filepath)}\n")
    else:
        # Print to stdout with a subtle border
        print(separator("═"))
        print(md)
        print(separator("═"))
        print(f"\n  {dim('Save to file with --save')}\n")


# ---------------------------------------------------------------------------
# Command: seed
# ---------------------------------------------------------------------------

def cmd_seed(args: argparse.Namespace, repo: DecisionRepository) -> None:
    """
    Load sample data for demo / testing, including execution fills.

    Args:
        args: Parsed arguments (force).
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
            "executions": [
                {"symbol": "SPY", "side": "buy", "quantity": 12.15, "price": 473.20,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_001a",
                 "execution_time": "2024-02-01T09:31:04"},
                {"symbol": "QQQ", "side": "buy", "quantity": 6.83, "price": 415.30,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_001b",
                 "execution_time": "2024-02-01T09:31:12"},
            ],
            "outcome": {"actual_return_30d": 0.041, "benchmark_return_30d": 0.038,
                        "ai_only_return_30d": 0.041,
                        "asset_returns": {"SPY": 0.038, "QQQ": 0.044}},
        },
        {
            "date": "2024-02-29", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "QQQ": 0.5}, "ai_confidence": 0.76,
            "ai_momentum_scores": {"SPY": 0.071, "QQQ": 0.063, "TLT": -0.018, "GLD": 0.031, "DBC": -0.022},
            "ai_selected_assets": ["SPY", "QQQ"],
            "human_decision": "modify",
            "human_weights": {"SPY": 0.5, "GLD": 0.5},
            "human_reason": "Concerned about tech concentration; adding gold as macro hedge.",
            "market_regime": "bull",
            "executions": [
                {"symbol": "SPY", "side": "buy", "quantity": 11.90, "price": 505.10,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_002a",
                 "execution_time": "2024-03-01T09:31:08"},
                {"symbol": "GLD", "side": "buy", "quantity": 25.20, "price": 189.30,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_002b",
                 "execution_time": "2024-03-01T09:31:15"},
            ],
            "outcome": {"actual_return_30d": 0.028, "benchmark_return_30d": 0.032,
                        "ai_only_return_30d": 0.021,
                        "asset_returns": {"SPY": 0.032, "GLD": 0.024}},
        },
        {
            "date": "2024-03-28", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"QQQ": 0.5, "GLD": 0.5}, "ai_confidence": 0.62,
            "ai_momentum_scores": {"SPY": 0.055, "QQQ": 0.068, "TLT": -0.042, "GLD": 0.057, "DBC": 0.011},
            "ai_selected_assets": ["QQQ", "GLD"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "bull",
            "executions": [
                {"symbol": "QQQ", "side": "buy", "quantity": 6.60, "price": 438.10,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_003a",
                 "execution_time": "2024-04-01T09:31:22"},
                {"symbol": "GLD", "side": "buy", "quantity": 24.55, "price": 203.20,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_003b",
                 "execution_time": "2024-04-01T09:31:30"},
            ],
            "outcome": {"actual_return_30d": 0.019, "benchmark_return_30d": 0.024,
                        "ai_only_return_30d": 0.019,
                        "asset_returns": {"QQQ": 0.023, "GLD": 0.015}},
        },
        {
            "date": "2024-04-30", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"TLT": 0.5, "GLD": 0.5}, "ai_confidence": 0.44,
            "ai_momentum_scores": {"SPY": -0.012, "QQQ": -0.021, "TLT": 0.018, "GLD": 0.034, "DBC": -0.007},
            "ai_selected_assets": ["TLT", "GLD"],
            "human_decision": "reject",
            "human_reason": "Low confidence (44%). Fed meeting next week; going to cash.",
            "market_regime": "sideways",
            "executions": [],  # rejected, no fills
            "outcome": None,
        },
        {
            "date": "2024-05-31", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "GLD": 0.5}, "ai_confidence": 0.69,
            "ai_momentum_scores": {"SPY": 0.061, "QQQ": 0.042, "TLT": -0.028, "GLD": 0.073, "DBC": 0.019},
            "ai_selected_assets": ["SPY", "GLD"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "bull",
            "executions": [
                {"symbol": "SPY", "side": "buy", "quantity": 10.75, "price": 528.20,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_005a",
                 "execution_time": "2024-06-03T09:31:15"},
                {"symbol": "GLD", "side": "buy", "quantity": 23.80, "price": 214.10,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_005b",
                 "execution_time": "2024-06-03T09:31:21"},
            ],
            "outcome": {"actual_return_30d": -0.022, "benchmark_return_30d": -0.019,
                        "ai_only_return_30d": -0.022,
                        "asset_returns": {"SPY": -0.018, "GLD": -0.026}},
        },
        {
            "date": "2024-06-28", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "QQQ": 0.5}, "ai_confidence": 0.83,
            "ai_momentum_scores": {"SPY": 0.094, "QQQ": 0.088, "TLT": -0.019, "GLD": 0.041, "DBC": -0.033},
            "ai_selected_assets": ["SPY", "QQQ"],
            "human_decision": "modify",
            "human_weights": {"SPY": 0.7, "QQQ": 0.3},
            "human_reason": "Overweight SPY; QQQ showing mean-reversion signals.",
            "market_regime": "bull",
            "executions": [
                {"symbol": "SPY", "side": "buy", "quantity": 14.90, "price": 547.40,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_006a",
                 "execution_time": "2024-07-01T09:31:09"},
                {"symbol": "QQQ", "side": "buy", "quantity": 4.15, "price": 479.90,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_006b",
                 "execution_time": "2024-07-01T09:31:16"},
            ],
            "outcome": {"actual_return_30d": 0.033, "benchmark_return_30d": 0.028,
                        "ai_only_return_30d": 0.031,
                        "asset_returns": {"SPY": 0.034, "QQQ": 0.029}},
        },
        {
            "date": "2024-07-31", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"SPY": 0.5, "TLT": 0.5}, "ai_confidence": 0.58,
            "ai_momentum_scores": {"SPY": 0.048, "QQQ": 0.031, "TLT": 0.044, "GLD": 0.039, "DBC": -0.011},
            "ai_selected_assets": ["SPY", "TLT"],
            "human_decision": "approve", "human_weights": None, "human_reason": None,
            "market_regime": "sideways",
            "executions": [
                {"symbol": "SPY", "side": "buy", "quantity": 9.50, "price": 543.60,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_007a",
                 "execution_time": "2024-08-01T09:31:18"},
                {"symbol": "TLT", "side": "buy", "quantity": 53.10, "price": 91.70,
                 "commission": 0.50, "broker": "alpaca", "order_id": "alp_007b",
                 "execution_time": "2024-08-01T09:31:25"},
            ],
            "outcome": {"actual_return_30d": 0.003, "benchmark_return_30d": 0.011,
                        "ai_only_return_30d": 0.003,
                        "asset_returns": {"SPY": 0.009, "TLT": -0.003}},
        },
        {
            "date": "2024-08-30", "strategy": "GTAA_126d_Top2",
            "ai_signal": {"GLD": 0.5, "DBC": 0.5}, "ai_confidence": 0.52,
            "ai_momentum_scores": {"SPY": 0.011, "QQQ": -0.008, "TLT": 0.022, "GLD": 0.059, "DBC": 0.048},
            "ai_selected_assets": ["GLD", "DBC"],
            "human_decision": "modify",
            "human_weights": {"GLD": 0.5, "TLT": 0.5},
            "human_reason": "Replacing DBC with TLT; DBC momentum driven by oil spike, prefer bonds.",
            "market_regime": "sideways",
            "executions": [],   # recent, not yet executed
            "outcome": None,
        },
    ]

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

        # Log execution fills
        fills_logged = 0
        for ex in record.get("executions", []):
            repo.add_execution(
                decision_id=decision_id,
                symbol=ex["symbol"],
                side=ex["side"],
                quantity=ex["quantity"],
                price=ex["price"],
                execution_time=ex["execution_time"],
                commission=ex.get("commission", 0.0),
                broker=ex.get("broker", "paper"),
                order_id=ex.get("order_id"),
            )
            fills_logged += 1

        # Log outcome
        outcome_str = ""
        if record.get("outcome") and record["human_decision"] != "reject":
            o = record["outcome"]
            repo.log_outcome(
                decision_id=decision_id,
                actual_return_30d=o["actual_return_30d"],
                benchmark_return_30d=o.get("benchmark_return_30d"),
                asset_returns=o.get("asset_returns"),
                ai_only_return_30d=o.get("ai_only_return_30d"),
            )
            ret = o["actual_return_30d"]
            ret_str = green(pct(ret, signed=True)) if ret > 0 else red(pct(ret, signed=True))
            outcome_str = f" | outcome: {ret_str}"

        fills_str = f" | {dim(f'{fills_logged} fills')}" if fills_logged else ""
        print(f"  {green('✓')} [{decision_id}] {record['date']} | "
              f"{human_decision_color(record['human_decision'])}{fills_str}{outcome_str}")

    print(f"\n  {green('✓ Seed complete.')}")
    print(f"  {dim('Run: python decisions/cli.py analyze')}")
    print(f"  {dim('Run: python decisions/cli.py trace --id 2')}")
    print(f"  {dim('Run: python decisions/cli.py review --id 2')}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python decisions/cli.py",
        description="AI Investment Decision Journal — Columbia MAFN Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # log
    p_log = sub.add_parser("log", help="Record a new AI recommendation and human decision")
    p_log.add_argument("--date", help="Signal date YYYY-MM-DD")
    p_log.add_argument("--strategy", help="Strategy name")
    p_log.add_argument("--ai-signal", dest="ai_signal", help='JSON weights e.g. \'{"SPY":0.5}\'')
    p_log.add_argument("--ai-confidence", dest="ai_confidence", type=float)
    p_log.add_argument("--human-decision", dest="human_decision",
                       choices=["approve", "modify", "reject"])

    # list
    p_list = sub.add_parser("list", help="View decision history")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.add_argument("--filter", choices=["approve", "modify", "reject"])
    p_list.add_argument("--strategy")

    # outcome
    p_out = sub.add_parser("outcome", help="Record realized return for a past decision")
    p_out.add_argument("--id", type=int)
    p_out.add_argument("--return", dest="ret", type=float, help="Realized return (e.g. 0.034)")
    p_out.add_argument("--benchmark", type=float, help="SPY benchmark return")
    p_out.add_argument("--ai-only", dest="ai_only", type=float, help="AI-only counterfactual return")

    # pending
    p_pend = sub.add_parser("pending", help="List decisions awaiting outcome data")
    p_pend.add_argument("--days", type=int, default=30)

    # analyze
    sub.add_parser("analyze", help="Print full analytics report")

    # exec  (NEW)
    p_exec = sub.add_parser("exec", help="Record a broker fill for a decision")
    p_exec.add_argument("--id", type=int, help="Decision ID")
    p_exec.add_argument("--symbol", help="Ticker symbol (e.g. SPY)")
    p_exec.add_argument("--side", choices=["buy", "sell", "sell_short"], help="Trade direction")
    p_exec.add_argument("--qty", type=float, help="Quantity (shares/units)")
    p_exec.add_argument("--price", type=float, help="Fill price per unit ($)")
    p_exec.add_argument("--commission", type=float, help="Commission in $ (default 0)")
    p_exec.add_argument("--commission-type", dest="commission_type",
                        choices=["flat", "bps"], default="flat")
    p_exec.add_argument("--broker", default="paper", help="Broker identifier")
    p_exec.add_argument("--order-id", dest="order_id", help="Broker order reference")
    p_exec.add_argument("--time", help="Execution time ISO string (default: now)")

    # trace  (NEW)
    p_trace = sub.add_parser("trace", help="Display complete decision lifecycle timeline")
    p_trace.add_argument("--id", type=int, help="Decision ID")

    # review  (NEW)
    p_review = sub.add_parser("review", help="Generate Markdown post-decision review report")
    p_review.add_argument("--id", type=int, help="Decision ID")
    p_review.add_argument("--llm", action="store_true",
                          help="Use LLM for narrative (requires ANTHROPIC_API_KEY)")
    p_review.add_argument("--save", action="store_true", help="Save report to file")
    p_review.add_argument("--output-dir", dest="output_dir", default="reports",
                          help="Directory for saved reports")

    # seed
    p_seed = sub.add_parser("seed", help="Load sample data for demo/testing")
    p_seed.add_argument("--force", action="store_true", help="Add seed data to non-empty DB")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, initialize database, dispatch to command handler."""
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
        "exec":    cmd_exec,
        "trace":   cmd_trace,
        "review":  cmd_review,
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
