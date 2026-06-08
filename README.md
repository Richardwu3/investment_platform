# FinPilot -- AI-Human Decision Intelligence System for Investment Management
**Code = Truth. LLM = Interface. Human = Final Decision.**

## Problem

Investors increasingly use AI-generated investment advice. But most treat AI as a black box:

- AI recommendations are **never tracked**
- Human overrides are **not measured** 
- Outcomes are **not linked back to decisions**

Without feedback loops, investors cannot answer the most important question: 
**When should I trust AI? When should I intervene?**

## Solution

FinPilot creates a **closed-loop decision system**:

AI Recommendation → Human Decision → Execution → Outcome → Review

Every decision is logged. Every override is measured. Every outcome is tracked.
The system continuously answers:

| Question | How FinPilot Answers |
|-----------|---------------|
| **Is AI directionally correct?** | AI Accuracy vs Benchmark |
| **Do human modifications help or hurt?** | Human Value-Add = Actual - AI only |
| **In which regimes does AI platform best?** | Regime-conditional accuracy |

### North Star Metrics
| Metric | Definition | Target |
|-------|------|------------------|
| **AI Calibration Error** | Confidence vs actual accuracy | <10% |
| **Human Value-Add** | Mean (actual - AI only) on overrides | >0 |
| **Override Rate** | % of decisions modified or rejected | 20-40% |
| **Decision Review Completion** | % of decisions with post-mortem | >80% |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: SIGNAL ENGINE                    │
│  GTAA Momentum (Faber 2007) → 5-Layer Validation            │
│  Universe: SPY, QQQ, TLT, GLD, DBC | Top2 equal weight      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: DECISION INTELLIGENCE                  │
│                                                             │
│  AI Recommendation (weights + confidence + momentum scores) │
│           ↓                                                 │
│  Human Review (approve / modify / reject + reason)          │
│           ↓                                                 │
│  Execution Logging (fills: symbol, qty, price, commission)  │
│           ↓                                                 │
│  Outcome Tracking (30d return vs benchmark vs AI-only)      │
│           ↓                                                 │
│  Review Copilot (Markdown retrospectives)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
investment_platform/
├── strategies/          # Signal Engine
│   ├── base.py
│   └── gtaa.py
├── backtest/            # 5-Layer Validation
│   ├── engine.py
│   └── run_first_backtest.py
├── decisions/           # Decision Intelligence Layer
│   ├── db.py
│   ├── analyzer.py
│   ├── confidence.py
│   ├── cli.py
│   └── review_copilot.py
├── tests/               # 31 unit tests
├── data/                # Price cache
├── reports/             # Backtest outputs
└── requirements.txt
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Phase 1: Signal Engine
python backtest/run_first_backtest.py --fast

# Phase 2: Decision Journal (30 seconds to see value)
# 1. Load demo data (8 decisions with fills + outcomes)
python decisions/cli.py seed

# 2. See a complete decision lifecycle 
python decisions/cli.py trace --id 2

# 3. Generate an AI-powered review report
python decisions/cli.py review --id 2

# 4. Run full analytics
python decisions/cli.py analyze
```

---

## CLI Reference

| Command | Purpose |
|-------|------|
| seed | Load 8 demo decisions | 
| trace --id N | Complete lifecycle | 
| review --id N | Generate Markdown report | 
| analyze | Full analytics report | 
| log| Record a new decision | 
| exec --id N ...| Log a fill| 
| outcome --id N --return X| Record 30d result | 
| list | View history | 


## Sample Output

trace --id 2

═══════════════════════════════════════════════════════════════
  FULL DECISION TRACE — #2
═══════════════════════════════════════════════════════════════

  ● DECISION (2024-02-29)
  ├─ AI: SPY 50% | QQQ 50% (confidence: 76%)
  ├─ Human: MODIFIED → SPY 50% | GLD 50%
  └─ Reason: "Concerned about tech concentration; adding gold"

  ● EXECUTIONS (2 fills)
  ├─ BUY SPY × 11.9 @ $505.10
  └─ BUY GLD × 25.2 @ $189.30

  ● OUTCOME (30d)
  ├─ Actual: +2.8% | Benchmark: +3.2% | Excess: -0.4%
  ├─ AI-only: +2.1% | Human Value-Add: +0.7%
  └─ Verdict: ✓ AI direction correct | Human improved outcome

analyze

[ HUMAN DECISION BREAKDOWN ]
  Total decisions:    8
  Approved:           4 (50%)
  Modified:           3 (38%)
  Rejected:           1 (12%)

[ AI ACCURACY ]
  AI Accuracy Rate:   60%
  Correct calls mean return: +3.2%
  Wrong calls mean return:   -1.8%

[ HUMAN VALUE-ADD ]
  Modifications with outcomes: 3
  Mean value-add: +0.5%
  % modifications helpful: 67%
  Cumulative human alpha: +1.5%

## Why Not Just Use ChatGPT?

| Feature | ChatGPT | FinPilot |
|-------|------|------------------|
| Signal source | "I think" | Backtested momentum |
| Decision record | Lost in chat | SQLite database |
| Outcome tracking | None | 30-day follow-up + benchmark |
| Human value-add | Can't measure | Actual - AI-only |
| Post-mortem | Manual | Automated Markdown |
| Auditability | None | Full traceability |
| LLM role | Market prediction (dangerous) | Narrative only |

---

## Tech Stack

**Signal Engine:** Python, pandas numpy

**Backtest:** 5-layer validation (walk-forward, regime analysis, bootstrap)

**Decision Layer:** SQLite, custom CLI

**LLM integration:** Anthropic Claude API (fallback to rule-based)

**Tests:** 60+ unit tests

---

## GTAA Strategy Details

- **Universe**: SPY (US Equity), QQQ (Tech/Growth), TLT (Long Duration Bond), GLD (Gold), DBC (Commodities)
- **Signal**: 126 trailing trading day momentum = P(t-1) / P(t-1-126) - 1
- **Selection**: Top 2 assets by momentum score
- **Weights**: Equal weight (50/50)
- **Rebalance**: Month-end signal → month+1 day 1 execution
- **Cash Rule**: If selected asset has negative momentum → go to cash (configurable)
- **Costs**: Slippage 10bps + Commission 5bps (one-way, applied on turnover)
- **Reference**: Faber (2007), "A Quantitative Approach to Tactical Asset Allocation"

---

## Author

Yuchuan Wu

