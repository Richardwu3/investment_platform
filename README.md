# FinPilot -- AI-Augmented Investment Decision Platform
**Code = Truth. LLM = Interface. Human = Final Decision.**

## What This Is

FinPilot is an **AI-augmented investment decision platform** that helps individual investors make systematic, traceable allocation decisions.

Unlike typical "AI trading" projects, FinPilot:

- **Does not let LLMs predict markets** (they can't)
- **Records every decision** (AI said X, human changed to Y, outcome was Z)
- **Learns from feedback** (when does AI work? when should humans override?)

## Architecture

```
Signal Engine (GTAA)
↓
AI Explanation
↓
Human Review → Decision Log → Outcome Tracking → Performance Review
↓
Execution (IBKR)
```

### Design Philosophy
| Principle | Implementation |
|-----------|---------------|
| **Code = Truth** | All investment signals come from deterministic, backtested rules |
| **LLM = Interface** | LLM explains, summarizes, researches — never predicts |
| **Human-in-the-Loop** | AI recommends, human approves/modifies/rejects |
| **Decision Traceability** | Every recommendation and override is logged |
| **Feedback Learning** | Outcomes tracked to improve future recommendations |

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
│   └── cli.py
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

# Phase 2: Decision Journal
python decisions/cli.py seed      # Load demo data
python decisions/cli.py list      # View history
python decisions/cli.py analyze   # AI accuracy report
python decisions/cli.py log       # Record a decision
```

---

## Phase 1: Singal Engine(GTAA)
Generates monthly asset allocation signals based on 6-month momentum.

## The 5-Layer Validation Framework

| Layer | Name | Question Answered |
|-------|------|------------------|
| 1 | Standard Backtest | Does the strategy have edge over the full sample?(Sharpe ratio) |
| 2 | Walk-Forward | Does performance generalize out-of-sample? |
| 3 | Parameter Stability | Is performance robust to parameter variation?(heatmap) |
| 4 | Market Regime | In which environments does the strategy add value? |
| 5 | Block Bootstrap | Is performance statistically significant vs. luck? |

---

## Phase 2: Decision Intelligence Layer(Core Differentiator)

**Logs every AI recommendation**(what, when, confidence)

**Records human decisions**(approve / modify / reject + reason)

**Tracks outcomes**(30-day realized return)

**Measures AI accuracy**by confidence level

**Quantifies human value-add**(did modifications help?)

## CLI Commands

| Command | Purpose |
|-------|------|
| seed | Load 8 demo decisions | 
| list | View decision history | 
| log | Record AI + human decision | 
| outcome | Fill realized return | 
| analyze | Generate analytics report |

## Sample Output

═══ DECISION JOURNAL — ANALYTICS REPORT ═══

[ HUMAN DECISION BREAKDOWN ]
  Total decisions:    8
  Approved:           4 (50%)
  Modified:           3 (37.5%)
  Rejected:           1 (12.5%)

[ AI ACCURACY ]
  AI Accuracy Rate:   60%
  Correct calls:      3 | Mean return: +3.2%
  Wrong calls:        2 | Mean return: -1.8%

[ HUMAN VALUE-ADD ]
  Mean value-add:     +0.5%
  % modifications helpful: 67%

[ CONFIDENCE CALIBRATION ]
  High confidence (>0.7): 75% accuracy
  Low confidence (<0.5):  33% accuracy

## Why Not Just Use ChatGPT?

| Feature | ChatGPT | FinPilot |
|-------|------|------------------|
| Signal source | "I think" | Backtested rules |
| Decision record | None | Full traceability |
| Outcome tracking | None | 30-day follow-up |
| Auditability | None | Every step logged |
| Investment philosophy | Black box | Transparent, explainable |

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

