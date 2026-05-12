# Optimal Block Sealing in Ethereum's PBS Architecture

**Dylan Tran · Audrey Acken · Colin Calvetti**
CS 6998 — Blockchains

---

## Overview

In Ethereum's Proposer-Builder Separation (PBS) architecture, block builders must decide in real time which transactions to include before sealing a block. This is an **optimal stopping problem**: each arriving transaction must be accepted or rejected immediately, without knowing what comes later.

This project formalizes the sealing problem as a Poisson-deadline knapsack, derives the theoretically optimal threshold policy via the Bellman equation, and proposes Thompson Sampling as a practical adaptive alternative that requires no upfront knowledge of the transaction arrival rate or value distribution. We simulate single-builder and two-builder competition scenarios and measure the efficiency loss from winner-take-all selection.

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd optimal-block-sealing

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac / Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# For the notebook only
pip install jupyter matplotlib
```

**Python 3.10+ required.**

---

## Running the Interactive App

```bash
streamlit run app.py
```

Opens a browser at `http://localhost:8501`. The sidebar lets you:

- Choose a data source (synthetic, upload CSV, or live Flashbots relay data)
- Select which algorithms to compare
- Configure the two-builder race matchup

See [App Walkthrough](#app-walkthrough) below for what each chart means.

---

## Running the Analysis Notebook

```bash
jupyter notebook analysis.ipynb
```

Reproduces all figures used in the report:

| Cell | Figure | Saved to |
|------|--------|----------|
| Competitive ratio sweep | Ratio vs. load λ/μ | `figures/competitive_ratio.png` |
| Variance vs. load | Outcome std dev vs. load λ/μ | `figures/variance_vs_load.png` |
| Two-builder competition | Efficiency loss table by matchup | `figures/two_builder_table.png` |
| Sensitivity analysis | Ratio across value distributions | `figures/sensitivity.png` |

Figures are written to the `figures/` directory (created automatically).

---

## Generating Sample Mempool Data

To test the app's CSV upload feature with realistic synthetic transaction data:

```bash
python generate_sample_data.py
```

Writes `sample_mempool.csv` — 300 transactions with columns `value`, `gas`, `arrival_time`. Values are modeled after Ethereum priority fee tips (heavy-tailed), gas limits follow the distribution of real transaction types (transfers, ERC-20, DeFi).

Upload this file in the app under **Data Source → Upload CSV**.

---

## File Structure

```
optimal-block-sealing/
│
├── simulation.py            # Core algorithms and simulation engine
├── app.py                   # Streamlit interactive app
├── figures.py               # Plotly chart builders (used by app.py)
├── data_sources.py          # Flashbots relay API, CSV parser
├── analysis.ipynb           # Reproduces all report figures
├── generate_sample_data.py  # Generates sample_mempool.csv for CSV upload
└── requirements.txt         # Python dependencies
```

---

## Algorithms

### Offline Knapsack (`offline_knapsack`)
Optimal in hindsight — sees all transactions before deciding. Solved via dynamic programming. Used as the upper bound benchmark only; not achievable in practice.

### Greedy Online (`greedy_unsorted`)
Accept every transaction that fits in the remaining gas capacity, first-come first-served. Naive baseline. Performs poorly under high load because it wastes capacity on low-density transactions early in the slot.

### Optimal Threshold — c\*(b) (`compute_threshold` + `apply_threshold`)
Theoretically optimal online policy derived from the Bellman equation for the Poisson-deadline stopping model. Computes a shadow price c\*(b) for each remaining gas level b — accept a transaction if and only if its value/gas density exceeds c\*(b). Requires knowing the arrival rate λ, slot-end rate μ, and value distribution upfront.

### Thompson Sampling (`thompson_sampling`)
Practical adaptive policy. Maintains a Gamma posterior over transaction density, updated online after each observed transaction. At each step, **samples** a threshold from the posterior (true Thompson exploration) rather than using the posterior mean, adjusted for slot-end urgency via μ/λ. Requires no upfront distributional knowledge — learns λ and the value distribution in real time.

---

## Key Parameters

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Arrival rate | λ | 15 tx/sec | How fast transactions enter the mempool |
| Slot-end rate | μ | 1.0 | Rate of exponential slot deadline (mean slot = 1/μ sec) |
| Block capacity | B | 50 gas units | Total gas the block can hold |
| Load | λ/μ | 15 | Expected transactions per slot; algorithms diverge when load >> B/gas_mean |

The model uses **Poisson arrivals** with an **exponential deadline** (memoryless slot end), which makes the optimal threshold c\*(b) depend only on remaining gas — not on elapsed time. This is the key structural property exploited by the Bellman derivation.

---

## Key Results

**Competitive ratio vs. load** — Greedy and Thompson invert as load increases. At low load (λ/μ ≤ 2), Greedy achieves a perfect 1.0 competitive ratio (accepting everything is optimal when the block never fills) while Thompson's uninformed posterior over-rejects, bottoming at 0.48. Past load ~15, Greedy collapses to 0.54 at load 40 while Thompson improves monotonically to 0.74, nearly matching Threshold (0.77).

**Variance vs. load** — Thompson carries the highest outcome variance of all online policies at high load (std 0.59 vs Greedy's 0.26 at load 40, both normalized to OPT mean). This persistent variance — from posterior sampling rather than deterministic decisions — is the direct cause of Thompson's weak showing in competition.

**Two-builder competition** — Thompson vs Thompson (34.0% efficiency loss) is the worst matchup, worse than Greedy vs Greedy (24.5%), because high variance means both builders can simultaneously misfire on the same slot. Threshold vs Threshold achieves the best outcome at 11.3% loss.

**Distribution sensitivity** — Threshold's edge over Thompson shrinks from 0.27 on Exponential (its calibration distribution) to 0.10 on Pareto. Thompson improves on heavy-tailed distributions while Threshold degrades, closing most of the gap when the distributional assumption is violated.

---

## App Walkthrough

### Single Builder Analysis tab
- **Bar chart** — total block value captured by each algorithm on one slot
- **Line chart** — cumulative value over time; shows *when* each algorithm picks up value
- **Thompson threshold chart** — how the adaptive threshold evolves as TS learns the distribution; the early drop from the prior reflects calibration cost
- **c\*(b) curve** — optimal threshold as a function of remaining gas; non-increasing (be selective when the block is empty, permissive when nearly full)
- **Competitive ratio table** — each algorithm's value as a fraction of the offline optimal

### Two-Builder Race tab
Two builders draw independent transaction streams and run their policies. Winner-take-all: the higher block value wins the slot.

- **Sample trial chart** — one representative race showing both builders' value accumulation
- **Win rate donut** — fraction of trials each builder wins across all simulated slots
- **Value distribution histogram** — spread of final block values; overlap indicates how much outcome is driven by luck vs. policy
- **Efficiency loss callout** — mean value destroyed by competition (only one builder's block is used per slot)
