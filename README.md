# RailroadPD: Multi-Game Railroad Coordination via Classical and Quantum Annealing

A computational experiment comparing simulated annealing (SA) and simulated quantum annealing (SQA) on a multi-game prisoner's dilemma modeled after the historical Northeast Corridor railroad industry. The central question: does **cross-game coupling** — not raw problem size — drive quantum advantage?

This is the second experiment in a research program connecting quantum annealing to institutional economics (Chandler's *visible hand* thesis).

---

## Research Question

Six historical Northeast Corridor railroad firms simultaneously choose pricing and service frequency on three coupled route segments, subject to a shared fleet capacity constraint. The welfare objective is total industry profit.

**Primary conjecture:** the welfare gap between classical SA and quantum annealing is larger for the *coupled* multi-corridor problem than for three *independent* single-corridor problems of equivalent total QUBO size. Cross-corridor coupling creates a rugged energy landscape that classical annealing cannot escape; quantum tunneling can.

**Chandler connection:** `coupling_value = coupled_welfare − decoupled_welfare` operationalizes the visible hand argument. A positive coupling value means joint cross-corridor optimization (the visible hand) outperforms corridor-by-corridor negotiation (the market).

---

## Market Structure

### Firms

| Symbol | Historical Railroad |
|--------|-------------------|
| `:pennsylvania` | Pennsylvania Railroad (PRR) |
| `:ny_central` | New York Central (NYC) |
| `:erie` | Erie Railroad |
| `:bando` | Baltimore & Ohio (B&O) |
| `:new_haven` | New York, New Haven & Hartford (NYNH&H) |
| `:lackawanna` | Delaware, Lackawanna & Western (DL&W) |

### Corridors

| Symbol | Route | Active Firms |
|--------|-------|-------------|
| `:ny_phl` | New York — Philadelphia | 6 |
| `:phl_pit` | Philadelphia — Pittsburgh | 5 |
| `:pit_chi` | Pittsburgh — Chicago | 4 |

### Strategy Space

Each active firm–corridor pair controls 4 binary variables:
- `b_low`, `b_mid`, `b_hi` — rate choice (one-hot: $1.0, 2.0, 3.0$)
- `freq` — high-frequency service flag

**Total: 15 active pairs × 4 variables = 60 binary variables.**

---

## QUBO Formulation

The problem is encoded as a 60-variable QUBO minimizing `−W + penalties`:

- **Welfare term:** profit is a degree-3 polynomial in binary variables; quadratized using one-hot simplifications (products of rate bits for the same firm–corridor pair are always zero).
- **One-hot penalty** (`ONEHOT_PENALTY = 50.0`): enforces exactly one rate bit per active pair.
- **Fleet penalty** (`FLEET_PENALTY = 50.0`): each firm may operate high-frequency service on at most `FLEET_CAPACITY = 2` corridors.

---

## Solvers

### Exact Solver (`exact_solver.jl`)
Brute-force enumeration, feasible only for single-corridor subproblems:
- `:ny_phl`: $2^{24} \approx 16\text{M}$ configurations
- `:phl_pit`: $2^{20} \approx 1\text{M}$ configurations
- `:pit_chi`: $2^{16} \approx 65\text{K}$ configurations

The full 60-variable coupled problem ($2^{60}$) is computationally infeasible for exact methods. Ground truth for the coupled problem is the best solution found across all SA + SQA runs (1000 each).

### Classical SA (`classical_sa.jl`)
Geometric cooling: $T_0 = 10.0$, $\alpha = 0.997$, 50,000 steps.

Mixed proposal distribution:
- 50% — flip a frequency variable
- 30% — rotate rate (low → mid → hi → low), preserving one-hot feasibility
- 20% — joint rate + frequency swap

### Simulated Quantum Annealing (`sqa.jl`)
Trotter decomposition with $P = 20$ slices, $\beta = 15.0$, $\Gamma_0 = 5.0$, 50,000 steps. Same mixed proposals as SA. Hard-enforces one-hot feasibility after each accepted move; fleet constraint handled via penalty energy.

---

## Directory Structure

```
SimpleRail/
├── Project.toml
├── src/
│   ├── RailroadPD.jl          # module root and exports
│   ├── market_structure.jl    # firms, corridors, participation matrix
│   ├── demand.jl              # linear demand model and profit function
│   ├── qubo.jl                # QUBO matrix assembly
│   ├── exact_solver.jl        # brute-force (single corridors only)
│   ├── classical_sa.jl        # simulated annealing
│   ├── sqa.jl                 # simulated quantum annealing
│   └── metrics.jl             # welfare metrics, entropy, fleet utilization
├── experiments/
│   ├── run_coupled.jl         # main experiment: full 3-corridor coupled problem
│   ├── run_decoupled.jl       # baseline: three independent corridor problems
│   ├── run_sensitivity.jl     # parameter sweeps (fleet capacity, costs, competition)
│   └── quick_run.jl           # lightweight smoke test
├── results/                   # output CSVs (gitignored)
└── plots/
    └── figures.jl             # result visualization
```

---

## Getting Started

**Requirements:** Julia 1.10+

```julia
# From the Julia REPL in the project root
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

**Run the test suite:**

```julia
include("test/runtests.jl")
```

**Run the main experiment:**

```bash
julia --project experiments/run_coupled.jl
```

**Quick smoke test:**

```bash
julia --project experiments/quick_run.jl
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `coupling_value` | Coupled welfare − decoupled welfare; the profit attributable to cross-corridor coordination |
| `strategy_differentiation` | Shannon entropy of rate choices across firms per corridor; high = differentiated strategies |
| `logical_entropy_profile` | Ellerman partition entropy; measures indistinction in rate choices |
| `fleet_utilization` | Mean fraction of fleet capacity used across firms |
| `feasibility_rate` | Fraction of annealer runs returning a feasible configuration; target > 0.9 |

---

## Sensitivity Analysis

`run_sensitivity.jl` sweeps three parameters (200 SA + 200 SQA runs per point):

| Parameter | Values | Conjecture |
|-----------|--------|------------|
| `FLEET_CAPACITY` | 1, 2, 3 | Tighter fleet → stronger coupling → larger SA/SQA gap |
| `FIXED_COST` | 0.0, 5.0, 15.0 | Higher fixed cost discourages frequency → less coupling |
| Cross-rate sensitivity `d` | 1.0, 2.0, 4.0 | More competition → stronger defection incentives → harder coordination |

---

## Expected Results

**Success:** `coupling_value > 0` (cross-corridor coordination adds value) and SQA best welfare > SA best welfare on the coupled problem but not on decoupled subproblems — confirming that coupling creates the hard landscape, not problem size.

**Null result (also informative):** identical SA and SQA performance on the coupled problem would indicate that n=6, K=3 is still within the easy regime for classical SA. The sensitivity analysis identifies the threshold.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `CSV` | 0.10 | Results export |
| `DataFrames` | 1.8 | Tabular results |
| `StatsBase` | 0.34 | Summary statistics |
| `Statistics` | stdlib | Mean, variance |
| `Random` | stdlib | RNG seeding |

---

## Citation / Contact

Part of a research program on quantum annealing and institutional economics.
Contact: jsschuler@oeconomia.ai
