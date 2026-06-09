# CLAUDE.md — Multi-Game Railroad Coordination: Classical vs Quantum Annealing

## Project overview

This project extends the network prisoner's dilemma experiment to a multi-game
railroad coordination problem. Six historical Northeast corridor railroad firms
play simultaneous pricing and frequency games on three coupled route segments,
connected by a shared fleet capacity constraint. The welfare function is total
industry profit. The primary research question is whether cross-game coupling —
not raw problem size — drives quantum advantage, and whether the welfare-optimal
solution requires correlated strategy profiles that single-game analysis misses.

This is the second experiment in a research program connecting quantum annealing
to institutional economics (Chandler's visible hand thesis). The theoretical
claim being tested: the welfare gap between classical SA and quantum annealing
is larger for the coupled multi-game problem than for three independent
single-game problems of equivalent total QUBO size.

## Language and environment

- Julia 1.10+
- All dependencies managed via Project.toml and Manifest.toml
- No Python, no Jupyter. Plain .jl scripts only.
- Entry point for each experiment is a standalone script in experiments/

## Directory structure

```
railroad_pd/
├── CLAUDE.md
├── Project.toml
├── src/
│   ├── RailroadPD.jl          # top-level module, exports all public functions
│   ├── market_structure.jl    # firm/corridor participation, strategy spaces
│   ├── demand.jl              # linear demand model and profit function
│   ├── qubo.jl                # QUBO encoding: profit + fleet + one-hot penalties
│   ├── exact_solver.jl        # brute force over all feasible configurations
│   ├── classical_sa.jl        # simulated annealing (reuse from quantum_pd, adapt)
│   ├── sqa.jl                 # simulated quantum annealing (reuse, adapt)
│   ├── exact_quantum.jl       # exact Schrödinger evolution (small enough: 45 vars
│   │                          #   is too large; see note below)
│   └── metrics.jl             # profit gap, logical entropy, cooperation rates,
│                              #   fleet utilization, strategy profile differentiation
├── experiments/
│   ├── run_coupled.jl         # main experiment: coupled 3-corridor problem
│   ├── run_decoupled.jl       # baseline: three independent single-corridor problems
│   └── run_sensitivity.jl     # vary fleet capacity, penalty weights, demand params
├── results/
└── plots/
    └── figures.jl
```

## Market structure

### Firms (6 total, historically motivated)

```julia
const FIRMS = [
    :pennsylvania,    # PRR
    :ny_central,      # NYC
    :erie,            # Erie
    :bando,           # B&O
    :new_haven,       # NYNH&H
    :lackawanna,      # DL&W
]
```

### Corridors (3 total)

```julia
const CORRIDORS = [:ny_phl, :phl_pit, :pit_chi]
```

### Participation matrix

Firm × corridor presence (true = firm operates on corridor):

```julia
const PARTICIPATION = Dict(
    (:pennsylvania, :ny_phl)  => true,
    (:pennsylvania, :phl_pit) => true,
    (:pennsylvania, :pit_chi) => true,
    (:ny_central,  :ny_phl)  => true,
    (:ny_central,  :phl_pit) => true,
    (:ny_central,  :pit_chi) => true,
    (:erie,        :ny_phl)  => true,
    (:erie,        :phl_pit) => true,
    (:erie,        :pit_chi) => true,
    (:bando,       :ny_phl)  => true,
    (:bando,       :phl_pit) => true,
    (:bando,       :pit_chi) => true,
    (:new_haven,   :ny_phl)  => true,
    (:lackawanna,  :ny_phl)  => true,
    (:lackawanna,  :phl_pit) => true,
)
# ny_phl: 6 firms, phl_pit: 5 firms, pit_chi: 4 firms
# Total firm-corridor pairs: 15
```

### Strategy space per firm-corridor pair

Each active firm-corridor pair has exactly 3 binary variables:

1. `b_low[i,k]`  — firm i sets LOW rate on corridor k
2. `b_mid[i,k]`  — firm i sets MID rate on corridor k
3. `b_hi[i,k]`   — firm i sets HIGH rate on corridor k

Rate levels: r_low = 1.0, r_mid = 2.0, r_high = 3.0

4. `f[i,k]`      — firm i runs HIGH frequency on corridor k (0 = low, 1 = high)

Total variables: 15 pairs × 4 variables = 60 binary variables.

Variable indexing: assign a canonical integer index to each binary variable.
Store the mapping as a Dict from (firm, corridor, variable_type) to index.
variable_type ∈ {:b_low, :b_mid, :b_hi, :freq}

### Note on exact quantum simulation

60 binary variables → 2^60 dimensional Hilbert space. Exact quantum simulation
is completely infeasible. Do not implement exact_quantum.jl for this experiment.

The solver comparison is:
  - Exact classical (brute force): feasible only for the decoupled single-corridor
    subproblems (largest has 6 firms × 4 vars = 24 variables → 2^24 ≈ 16M configs,
    borderline feasible with fast implementation)
  - Classical SA: main baseline
  - SQA: main quantum comparison

For the coupled problem (60 variables), the exact solver is also infeasible.
Use the following as ground truth proxies:
  - Best solution found across ALL runs (SA + SQA combined), 1000 runs each
  - This gives a lower bound on the welfare gap rather than the exact gap

Document this limitation explicitly in results.

## Demand and profit model

### Linear demand

For firm i on corridor k with n_k competing firms:

```
D[i,k] = a[k] - b * r[i,k] + d * sum(r[j,k] for j != i, j in corridor k)
            + e * f[i,k] - g * sum(f[j,k] for j != i, j in corridor k)
```

Parameters (fixed for round 1):

```julia
const DEMAND_PARAMS = (
    a  = 20.0,   # base demand per corridor (same for all corridors)
    b  = 5.0,    # own-rate demand sensitivity
    d  = 2.0,    # cross-rate demand sensitivity (competitor rate increase raises own demand)
    e  = 3.0,    # own-frequency demand premium
    g  = 1.0,    # competitor frequency substitution effect
)
```

Demand must be clipped to [0, Inf) — negative demand is not permitted.

### Profit function

```
pi[i,k] = (r[i,k] - COST) * D[i,k] - FIXED_COST * f[i,k]
```

```julia
const COST       = 1.0   # marginal cost (homogeneous across firms and corridors)
const FIXED_COST = 5.0   # fixed cost of high-frequency service
```

Total firm profit:

```
Pi[i] = sum(pi[i,k] for k where PARTICIPATION[i,k])
```

Total industry welfare (objective to maximize):

```
W = sum(Pi[i] for i in FIRMS)
```

### Expanding profit in binary variables

The rate r[i,k] is encoded via one-hot:

```
r[i,k] = r_low * b_low[i,k] + r_mid * b_mid[i,k] + r_hi * b_hi[i,k]
```

The demand D[i,k] is linear in {b_low, b_mid, b_hi, f} variables.
The profit pi[i,k] = (r[i,k] - COST) * D[i,k] - FIXED_COST * f[i,k] is therefore
a degree-3 polynomial in binary variables (rate × demand cross-terms).

Degree-3 terms must be quadratized. Use the standard substitution:
for any product b_x * b_y * b_z, introduce auxiliary variable w = b_x * b_y,
enforced by penalty: QUAD_PENALTY * (b_x * b_y - 2*b_x*w - 2*b_y*w + w + 3*w^2).

In practice, the one-hot constraint guarantees that at most one of {b_low, b_mid,
b_hi} is nonzero, so products like b_low[i,k] * b_mid[i,k] = 0 always.
This simplifies the quadratization considerably — exploit this.

Specifically:
- r[i,k]^2 = r_low^2 * b_low + r_mid^2 * b_mid + r_hi^2 * b_hi  (by one-hot)
- r[i,k] * f[i,k] = r_low * b_low * f + r_mid * b_mid * f + r_hi * b_hi * f
  Each term is a product of two distinct binary variables — directly quadratic.
- r[i,k] * r[j,k] for i != j requires expanding the one-hot products:
  sum over (x,y) in {low,mid,hi}^2 of r_x * r_y * b_x[i,k] * b_y[j,k]
  Each term is a product of two distinct binary variables — directly quadratic.

Document the full expansion in a comment block in qubo.jl. The expansion is
mechanical but must be done carefully. Verify against direct profit computation
on test configurations.

## QUBO construction

The full QUBO objective to minimize is:

```
-W_total(b, f) + FLEET_PENALTY * sum_i penalty_fleet(i) 
               + ONEHOT_PENALTY * sum_{i,k} penalty_onehot(i,k)
```

### Fleet constraint penalty

Each firm can run high-frequency service on at most FLEET_CAPACITY corridors:

```julia
const FLEET_CAPACITY = 2
const FLEET_PENALTY  = 50.0
```

For firm i:

```
penalty_fleet(i) = (sum(f[i,k] for k where PARTICIPATION[i,k]) - FLEET_CAPACITY)^2
```

Expand as a quadratic in f[i,k] variables:
- Diagonal terms: (1 - 2*FLEET_CAPACITY) * f[i,k]  for each active k
- Off-diagonal terms: 2 * f[i,k] * f[i,k']  for each pair k < k'
- Constant: FLEET_CAPACITY^2  (tracked for welfare recovery, not in Q matrix)

Note: the penalty is zero when sum(f) <= FLEET_CAPACITY and positive otherwise.
The quadratic form penalizes any violation but does not distinguish between
sum(f) = FLEET_CAPACITY + 1 and sum(f) = FLEET_CAPACITY + 2.
This is intentional — all violations are penalized, severity increases quadratically.

### One-hot constraint penalty

For each active firm-corridor pair (i,k):

```julia
const ONEHOT_PENALTY = 50.0
```

```
penalty_onehot(i,k) = (b_low[i,k] + b_mid[i,k] + b_hi[i,k] - 1)^2
```

Expand:
- Diagonal terms: -1 * b_x[i,k]  for x in {low, mid, hi}  (linear → diagonal Q)
- Off-diagonal terms: 2 * b_x[i,k] * b_y[i,k]  for x < y in {low, mid, hi}

### QUBO matrix assembly

```julia
"""
    build_qubo(participation, demand_params, cost, fixed_cost,
               fleet_capacity, fleet_penalty, onehot_penalty)
    -> (Q::Matrix{Float64}, offset::Float64, var_index::Dict)

Build the full QUBO matrix for the railroad coordination problem.
Returns:
  Q           : n_vars × n_vars upper-triangular QUBO matrix
  offset      : constant term for welfare recovery
  var_index   : Dict mapping (firm, corridor, var_type) -> integer index
                var_type ∈ [:b_low, :b_mid, :b_hi, :freq]
"""
function build_qubo(...)
```

Verification test (implement in test/runtests.jl):
For any configuration vector x, confirm:
  welfare_from_config(x, Q, offset, var_index) ≈ 
  compute_welfare_direct(x, var_index, participation, demand_params, cost, fixed_cost)
to within 1e-8. Test on at least 100 random feasible configurations.

```julia
"""
    compute_welfare_direct(x, var_index, participation, demand_params, cost, fixed_cost)
    -> Float64

Compute total industry welfare directly from the profit formula,
without using the QUBO matrix. Used for verification only.
Enforces one-hot: if sum(b_low, b_mid, b_hi) != 1 for any active pair, return -Inf.
Enforces fleet: if sum(f[i,k]) > FLEET_CAPACITY for any firm, return -Inf.
"""
function compute_welfare_direct(...)
```

## Feasibility

A configuration is **feasible** if:
1. For every active (i,k): exactly one of {b_low, b_mid, b_hi} = 1
2. For every firm i: sum(f[i,k] over active k) <= FLEET_CAPACITY

Infeasible configurations have large positive QUBO energy due to penalty terms.
The annealer should avoid them, but may visit them transiently.

Track feasibility rate in all ensemble runs: fraction of returned configurations
that are feasible. If feasibility rate < 0.8, increase penalty weights.

## Solvers

### exact_solver.jl

Brute force is infeasible for the full 60-variable coupled problem.
Implement exact solver only for the decoupled single-corridor subproblems.

For corridor :ny_phl (6 firms, 24 variables): 2^24 = 16,777,216 configurations.
Use BitArray enumeration with vectorized welfare computation. Target: < 60 seconds.

For corridors :phl_pit (5 firms, 20 variables) and :pit_chi (4 firms, 16 variables):
straightforward.

```julia
"""
    exact_solve_corridor(corridor, participation, demand_params,
                         cost, fixed_cost, fleet_capacity)
    -> (best_config::Vector{Int}, best_welfare::Float64, n_optima::Int)

Enumerate all 2^(4*n_k) configurations for a single corridor.
Return best feasible configuration, its welfare, and count of tied optima.
Only considers fleet constraint within this corridor (cross-corridor coupling absent).
"""
function exact_solve_corridor(...)
```

### classical_sa.jl

Adapt from quantum_pd project. Key changes:

**Proposal distribution:** mixed proposals —
- With probability 0.5: flip a single frequency variable f[i,k]
- With probability 0.3: rotate rate for a random (i,k): cycle b_low→b_mid→b_hi→b_low
  (this preserves one-hot feasibility without penalty sampling overhead)
- With probability 0.2: swap rate+frequency jointly for a random (i,k)

The rate rotation proposal maintains one-hot feasibility automatically. This is
more efficient than independently flipping rate bits and relying on penalties.

**Cooling schedule:** geometric, T0 = 10.0, alpha = 0.997, max_steps = 50_000.
T0 is higher than the single-game experiment because the profit scale is larger.
Set T0 to approximately the standard deviation of welfare across random feasible
configurations — compute this empirically before the main run.

**Initialization:** start from a random feasible configuration (satisfying both
one-hot and fleet constraints). Implement:

```julia
"""
    random_feasible_config(var_index, participation, fleet_capacity, rng)
    -> Vector{Int}

Generate a uniformly random feasible configuration.
For each active (i,k): choose one of {b_low, b_mid, b_hi} uniformly at random.
For each firm i: choose min(n_active_corridors, fleet_capacity) corridors uniformly
at random to set f=1; set f=0 for remaining corridors.
"""
function random_feasible_config(...)
```

### sqa.jl

Adapt from quantum_pd project. Key changes:

**Trotter slices:** P = 20 (unchanged)
**Beta:** 15.0 (slightly higher than single-game due to larger energy scale)
**Gamma_0:** 5.0 (higher initial transverse field)
**Steps:** 50_000 (matched to SA)

**Proposal distribution:** same mixed proposals as SA (rate rotation, frequency
flip, joint swap). Apply to a randomly chosen Trotter slice at each step.

**Best solution tracking:** track best welfare and config across ALL Trotter
slices at EVERY accepted step. (This was the bug in round 1 — do not repeat.)

**Feasibility enforcement in SQA:** after each accepted flip, check feasibility.
If the flip creates a one-hot violation (two rate bits set for same pair), undo
the flip. This hard-enforces one-hot rather than relying on penalty energy alone.
Fleet constraint: do NOT hard-enforce (let penalty handle it) — fleet violations
are softer and the system should be able to transiently violate them.

### No exact quantum solver

Do not implement exact_quantum.jl. The Hilbert space is 2^60 — infeasible.

## Decoupled baseline

The decoupled baseline solves three independent single-corridor problems and
concatenates the solutions. This is the counterfactual: what would three separate
industry associations achieve, each optimizing one corridor independently, with
no cross-corridor coordination?

```julia
"""
    run_decoupled_baseline(participation, demand_params, cost, fixed_cost,
                           fleet_capacity, n_sa_runs, n_sqa_runs, rng)
    -> NamedTuple

For each corridor independently:
  1. Build single-corridor QUBO (no fleet coupling across corridors)
  2. Exact solve (feasible for all three corridors)
  3. Run SA ensemble
  4. Run SQA ensemble

The decoupled welfare is the sum of the three corridor optima.
The decoupled fleet constraint is applied post-hoc: if the concatenated solution
violates fleet capacity for any firm, apply greedy repair (drop the
lowest-profit high-frequency service until feasible).

Return: per-corridor welfare, concatenated welfare (pre-repair),
        repaired welfare, fleet violation rate before repair.
"""
function run_decoupled_baseline(...)
```

The welfare gap between the coupled optimum and the decoupled optimum is the
**coordination value of cross-corridor coupling** — the profit that bilateral
corridor-by-corridor negotiation leaves on the table.

## Metrics

```julia
"""
    strategy_differentiation(configs, var_index, participation) -> Float64

Measure how differentiated the strategy profiles are across firms.
For each corridor k, compute the entropy of the rate distribution across firms:
  H^k = -sum_x p_x * log(p_x)  where p_x = fraction of firms choosing rate x
Average H^k across corridors. High value = firms are choosing different rates
(differentiated equilibrium). Low value = firms are choosing the same rate
(coordination or collusion).
"""
function strategy_differentiation(...)

"""
    fleet_utilization(config, var_index, participation, fleet_capacity) -> Float64

Mean fraction of fleet capacity used across firms:
  mean over firms of (sum(f[i,k]) / fleet_capacity)
"""
function fleet_utilization(...)

"""
    corridor_welfare(config, var_index, participation, demand_params, cost, fixed_cost)
    -> Dict{Symbol, Float64}

Return welfare broken down by corridor.
"""
function corridor_welfare(...)

"""
    coupling_value(coupled_welfare, decoupled_welfare) -> Float64

The coordination value of cross-corridor coupling:
  coupled_welfare - decoupled_welfare
Positive values mean cross-corridor coordination adds value.
"""
function coupling_value(...)

"""
    logical_entropy_profile(config, var_index, participation) -> Float64

Compute logical entropy of the rate-choice partition.
For corridor k, the partition assigns firms to blocks based on their rate choice.
h(pi^k) = 1 - sum_x Pr(x)^2  where Pr(x) = fraction of firms choosing rate x.
Average across corridors. This measures coordination indistinction:
  low entropy = firms indistinct in rate choice (all chose same rate)
  high entropy = firms highly differentiated
"""
function logical_entropy_profile(...)
```

## Experiment scripts

### experiments/run_coupled.jl

Main experiment. Coupled 3-corridor problem, all 6 firms, fleet constraint active.

1. Build full QUBO (60 variables)
2. Run SA ensemble: N_SA_RUNS = 1000 runs
3. Run SQA ensemble: N_SQA_RUNS = 1000 runs
4. Best solution = highest welfare across all 2000 runs (ground truth proxy)
5. For each run record: final welfare, feasibility, fleet utilization,
   strategy differentiation, logical entropy, corridor welfare breakdown
6. Write to results/coupled.csv

Also run: decoupled baseline (calls run_decoupled_baseline)
Write to results/decoupled.csv

Key output: coupling_value = coupled_best - decoupled_best

```julia
const N_SA_RUNS  = 1000
const N_SQA_RUNS = 1000
```

### experiments/run_decoupled.jl

Standalone decoupled run. Solve each corridor independently with exact solver
(ground truth) and SA/SQA ensembles. Apply post-hoc fleet repair.

This establishes the exact welfare achievable without cross-corridor coordination.

### experiments/run_sensitivity.jl

Vary the following parameters one at a time, holding others at baseline:

```julia
FLEET_CAPACITIES = [1, 2, 3]          # tightening/loosening fleet constraint
FIXED_COSTS      = [0.0, 5.0, 15.0]   # low/baseline/high frequency cost
D_PARAMS         = [1.0, 2.0, 4.0]    # cross-rate sensitivity (competition intensity)
```

For each parameter value: run coupled experiment with N_SA_RUNS = 200, N_SQA_RUNS = 200.
Track coupling_value and (SA best welfare - SQA best welfare) as functions of parameters.

The key conjecture: coupling_value is increasing in competition intensity (d) and
decreasing in fleet_capacity. When fleet is unconstrained (fleet_capacity >= 3),
the games decouple and coupling_value → 0.

## Testing

```julia
# test/runtests.jl — all must pass before running experiments

# 1. QUBO verification
# For 50 random feasible configs: welfare_from_config ≈ compute_welfare_direct
test_qubo_correctness()

# 2. One-hot feasibility
# random_feasible_config always returns one-hot-feasible config
test_random_feasible_onehot()

# 3. Fleet feasibility
# random_feasible_config always returns fleet-feasible config
test_random_feasible_fleet()

# 4. Profit sanity
# All-high-rate, low-frequency config has lower welfare than
# all-mid-rate, low-frequency config (undercutting incentive)
test_profit_undercutting_incentive()

# 5. Fleet penalty
# Config with firm running 3 high-freq services has strictly lower
# QUBO objective than config with same firm running 2 high-freq services
# (assuming FLEET_CAPACITY = 2 and FLEET_PENALTY = 50)
test_fleet_penalty_active()

# 6. SA feasibility rate
# Over 100 SA runs, feasibility rate > 0.9
test_sa_feasibility_rate()

# 7. SQA best-tracking
# SQA best welfare >= SQA final welfare on every run (no tracking bug)
test_sqa_best_tracking()

# 8. Decoupled >= independent
# Decoupled welfare (summed corridor optima, pre-repair) >= 
# welfare of any single-corridor optimal solution embedded in full config
test_decoupled_lower_bound()
```

## What success looks like

**Primary finding (expected):**
coupling_value > 0 — the cross-corridor coupled solution achieves higher
total industry profit than three independent corridor optimizations.
This is the baseline result: coordination across corridors adds value.

**Secondary finding (the main conjecture):**
SA best welfare < SQA best welfare on the coupled problem, but
SA best welfare ≈ SQA best welfare on each decoupled subproblem.
This would show that cross-corridor coupling creates the rugged landscape
that differentiates classical and quantum annealing — not problem size alone.

**Tertiary finding (sensitivity):**
coupling_value and the SA/SQA welfare gap both increase as fleet_capacity
decreases (tighter constraint = more cross-corridor coupling = harder problem)
and as cross-rate sensitivity d increases (more competitive = stronger
defection incentives = harder coordination).

**Null result (also informative):**
If SA and SQA perform identically on the coupled problem, the conclusion is
that n=6 firms with K=3 corridors is still too easy for classical SA to fail,
and the interesting regime requires either more corridors, more firms, or
richer strategy spaces. Document parameter ranges and identify the threshold.

## Connection to theoretical framework

The coupling_value metric operationalizes the Chandler argument:
- coupling_value = 0: corridor-by-corridor negotiation (classical decentralized
  annealing) achieves the same outcome as joint optimization. Markets work.
- coupling_value > 0: joint optimization achieves more than independent
  corridor optimization. The visible hand adds value.
- SA welfare gap > 0 on coupled but not decoupled: the landscape created by
  cross-corridor coupling is genuinely hard for classical SA. The quantum
  advantage is structural, not just computational.

The logical entropy profile tracks Ellerman's partition structure:
low entropy = firms indistinguishable in rate choice (coordination / collusion)
high entropy = firms differentiated (specialization, comparative advantage)
The welfare-optimal solution may have high logical entropy if comparative
advantage across corridors requires differentiated strategies.