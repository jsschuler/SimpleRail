using Random
using CSV
using DataFrames

include("../src/RailroadPD.jl")
using .RailroadPD

const N_SA_RUNS  = 200
const N_SQA_RUNS = 200
const SEED       = 42

const FLEET_CAPACITIES = [1, 2, 3]
const FIXED_COSTS      = [0.0, 5.0, 15.0]
const D_PARAMS         = [1.0, 2.0, 4.0]

function run_one(fleet_cap, fixed_cost_val, d_val, rng)
    params = (
        a = DEMAND_PARAMS.a,
        b = DEMAND_PARAMS.b,
        d = d_val,
        e = DEMAND_PARAMS.e,
        g = DEMAND_PARAMS.g,
    )

    Q, offset, var_index = build_qubo(PARTICIPATION, params, COST, fixed_cost_val,
                                       fleet_cap, FLEET_PENALTY, ONEHOT_PENALTY)

    sa_bests = Float64[]
    for _ in 1:N_SA_RUNS
        _, bw, _, _, _ = run_sa(Q, offset, var_index; fleet_capacity=fleet_cap, rng=rng)
        push!(sa_bests, bw)
    end

    sqa_bests = Float64[]
    for _ in 1:N_SQA_RUNS
        _, bw, _, _, _ = run_sqa(Q, offset, var_index; fleet_capacity=fleet_cap, rng=rng)
        push!(sqa_bests, bw)
    end

    sa_best  = maximum(sa_bests)
    sqa_best = maximum(sqa_bests)

    # Decoupled baseline for coupling value
    decoupled = run_decoupled_baseline(PARTICIPATION, params, COST, fixed_cost_val,
                                       fleet_cap, 50, 50, rng)
    cv = coupling_value(max(sa_best, sqa_best), decoupled.decoupled_exact_welfare)

    (
        fleet_capacity   = fleet_cap,
        fixed_cost       = fixed_cost_val,
        d_param          = d_val,
        sa_best          = sa_best,
        sqa_best         = sqa_best,
        sa_sqa_gap       = sqa_best - sa_best,
        decoupled_exact  = decoupled.decoupled_exact_welfare,
        coupling_value   = cv,
    )
end

function main()
    rows = []

    println("Sensitivity analysis: varying fleet_capacity, fixed_cost, d_param")
    println("(Each condition: $N_SA_RUNS SA + $N_SQA_RUNS SQA runs)\n")

    # Vary fleet capacity (baseline: FIXED_COST=5.0, d=2.0)
    println("--- Varying fleet_capacity ---")
    for fc in FLEET_CAPACITIES
        rng = MersenneTwister(SEED)
        row = run_one(fc, FIXED_COST, DEMAND_PARAMS.d, rng)
        push!(rows, merge(row, (varied_param="fleet_capacity",)))
        println("  fleet_capacity=$fc: coupling_value=$(round(row.coupling_value,digits=3)), sa_sqa_gap=$(round(row.sa_sqa_gap,digits=3))")
    end

    # Vary fixed cost (baseline: FLEET_CAPACITY=2, d=2.0)
    println("--- Varying fixed_cost ---")
    for fc_val in FIXED_COSTS
        rng = MersenneTwister(SEED)
        row = run_one(FLEET_CAPACITY, fc_val, DEMAND_PARAMS.d, rng)
        push!(rows, merge(row, (varied_param="fixed_cost",)))
        println("  fixed_cost=$fc_val: coupling_value=$(round(row.coupling_value,digits=3)), sa_sqa_gap=$(round(row.sa_sqa_gap,digits=3))")
    end

    # Vary d param (baseline: FLEET_CAPACITY=2, FIXED_COST=5.0)
    println("--- Varying d_param ---")
    for d in D_PARAMS
        rng = MersenneTwister(SEED)
        row = run_one(FLEET_CAPACITY, FIXED_COST, d, rng)
        push!(rows, merge(row, (varied_param="d_param",)))
        println("  d=$d: coupling_value=$(round(row.coupling_value,digits=3)), sa_sqa_gap=$(round(row.sa_sqa_gap,digits=3))")
    end

    mkpath("results")
    df = DataFrame(rows)
    CSV.write("results/sensitivity.csv", df)
    println("\nWrote results/sensitivity.csv")

    # Print conjecture check
    println("\n--- Conjecture check ---")
    fleet_rows = filter(r -> r.varied_param == "fleet_capacity", rows)
    sort!(fleet_rows, by=r -> r.fleet_capacity)
    println("coupling_value as fleet_capacity increases (expect decreasing):")
    for r in fleet_rows
        println("  fleet_capacity=$(r.fleet_capacity): $(round(r.coupling_value, digits=3))")
    end

    d_rows = filter(r -> r.varied_param == "d_param", rows)
    sort!(d_rows, by=r -> r.d_param)
    println("coupling_value as d increases (expect increasing):")
    for r in d_rows
        println("  d=$(r.d_param): $(round(r.coupling_value, digits=3))")
    end
end

main()
