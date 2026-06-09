using Random
using CSV
using DataFrames

include("../src/RailroadPD.jl")
using .RailroadPD

const SEED = 42

function main()
    rng = MersenneTwister(SEED)

    println("Running decoupled baseline (exact + SA + SQA per corridor)...")
    result = run_decoupled_baseline(PARTICIPATION, DEMAND_PARAMS, COST, FIXED_COST,
                                    FLEET_CAPACITY, 1000, 1000, rng)

    println("\nPer-corridor results:")
    for corr in CORRIDORS
        r = result.corridor_results[corr]
        firms_in = active_firms_for_corridor(corr)
        n_vars   = length(firms_in) * 4
        println("  $corr ($(length(firms_in)) firms, $n_vars vars, 2^$n_vars configs):")
        println("    Exact welfare:  $(round(r.exact_welfare, digits=4))  ($(r.n_optima) tied optima)")
        println("    SA best:        $(round(r.sa_best,       digits=4))")
        println("    SQA best:       $(round(r.sqa_best,      digits=4))")
        println("    SA mean:        $(round(r.sa_mean,       digits=4))")
        println("    SQA mean:       $(round(r.sqa_mean,      digits=4))")
    end

    println("\nAggregated decoupled welfare:")
    println("  Exact (sum of corridor optima): $(round(result.decoupled_exact_welfare, digits=4))")
    println("  SA best (sum):                  $(round(result.decoupled_sa_best,       digits=4))")
    println("  SQA best (sum):                 $(round(result.decoupled_sqa_best,      digits=4))")
    println("  Fleet violation rate (pre-repair, SA concat): $(round(result.fleet_violation_rate, digits=4))")
    println("  Repaired SA welfare:            $(round(result.repaired_welfare,        digits=4))")

    # Write results
    mkpath("results")
    rows = []
    for corr in CORRIDORS
        r = result.corridor_results[corr]
        push!(rows, (
            corridor       = String(corr),
            n_firms        = length(active_firms_for_corridor(corr)),
            exact_welfare  = r.exact_welfare,
            n_optima       = r.n_optima,
            sa_best        = r.sa_best,
            sa_mean        = r.sa_mean,
            sqa_best       = r.sqa_best,
            sqa_mean       = r.sqa_mean,
        ))
    end
    df = DataFrame(rows)
    CSV.write("results/decoupled.csv", df)
    println("\nWrote results/decoupled.csv")

    summary = DataFrame(
        decoupled_exact_welfare = [result.decoupled_exact_welfare],
        decoupled_sa_best       = [result.decoupled_sa_best],
        decoupled_sqa_best      = [result.decoupled_sqa_best],
        fleet_violation_rate    = [result.fleet_violation_rate],
        repaired_welfare        = [result.repaired_welfare],
    )
    CSV.write("results/decoupled_summary.csv", summary)
    println("Wrote results/decoupled_summary.csv")
end

main()
