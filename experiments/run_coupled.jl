using Random
using CSV
using DataFrames

include("../src/RailroadPD.jl")
using .RailroadPD

const N_SA_RUNS  = 1000
const N_SQA_RUNS = 1000
const SEED       = 42

function main()
    rng = MersenneTwister(SEED)

    println("Building QUBO (60 variables)...")
    Q, offset, var_index = build_qubo()
    println("  n_vars = $(length(var_index)), QUBO built.")

    # -----------------------------------------------------------------------
    # SA ensemble
    # -----------------------------------------------------------------------
    println("Running SA ensemble ($N_SA_RUNS runs)...")
    sa_rows = []
    sa_best_welfare = -Inf
    sa_best_config  = nothing

    for run in 1:N_SA_RUNS
        best_x, best_w, _, final_w, feasible =
            run_sa(Q, offset, var_index; rng=rng)

        if best_w > sa_best_welfare
            sa_best_welfare = best_w
            sa_best_config  = best_x
        end

        cw = corridor_welfare(best_x, var_index)
        push!(sa_rows, (
            solver              = "SA",
            run                 = run,
            best_welfare        = best_w,
            final_welfare       = final_w,
            feasible            = feasible,
            fleet_util          = fleet_utilization(best_x, var_index),
            strat_diff          = strategy_differentiation(best_x, var_index),
            log_entropy         = logical_entropy_profile(best_x, var_index),
            welfare_ny_phl      = cw[:ny_phl],
            welfare_phl_pit     = cw[:phl_pit],
            welfare_pit_chi     = cw[:pit_chi],
        ))
        run % 100 == 0 && println("  SA run $run / $N_SA_RUNS, best so far: $(round(sa_best_welfare, digits=2))")
    end

    # -----------------------------------------------------------------------
    # SQA ensemble
    # -----------------------------------------------------------------------
    println("Running SQA ensemble ($N_SQA_RUNS runs)...")
    sqa_rows = []
    sqa_best_welfare = -Inf
    sqa_best_config  = nothing

    for run in 1:N_SQA_RUNS
        best_x, best_w, _, final_w, feasible =
            run_sqa(Q, offset, var_index; rng=rng)

        if best_w > sqa_best_welfare
            sqa_best_welfare = best_w
            sqa_best_config  = best_x
        end

        cw = corridor_welfare(best_x, var_index)
        push!(sqa_rows, (
            solver              = "SQA",
            run                 = run,
            best_welfare        = best_w,
            final_welfare       = final_w,
            feasible            = feasible,
            fleet_util          = fleet_utilization(best_x, var_index),
            strat_diff          = strategy_differentiation(best_x, var_index),
            log_entropy         = logical_entropy_profile(best_x, var_index),
            welfare_ny_phl      = cw[:ny_phl],
            welfare_phl_pit     = cw[:phl_pit],
            welfare_pit_chi     = cw[:pit_chi],
        ))
        run % 100 == 0 && println("  SQA run $run / $N_SQA_RUNS, best so far: $(round(sqa_best_welfare, digits=2))")
    end

    # -----------------------------------------------------------------------
    # Ground truth proxy: best across all 2000 runs
    # -----------------------------------------------------------------------
    coupled_best_welfare = max(sa_best_welfare, sqa_best_welfare)
    println("\nCoupled problem best welfare (proxy): $(round(coupled_best_welfare, digits=4))")
    println("  SA best:  $(round(sa_best_welfare,  digits=4))")
    println("  SQA best: $(round(sqa_best_welfare, digits=4))")
    println("  SA/SQA welfare gap: $(round(sqa_best_welfare - sa_best_welfare, digits=4))")

    # -----------------------------------------------------------------------
    # Decoupled baseline
    # -----------------------------------------------------------------------
    println("\nRunning decoupled baseline...")
    decoupled = run_decoupled_baseline(PARTICIPATION, DEMAND_PARAMS, COST, FIXED_COST,
                                       FLEET_CAPACITY, 200, 200, rng)
    println("  Decoupled exact welfare: $(round(decoupled.decoupled_exact_welfare, digits=4))")
    println("  Decoupled SA best:       $(round(decoupled.decoupled_sa_best,       digits=4))")
    println("  Decoupled SQA best:      $(round(decoupled.decoupled_sqa_best,      digits=4))")
    println("  Fleet violation rate (pre-repair): $(round(decoupled.fleet_violation_rate, digits=4))")
    println("  Repaired decoupled welfare:        $(round(decoupled.repaired_welfare,      digits=4))")

    cv = coupling_value(coupled_best_welfare, decoupled.decoupled_exact_welfare)
    println("\n  >>> Coupling value (coordination gain): $(round(cv, digits=4)) <<<")

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    mkpath("results")
    all_rows = vcat(sa_rows, sqa_rows)
    df = DataFrame(all_rows)
    CSV.write("results/coupled.csv", df)
    println("\nWrote results/coupled.csv")

    # Summary row
    summary = DataFrame(
        coupled_best_welfare    = [coupled_best_welfare],
        sa_best_welfare         = [sa_best_welfare],
        sqa_best_welfare        = [sqa_best_welfare],
        sa_sqa_gap              = [sqa_best_welfare - sa_best_welfare],
        decoupled_exact_welfare = [decoupled.decoupled_exact_welfare],
        decoupled_sa_best       = [decoupled.decoupled_sa_best],
        decoupled_sqa_best      = [decoupled.decoupled_sqa_best],
        coupling_value          = [cv],
        sa_feasibility_rate     = [mean(r.feasible for r in sa_rows)],
        sqa_feasibility_rate    = [mean(r.feasible for r in sqa_rows)],
        fleet_violation_rate    = [decoupled.fleet_violation_rate],
        repaired_welfare        = [decoupled.repaired_welfare],
    )
    CSV.write("results/coupled_summary.csv", summary)
    println("Wrote results/coupled_summary.csv")

    # Warn if feasibility rate is low
    sa_feas  = mean(r.feasible for r in sa_rows)
    sqa_feas = mean(r.feasible for r in sqa_rows)
    sa_feas  < 0.8 && @warn "SA feasibility rate low: $sa_feas — consider increasing penalty weights"
    sqa_feas < 0.8 && @warn "SQA feasibility rate low: $sqa_feas — consider increasing penalty weights"

    # Note on exact solver limitation
    println("""
\nNOTE: The coupled problem has 60 binary variables (2^60 configurations).
Exact brute-force enumeration is infeasible. The welfare reported above is a
lower bound on the true optimum, derived as the best solution found across
$(N_SA_RUNS + N_SQA_RUNS) SA + SQA runs.
""")
end

main()
