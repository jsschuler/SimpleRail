include("../src/RailroadPD.jl")
using .RailroadPD, Random, Statistics

function quick_run(n_runs=50)
    rng = MersenneTwister(42)

    println("Building QUBO...")
    Q, offset, var_index = build_qubo()
    println("  n_vars = ", length(var_index))

    # -----------------------------------------------------------------------
    # Coupled problem: SA ensemble
    # -----------------------------------------------------------------------
    println("\nRunning SA ($n_runs runs) on coupled problem...")
    sa_bests   = Float64[]
    sa_feas    = Bool[]
    sa_best_x  = nothing
    sa_best_w  = -Inf
    for _ in 1:n_runs
        bx, bw, _, _, feas = run_sa(Q, offset, var_index; rng=rng)
        push!(sa_bests, bw)
        push!(sa_feas, feas)
        if bw > sa_best_w; sa_best_w = bw; sa_best_x = bx; end
    end
    println("  best welfare : ", round(maximum(sa_bests), digits=2))
    println("  mean welfare : ", round(mean(sa_bests), digits=2))
    println("  std welfare  : ", round(std(sa_bests), digits=2))
    println("  feasibility  : ", round(mean(sa_feas)*100, digits=1), "%")

    # -----------------------------------------------------------------------
    # Coupled problem: SQA ensemble
    # -----------------------------------------------------------------------
    println("\nRunning SQA ($n_runs runs) on coupled problem...")
    sqa_bests  = Float64[]
    sqa_feas   = Bool[]
    sqa_best_x = nothing
    sqa_best_w = -Inf
    for _ in 1:n_runs
        bx, bw, _, _, feas = run_sqa(Q, offset, var_index; rng=rng)
        push!(sqa_bests, bw)
        push!(sqa_feas, feas)
        if bw > sqa_best_w; sqa_best_w = bw; sqa_best_x = bx; end
    end
    println("  best welfare : ", round(maximum(sqa_bests), digits=2))
    println("  mean welfare : ", round(mean(sqa_bests), digits=2))
    println("  std welfare  : ", round(std(sqa_bests), digits=2))
    println("  feasibility  : ", round(mean(sqa_feas)*100, digits=1), "%")

    # -----------------------------------------------------------------------
    # Best config across all runs
    # -----------------------------------------------------------------------
    coupled_best_w = max(sa_best_w, sqa_best_w)
    coupled_best_x = sa_best_w >= sqa_best_w ? sa_best_x : sqa_best_x

    println("\n--- Best config (coupled) ---")
    println("  welfare      : ", round(coupled_best_w, digits=2))
    cw = corridor_welfare(coupled_best_x, var_index)
    println("  by corridor  : ny_phl=", round(cw[:ny_phl],  digits=2),
            "  phl_pit=", round(cw[:phl_pit], digits=2),
            "  pit_chi=", round(cw[:pit_chi], digits=2))
    println("  fleet util   : ", round(fleet_utilization(coupled_best_x, var_index)*100, digits=1), "%")
    println("  strategy H   : ", round(strategy_differentiation(coupled_best_x, var_index), digits=3),
            "  (0=all same rate, ln3≈1.10=fully differentiated)")
    println("  logical ent  : ", round(logical_entropy_profile(coupled_best_x, var_index), digits=3),
            "  (0=indistinct, 0.67=max differentiation with 3 rate levels)")

    # Print rate choices in best config
    println("\n  Rate choices in best config:")
    for k in CORRIDORS
        firms_in = active_firms_for_corridor(k)
        choices = String[]
        for f in firms_in
            if coupled_best_x[var_index[(f, k, :b_low)]] == 1
                push!(choices, "$(f)=low")
            elseif coupled_best_x[var_index[(f, k, :b_mid)]] == 1
                push!(choices, "$(f)=mid")
            else
                push!(choices, "$(f)=hi")
            end
        end
        freqs = [f for f in firms_in if coupled_best_x[var_index[(f, k, :freq)]] == 1]
        println("    $k: ", join(choices, ", "))
        println("        hi-freq: ", isempty(freqs) ? "none" : join(freqs, ", "))
    end

    # -----------------------------------------------------------------------
    # Decoupled baseline (exact solve per corridor + SA/SQA)
    # -----------------------------------------------------------------------
    println("\nRunning decoupled baseline (exact + 30 SA + 30 SQA per corridor)...")
    dec = run_decoupled_baseline(PARTICIPATION, DEMAND_PARAMS, COST, FIXED_COST,
                                  FLEET_CAPACITY, 30, 30, MersenneTwister(99))

    println("\n--- Decoupled baseline ---")
    for corr in CORRIDORS
        r  = dec.corridor_results[corr]
        nf = length(active_firms_for_corridor(corr))
        println("  $corr ($nf firms):")
        println("    exact welfare = ", round(r.exact_welfare, digits=2),
                "  (", r.n_optima, " tied optima)")
        println("    SA best = ", round(r.sa_best, digits=2),
                "  SQA best = ", round(r.sqa_best, digits=2))
    end
    println("  Summed exact welfare : ", round(dec.decoupled_exact_welfare, digits=2))
    println("  Summed SA best       : ", round(dec.decoupled_sa_best, digits=2))
    println("  Summed SQA best      : ", round(dec.decoupled_sqa_best, digits=2))
    println("  Fleet violation rate (pre-repair, concat SA): ",
            round(dec.fleet_violation_rate*100, digits=1), "%")
    println("  Repaired welfare     : ", round(dec.repaired_welfare, digits=2))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    cv = coupling_value(coupled_best_w, dec.decoupled_exact_welfare)
    println("\n", "="^55)
    println("SUMMARY")
    println("="^55)
    println("  Coupled best welfare (SA+SQA proxy) : ", round(coupled_best_w, digits=2))
    println("  Decoupled exact welfare (sum)        : ", round(dec.decoupled_exact_welfare, digits=2))
    println("  Coupling value (coordination gain)   : ", round(cv, digits=2))
    println("  SA best  : ", round(maximum(sa_bests),  digits=2))
    println("  SQA best : ", round(maximum(sqa_bests), digits=2))
    println("  SA/SQA gap (best-best)               : ",
            round(maximum(sqa_bests) - maximum(sa_bests), digits=2))
    println("  SA mean  : ", round(mean(sa_bests),  digits=2),
            "  SQA mean : ", round(mean(sqa_bests), digits=2))
    println("  SQA/SA mean advantage                : ",
            round(mean(sqa_bests) - mean(sa_bests), digits=2))
    println("="^55)
    println("\nNOTE: $n_runs runs each; full run uses 1000.")
    println("NOTE: Coupled exact solution is infeasible (2^60). Values are lower bounds.")
end

quick_run(50)
