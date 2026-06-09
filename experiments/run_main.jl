"""
Main experiment: full SA+SQA ensemble in the fleet-constraint-binding regime.

Two conditions are run:
  A: g=0.5, fixed_cost=1.0  (mild binding, coupling_value ≈ -1)
  B: g=0.0, fixed_cost=2.0  (strong binding, coupling_value ≈ -16)

Primary metric: SA/SQA best-welfare gap on the COUPLED problem vs the
DECOUPLED subproblems.  Decoupled subproblems are easy for both solvers
(both find exact optima); if the gap is larger on the coupled problem,
cross-corridor fleet coupling is the source of landscape hardness.
"""

include("../src/RailroadPD.jl")
using .RailroadPD, Random, Statistics, CSV, DataFrames

const N_COUPLED  = 1000   # SA and SQA runs on the 68-var coupled QUBO
const N_DECOUPLED = 200   # SA and SQA runs per corridor in decoupled baseline
const SEED = 42

# ---------------------------------------------------------------------------
# Parameter conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    (label="A_g05_fc1", g=0.5, fixed_cost=1.0),
    (label="B_g00_fc2", g=0.0, fixed_cost=2.0),
]

# ---------------------------------------------------------------------------
function run_condition(label, g_val, fc_val, rng)

    params = (a=DEMAND_PARAMS.a, b=DEMAND_PARAMS.b,
              d=DEMAND_PARAMS.d, e=DEMAND_PARAMS.e,
              g=g_val)

    println("\n", "="^60)
    println("CONDITION $label  (g=$g_val, fixed_cost=$fc_val)")
    println("="^60)

    # ------------------------------------------------------------------
    # Coupled problem
    # ------------------------------------------------------------------
    println("Building coupled QUBO...")
    Q, offset, var_index = build_qubo(PARTICIPATION, params, COST, fc_val,
                                       FLEET_CAPACITY, FLEET_PENALTY, ONEHOT_PENALTY)
    println("  n_vars = $(length(var_index))")

    sa_rows  = []
    sa_bests = Float64[]
    sa_best_x = nothing; sa_best_w = -Inf
    for run in 1:N_COUPLED
        bx, bw, _, fw, feas = run_sa(Q, offset, var_index; rng=rng)
        push!(sa_bests, bw)
        push!(sa_rows, (solver="SA", run=run, best_welfare=bw, final_welfare=fw,
                        feasible=feas,
                        fleet_util=fleet_utilization(bx, var_index),
                        strat_diff=strategy_differentiation(bx, var_index),
                        log_entropy=logical_entropy_profile(bx, var_index)))
        if bw > sa_best_w; sa_best_w = bw; sa_best_x = bx; end
        run % 200 == 0 &&
            println("  SA $run/$N_COUPLED  best=$(round(sa_best_w,digits=2))")
    end

    sqa_rows  = []
    sqa_bests = Float64[]
    sqa_best_x = nothing; sqa_best_w = -Inf
    for run in 1:N_COUPLED
        bx, bw, _, fw, feas = run_sqa(Q, offset, var_index; rng=rng)
        push!(sqa_bests, bw)
        push!(sqa_rows, (solver="SQA", run=run, best_welfare=bw, final_welfare=fw,
                         feasible=feas,
                         fleet_util=fleet_utilization(bx, var_index),
                         strat_diff=strategy_differentiation(bx, var_index),
                         log_entropy=logical_entropy_profile(bx, var_index)))
        if bw > sqa_best_w; sqa_best_w = bw; sqa_best_x = bx; end
        run % 200 == 0 &&
            println("  SQA $run/$N_COUPLED  best=$(round(sqa_best_w,digits=2))")
    end

    coupled_best = max(sa_best_w, sqa_best_w)

    println("\n--- Coupled results ---")
    println("  SA  best=$(round(sa_best_w,digits=2))  mean=$(round(mean(sa_bests),digits=2))  std=$(round(std(sa_bests),digits=2))  feas=$(round(mean(r.feasible for r in sa_rows)*100,digits=1))%")
    println("  SQA best=$(round(sqa_best_w,digits=2))  mean=$(round(mean(sqa_bests),digits=2))  std=$(round(std(sqa_bests),digits=2))  feas=$(round(mean(r.feasible for r in sqa_rows)*100,digits=1))%")
    println("  SA/SQA best gap : $(round(sqa_best_w - sa_best_w, digits=2))")
    println("  SA/SQA mean gap : $(round(mean(sqa_bests) - mean(sa_bests), digits=2))")

    # Best config details
    best_x = sqa_best_w >= sa_best_w ? sqa_best_x : sa_best_x
    cw = corridor_welfare(best_x, var_index, PARTICIPATION, params, COST, fc_val)
    println("  Best config welfare: $(round(coupled_best,digits=2))")
    println("    ny_phl=$(round(cw[:ny_phl],digits=1))  phl_pit=$(round(cw[:phl_pit],digits=1))  pit_chi=$(round(cw[:pit_chi],digits=1))")
    println("    fleet_util=$(round(fleet_utilization(best_x,var_index)*100,digits=1))%  strat_H=$(round(strategy_differentiation(best_x,var_index),digits=3))  log_ent=$(round(logical_entropy_profile(best_x,var_index),digits=3))")

    # ------------------------------------------------------------------
    # Decoupled baseline
    # ------------------------------------------------------------------
    println("\nRunning decoupled baseline ($N_DECOUPLED SA + $N_DECOUPLED SQA per corridor)...")
    dec = run_decoupled_baseline(PARTICIPATION, params, COST, fc_val,
                                  FLEET_CAPACITY, N_DECOUPLED, N_DECOUPLED,
                                  MersenneTwister(SEED + 1))

    println("\n--- Decoupled results ---")
    for corr in CORRIDORS
        r = dec.corridor_results[corr]
        nf = length(active_firms_for_corridor(corr))
        println("  $corr ($nf firms): exact=$(round(r.exact_welfare,digits=2))  SA=$(round(r.sa_best,digits=2))  SQA=$(round(r.sqa_best,digits=2))")
    end
    println("  Summed exact : $(round(dec.decoupled_exact_welfare,digits=2))")
    dec_sa_gap  = dec.decoupled_sa_best  - dec.decoupled_exact_welfare
    dec_sqa_gap = dec.decoupled_sqa_best - dec.decoupled_exact_welfare
    println("  SA  gap from exact: $(round(dec_sa_gap,  digits=2))")
    println("  SQA gap from exact: $(round(dec_sqa_gap, digits=2))")

    cv = coupling_value(coupled_best, dec.decoupled_exact_welfare)

    # ------------------------------------------------------------------
    # Key comparison
    # ------------------------------------------------------------------
    coupled_sa_gap  = sa_best_w  - coupled_best     # ≤ 0 (SA shortfall)
    coupled_sqa_gap = sqa_best_w - coupled_best     # ≤ 0 (SQA shortfall)
    println("\n--- KEY COMPARISON ---")
    println("  Coupling value (coupled best - decoupled exact): $(round(cv,digits=2))")
    println("  Coupled:    SA shortfall=$(round(coupled_best-sa_best_w,digits=2))  SQA shortfall=$(round(coupled_best-sqa_best_w,digits=2))")
    println("  Decoupled:  SA shortfall=$(round(-dec_sa_gap,digits=2))  SQA shortfall=$(round(-dec_sqa_gap,digits=2))")
    println("  SA/SQA BEST gap — coupled: $(round(sqa_best_w-sa_best_w,digits=2))   decoupled: $(round(dec.decoupled_sqa_best-dec.decoupled_sa_best,digits=2))")
    println("  SA/SQA MEAN gap — coupled: $(round(mean(sqa_bests)-mean(sa_bests),digits=2))")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    mkpath("results")
    df = DataFrame(vcat(sa_rows, sqa_rows))
    df[!, :condition]   .= label
    df[!, :g]           .= g_val
    df[!, :fixed_cost]  .= fc_val
    CSV.write("results/main_$(label).csv", df)

    (label=label, g=g_val, fixed_cost=fc_val,
     coupled_best=coupled_best, decoupled_exact=dec.decoupled_exact_welfare,
     coupling_value=cv,
     sa_best=sa_best_w, sqa_best=sqa_best_w,
     sa_mean=mean(sa_bests), sqa_mean=mean(sqa_bests),
     sa_std=std(sa_bests), sqa_std=std(sqa_bests),
     coupled_best_gap=sqa_best_w-sa_best_w,
     coupled_mean_gap=mean(sqa_bests)-mean(sa_bests),
     decoupled_sa_best=dec.decoupled_sa_best,
     decoupled_sqa_best=dec.decoupled_sqa_best,
     decoupled_best_gap=dec.decoupled_sqa_best-dec.decoupled_sa_best,
     sa_feas=mean(r.feasible for r in sa_rows),
     sqa_feas=mean(r.feasible for r in sqa_rows))
end

# ---------------------------------------------------------------------------
function main()
    rng = MersenneTwister(SEED)
    results = []
    for c in CONDITIONS
        r = run_condition(c.label, c.g, c.fixed_cost, rng)
        push!(results, r)
    end

    println("\n", "="^60)
    println("FINAL SUMMARY")
    println("="^60)
    println("Condition          | cv     | SA best | SQA best | best gap | mean gap | dec gap")
    println("-"^85)
    for r in results
        println("$(rpad(r.label,18)) | $(lpad(round(r.coupling_value,digits=1),6)) | $(lpad(round(r.sa_best,digits=1),7)) | $(lpad(round(r.sqa_best,digits=1),8)) | $(lpad(round(r.coupled_best_gap,digits=1),8)) | $(lpad(round(r.coupled_mean_gap,digits=1),8)) | $(lpad(round(r.decoupled_best_gap,digits=1),7))")
    end

    summary_df = DataFrame(results)
    CSV.write("results/main_summary.csv", summary_df)
    println("\nWrote results/main_summary.csv")
end

main()
