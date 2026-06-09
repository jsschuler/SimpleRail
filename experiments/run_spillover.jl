"""
Spillover model experiment: cross-corridor demand complementarities via multi-leg passengers.

New demand term: D[i,k] += h * avg_freq[k'] for each adjacent corridor k'.
With h > 0, the joint optimizer can coordinate freq allocation across corridors
to capture cross-corridor demand benefits — something the decoupled optimizer cannot.
This is the mechanism needed for coupling_value > 0 (Chandler visible-hand claim).

Sweeps h ∈ {0, 1, 2, 3, 4} to map the coupling_value surface and the SA/SQA gap.
"""

include("../src/RailroadPD.jl")
using .RailroadPD, Random, Statistics, CSV, DataFrames

const N_COUPLED   = 1000
const N_DECOUPLED = 200
const SEED = 99

const H_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0]

function run_h(h_val, rng)
    params = (a=DEMAND_PARAMS.a, b=DEMAND_PARAMS.b, d=DEMAND_PARAMS.d,
              e=DEMAND_PARAMS.e, g=DEMAND_PARAMS.g, h=h_val)

    println("\n  h=$h_val — building QUBO...")
    Q, offset, var_index = build_qubo(PARTICIPATION, params, COST, FIXED_COST,
                                       FLEET_CAPACITY, FLEET_PENALTY, ONEHOT_PENALTY)

    # Coupled SA
    sa_bests = Float64[]; sa_best_x = nothing; sa_best_w = -Inf
    for _ in 1:N_COUPLED
        bx, bw, _, _, _ = run_sa(Q, offset, var_index; rng=rng)
        push!(sa_bests, bw)
        if bw > sa_best_w; sa_best_w = bw; sa_best_x = bx; end
    end

    # Coupled SQA
    sqa_bests = Float64[]; sqa_best_x = nothing; sqa_best_w = -Inf
    for _ in 1:N_COUPLED
        bx, bw, _, _, _ = run_sqa(Q, offset, var_index; rng=rng)
        push!(sqa_bests, bw)
        if bw > sqa_best_w; sqa_best_w = bw; sqa_best_x = bx; end
    end

    coupled_best = max(sa_best_w, sqa_best_w)
    best_x = sqa_best_w >= sa_best_w ? sqa_best_x : sa_best_x

    # Decoupled baseline
    dec = run_decoupled_baseline(PARTICIPATION, params, COST, FIXED_COST,
                                  FLEET_CAPACITY, N_DECOUPLED, N_DECOUPLED,
                                  MersenneTwister(SEED + 1))
    cv = coupling_value(coupled_best, dec.decoupled_exact_welfare)

    cw = corridor_welfare(best_x, var_index, PARTICIPATION, params, COST, FIXED_COST)

    println("    h=$(h_val)  cv=$(round(cv,digits=1))  coupled=$(round(coupled_best,digits=1))  decp_exact=$(round(dec.decoupled_exact_welfare,digits=1))")
    println("    SA best=$(round(sa_best_w,digits=1)) mean=$(round(mean(sa_bests),digits=1)) std=$(round(std(sa_bests),digits=1))")
    println("    SQA best=$(round(sqa_best_w,digits=1)) mean=$(round(mean(sqa_bests),digits=1)) std=$(round(std(sqa_bests),digits=1))")
    println("    SA/SQA best gap=$(round(sqa_best_w-sa_best_w,digits=1))  mean gap=$(round(mean(sqa_bests)-mean(sa_bests),digits=1))")
    println("    fleet_util=$(round(fleet_utilization(best_x,var_index)*100,digits=1))%  strat_H=$(round(strategy_differentiation(best_x,var_index),digits=3))  log_ent=$(round(logical_entropy_profile(best_x,var_index),digits=3))")
    println("    corridor welfare: ny_phl=$(round(cw[:ny_phl],digits=1))  phl_pit=$(round(cw[:phl_pit],digits=1))  pit_chi=$(round(cw[:pit_chi],digits=1))")

    # Per-corridor decoupled results
    for corr in CORRIDORS
        r = dec.corridor_results[corr]
        println("    decoupled $corr: exact=$(round(r.exact_welfare,digits=1))  SA=$(round(r.sa_best,digits=1))  SQA=$(round(r.sqa_best,digits=1))")
    end

    (h=h_val, coupling_value=cv,
     coupled_best=coupled_best, decoupled_exact=dec.decoupled_exact_welfare,
     sa_best=sa_best_w, sqa_best=sqa_best_w,
     sa_mean=mean(sa_bests), sqa_mean=mean(sqa_bests),
     sa_std=std(sa_bests), sqa_std=std(sqa_bests),
     best_gap=sqa_best_w-sa_best_w,
     mean_gap=mean(sqa_bests)-mean(sa_bests),
     sa_shortfall=coupled_best-sa_best_w,
     sqa_shortfall=coupled_best-sqa_best_w,
     fleet_util=fleet_utilization(best_x,var_index),
     strat_diff=strategy_differentiation(best_x,var_index),
     log_ent=logical_entropy_profile(best_x,var_index),
     decp_sa_best=dec.decoupled_sa_best,
     decp_sqa_best=dec.decoupled_sqa_best)
end

function main()
    println("="^60)
    println("SPILLOVER MODEL: h sweep")
    println("="^60)
    println("$N_COUPLED SA + $N_COUPLED SQA coupled, $N_DECOUPLED+$N_DECOUPLED decoupled per corridor")

    rng = MersenneTwister(SEED)
    rows = []
    for h in H_VALUES
        push!(rows, run_h(h, rng))
    end

    println("\n", "="^60)
    println("SUMMARY TABLE")
    println("="^60)
    println("h    | cv      | coupled | decp_ex | SA best | SQA best | best_gap | mean_gap | fleet%")
    println("-"^90)
    for r in rows
        println("$(lpad(r.h,4)) | $(lpad(round(r.coupling_value,digits=1),7)) | $(lpad(round(r.coupled_best,digits=1),7)) | $(lpad(round(r.decoupled_exact,digits=1),7)) | $(lpad(round(r.sa_best,digits=1),7)) | $(lpad(round(r.sqa_best,digits=1),8)) | $(lpad(round(r.best_gap,digits=1),8)) | $(lpad(round(r.mean_gap,digits=1),8)) | $(lpad(round(r.fleet_util*100,digits=1),6))%")
    end

    mkpath("results")
    df = DataFrame(rows)
    CSV.write("results/spillover_sweep.csv", df)
    println("\nWrote results/spillover_sweep.csv")
end

main()
