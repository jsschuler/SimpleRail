using Test
using Random

include("../src/RailroadPD.jl")
using .RailroadPD

function test_qubo_correctness()
    Q, offset, var_index = build_qubo()
    rng = MersenneTwister(42)
    n_pass = 0
    for _ in 1:50
        x = random_feasible_config(var_index, PARTICIPATION, FLEET_CAPACITY, rng)
        w_qubo   = welfare_from_config(x, Q, offset)
        w_direct = compute_welfare_direct(x, var_index)
        if abs(w_qubo - w_direct) < 1e-8
            n_pass += 1
        else
            @warn "QUBO mismatch" w_qubo w_direct diff=abs(w_qubo-w_direct)
        end
    end
    @test n_pass == 50
end

function test_random_feasible_onehot()
    var_index = build_var_index()
    rng = MersenneTwister(1)
    for _ in 1:200
        x = random_feasible_config(var_index, PARTICIPATION, FLEET_CAPACITY, rng)
        for f in FIRMS, k in CORRIDORS
            get(PARTICIPATION, (f, k), false) || continue
            s = x[var_index[(f,k,:b_low)]] + x[var_index[(f,k,:b_mid)]] + x[var_index[(f,k,:b_hi)]]
            @test s == 1
        end
    end
end

function test_random_feasible_fleet()
    var_index = build_var_index()
    rng = MersenneTwister(2)
    for _ in 1:200
        x = random_feasible_config(var_index, PARTICIPATION, FLEET_CAPACITY, rng)
        for f in FIRMS
            active_k = active_corridors_for_firm(f)
            isempty(active_k) && continue
            freq_sum = sum(x[var_index[(f,k,:freq)]] for k in active_k)
            @test freq_sum <= FLEET_CAPACITY
        end
    end
end

function test_profit_undercutting_incentive()
    # All-high-rate vs all-mid-rate, both low-frequency
    var_index = build_var_index()
    n = length(var_index)
    x_hi  = zeros(Int, n)
    x_mid = zeros(Int, n)
    for f in FIRMS, k in CORRIDORS
        get(PARTICIPATION, (f, k), false) || continue
        x_hi[var_index[(f, k, :b_hi)]]  = 1
        x_mid[var_index[(f, k, :b_mid)]] = 1
        # freq stays 0
    end
    w_hi  = compute_welfare_direct(x_hi,  var_index)
    w_mid = compute_welfare_direct(x_mid, var_index)
    # Under the parameter values, mid rate should generate higher welfare (undercutting)
    @test w_mid > w_hi
end

function test_fleet_penalty_active()
    # A firm running 3 freq services (violating FLEET_CAPACITY=2) should have
    # higher QUBO energy than the same firm running 2 freq services.
    Q, offset, var_index = build_qubo()
    var_index2 = build_var_index()
    rng = MersenneTwister(3)
    x = random_feasible_config(var_index2, PARTICIPATION, FLEET_CAPACITY, rng)

    # pennsylvania participates in all 3 corridors — force 3 freq services
    firm = :pennsylvania
    x3 = copy(x)
    for k in CORRIDORS
        get(PARTICIPATION, (firm, k), false) || continue
        x3[var_index[(firm, k, :freq)]] = 1
    end

    # force exactly 2 freq services (set pit_chi to 0)
    x2 = copy(x3)
    x2[var_index[(firm, :pit_chi, :freq)]] = 0

    e2 = RailroadPD.qubo_energy(x2, Q)
    e3 = RailroadPD.qubo_energy(x3, Q)

    # Violating config should have higher QUBO energy (worse)
    @test e3 > e2
end

function test_sa_feasibility_rate()
    Q, offset, var_index = build_qubo()
    rng = MersenneTwister(4)
    n_feasible = 0
    n_runs = 100
    for _ in 1:n_runs
        _, _, _, _, feasible = run_sa(Q, offset, var_index; rng=rng)
        feasible && (n_feasible += 1)
    end
    rate = n_feasible / n_runs
    @test rate > 0.9
end

function test_sqa_best_tracking()
    Q, offset, var_index = build_qubo()
    rng = MersenneTwister(5)
    for _ in 1:20
        best_x, best_w, final_x, final_w, _ = run_sqa(Q, offset, var_index; rng=rng)
        @test best_w >= final_w - 1e-10
    end
end

function test_decoupled_lower_bound()
    # Decoupled welfare (sum of per-corridor exact optima) should be >= welfare of
    # a single-corridor optimal solution embedded in the full config.
    result = run_decoupled_baseline(PARTICIPATION, DEMAND_PARAMS, COST, FIXED_COST,
                                    FLEET_CAPACITY, 10, 10, MersenneTwister(6))
    for corr in CORRIDORS
        cw = result.corridor_results[corr].exact_welfare
        @test result.decoupled_exact_welfare >= cw - 1e-10
    end
end

@testset "RailroadPD" begin
    @testset "QUBO correctness"         begin test_qubo_correctness()         end
    @testset "One-hot feasibility"       begin test_random_feasible_onehot()    end
    @testset "Fleet feasibility"         begin test_random_feasible_fleet()     end
    @testset "Undercutting incentive"    begin test_profit_undercutting_incentive() end
    @testset "Fleet penalty active"      begin test_fleet_penalty_active()      end
    @testset "SA feasibility rate"       begin test_sa_feasibility_rate()       end
    @testset "SQA best tracking"         begin test_sqa_best_tracking()         end
    @testset "Decoupled lower bound"     begin test_decoupled_lower_bound()     end
end
