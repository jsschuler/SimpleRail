"""
    strategy_differentiation(config, var_index, participation) -> Float64

Shannon entropy of the rate distribution across firms, averaged over corridors.
High = differentiated strategies; low = coordination/collusion on one rate.
"""
function strategy_differentiation(config, var_index=build_var_index(),
                                   participation=PARTICIPATION)
    entropies = Float64[]
    for corr in CORRIDORS
        firms_in = active_firms_for_corridor(corr, participation)
        isempty(firms_in) && continue
        counts = Dict(:b_low => 0, :b_mid => 0, :b_hi => 0)
        for f in firms_in
            for vt in [:b_low, :b_mid, :b_hi]
                if config[var_index[(f, corr, vt)]] == 1
                    counts[vt] += 1
                end
            end
        end
        n_firms = length(firms_in)
        H = 0.0
        for vt in [:b_low, :b_mid, :b_hi]
            p = counts[vt] / n_firms
            p > 0 && (H -= p * log(p))
        end
        push!(entropies, H)
    end
    isempty(entropies) ? 0.0 : mean(entropies)
end

"""
    fleet_utilization(config, var_index, participation, fleet_capacity) -> Float64

Mean fraction of fleet capacity used across firms.
"""
function fleet_utilization(config, var_index=build_var_index(),
                            participation=PARTICIPATION,
                            fleet_capacity=FLEET_CAPACITY)
    utils = Float64[]
    for f in FIRMS
        active_k = active_corridors_for_firm(f, participation)
        isempty(active_k) && continue
        used = sum(config[var_index[(f, k, :freq)]] for k in active_k)
        push!(utils, used / fleet_capacity)
    end
    isempty(utils) ? 0.0 : mean(utils)
end

"""
    corridor_welfare(config, var_index, participation, demand_params, cost, fixed_cost)
    -> Dict{Symbol, Float64}

Welfare broken down by corridor.
"""
function corridor_welfare(config, var_index=build_var_index(),
                           participation=PARTICIPATION,
                           demand_params=DEMAND_PARAMS,
                           cost=COST,
                           fixed_cost=FIXED_COST)
    # decode rates and freqs
    rates = Dict{Tuple{Symbol,Symbol}, Float64}()
    freqs = Dict{Tuple{Symbol,Symbol}, Float64}()
    for f in FIRMS, k in CORRIDORS
        get(participation, (f, k), false) || continue
        bl = config[var_index[(f, k, :b_low)]]
        bm = config[var_index[(f, k, :b_mid)]]
        bh = config[var_index[(f, k, :b_hi)]]
        fr = config[var_index[(f, k, :freq)]]
        rates[(f, k)] = RATE_LOW * bl + RATE_MID * bm + RATE_HIGH * bh
        freqs[(f, k)] = Float64(fr)
    end

    result = Dict{Symbol, Float64}()
    for corr in CORRIDORS
        w = 0.0
        for f in FIRMS
            get(participation, (f, corr), false) || continue
            w += compute_profit(f, corr, rates, freqs,
                                demand_params, cost, fixed_cost, participation)
        end
        result[corr] = w
    end
    result
end

"""
    coupling_value(coupled_welfare, decoupled_welfare) -> Float64

Coordination value of cross-corridor coupling: coupled - decoupled.
"""
function coupling_value(coupled_welfare, decoupled_welfare)
    coupled_welfare - decoupled_welfare
end

"""
    logical_entropy_profile(config, var_index, participation) -> Float64

Logical entropy (Gini-Simpson index) of rate-choice partition, averaged over corridors.
h(pi^k) = 1 - sum_x Pr(x)^2.
Low = indistinct (all firms chose same rate); high = differentiated.
"""
function logical_entropy_profile(config, var_index=build_var_index(),
                                  participation=PARTICIPATION)
    entropies = Float64[]
    for corr in CORRIDORS
        firms_in = active_firms_for_corridor(corr, participation)
        isempty(firms_in) && continue
        counts = Dict(:b_low => 0, :b_mid => 0, :b_hi => 0)
        for f in firms_in
            for vt in [:b_low, :b_mid, :b_hi]
                if config[var_index[(f, corr, vt)]] == 1
                    counts[vt] += 1
                end
            end
        end
        n_firms = length(firms_in)
        h = 1.0 - sum((counts[vt] / n_firms)^2 for vt in [:b_low, :b_mid, :b_hi])
        push!(entropies, h)
    end
    isempty(entropies) ? 0.0 : mean(entropies)
end

"""
    greedy_fleet_repair!(config, var_index, participation, fleet_capacity,
                         demand_params, cost, fixed_cost)

Post-hoc greedy repair: for any firm exceeding fleet_capacity, drop the
high-frequency corridor with the lowest marginal profit contribution.
Modifies config in place.
"""
function greedy_fleet_repair!(config, var_index=build_var_index(),
                               participation=PARTICIPATION,
                               fleet_capacity=FLEET_CAPACITY,
                               demand_params=DEMAND_PARAMS,
                               cost=COST,
                               fixed_cost=FIXED_COST)
    for f in FIRMS
        active_k = active_corridors_for_firm(f, participation)
        isempty(active_k) && continue

        while sum(config[var_index[(f, k, :freq)]] for k in active_k) > fleet_capacity
            # Find the high-freq corridor with lowest freq-attributable profit
            hi_corridors = [k for k in active_k
                            if config[var_index[(f, k, :freq)]] == 1]
            isempty(hi_corridors) && break

            # Compute marginal cost of switching to low-freq for each
            worst_k  = hi_corridors[1]
            worst_dp = Inf
            for k in hi_corridors
                # Temporarily set to 0
                config[var_index[(f, k, :freq)]] = 0
                w_without = compute_welfare_direct(config, var_index, participation,
                                                   demand_params, cost, fixed_cost,
                                                   fleet_capacity + 99)  # skip fleet check
                config[var_index[(f, k, :freq)]] = 1
                # We want to drop the corridor with lowest welfare loss (least valuable freq)
                w_with = compute_welfare_direct(config, var_index, participation,
                                                demand_params, cost, fixed_cost,
                                                fleet_capacity + 99)
                dp = w_with - w_without
                if dp < worst_dp
                    worst_dp = dp
                    worst_k  = k
                end
            end
            config[var_index[(f, worst_k, :freq)]] = 0
        end
    end
    config
end

"""
    run_decoupled_baseline(participation, demand_params, cost, fixed_cost,
                           fleet_capacity, n_sa_runs, n_sqa_runs, rng)
    -> NamedTuple

For each corridor independently:
  1. Build single-corridor QUBO
  2. Exact solve
  3. Run SA ensemble
  4. Run SQA ensemble

Returns per-corridor welfare, concatenated welfare (pre-repair), repaired welfare,
fleet violation rate before repair.
"""
function run_decoupled_baseline(participation=PARTICIPATION,
                                demand_params=DEMAND_PARAMS,
                                cost=COST,
                                fixed_cost=FIXED_COST,
                                fleet_capacity=FLEET_CAPACITY,
                                n_sa_runs=100,
                                n_sqa_runs=100,
                                rng=Random.default_rng())

    var_index = build_var_index(participation)

    corridor_results = Dict{Symbol, NamedTuple}()

    for corr in CORRIDORS
        # Single-corridor participation sub-dict
        sub_part = Dict(k => v for (k, v) in participation if k[2] == corr)

        # Exact solve
        exact_config, exact_welfare, n_optima =
            exact_solve_corridor(corr, participation, demand_params,
                                 cost, fixed_cost, fleet_capacity)

        # Single-corridor QUBO (only firms in this corridor, only this corridor)
        Q_c, offset_c, _ = build_qubo(sub_part, demand_params, cost, fixed_cost,
                                       fleet_capacity, FLEET_PENALTY, ONEHOT_PENALTY)
        vi_c = build_var_index(sub_part)

        # SA ensemble
        sa_bests   = Float64[]
        sa_configs = Vector{Vector{Int}}()
        for _ in 1:n_sa_runs
            bc, bw, _, _, _ = run_sa(Q_c, offset_c, vi_c, sub_part;
                                     fleet_capacity=fleet_capacity, rng=rng)
            push!(sa_bests, bw)
            push!(sa_configs, bc)
        end

        # SQA ensemble
        sqa_bests   = Float64[]
        sqa_configs = Vector{Vector{Int}}()
        for _ in 1:n_sqa_runs
            bc, bw, _, _, _ = run_sqa(Q_c, offset_c, vi_c, sub_part;
                                      fleet_capacity=fleet_capacity, rng=rng)
            push!(sqa_bests, bw)
            push!(sqa_configs, bc)
        end

        corridor_results[corr] = (
            exact_welfare = exact_welfare,
            exact_config  = exact_config,
            n_optima      = n_optima,
            sa_best       = maximum(sa_bests),
            sa_mean       = mean(sa_bests),
            sqa_best      = maximum(sqa_bests),
            sqa_mean      = mean(sqa_bests),
            sa_best_config  = sa_configs[argmax(sa_bests)],
            sqa_best_config = sqa_configs[argmax(sqa_bests)],
        )
    end

    # Concatenate per-corridor exact solutions into a full config
    full_exact_config = zeros(Int, length(var_index))
    for corr in CORRIDORS
        src = corridor_results[corr].exact_config
        for (key, idx) in var_index
            key[2] == corr || continue
            if idx <= length(src) && src[idx] != 0
                full_exact_config[idx] = src[idx]
            end
        end
    end

    # Fill missing (firms not in a corridor stay 0; but we need one-hot for all active)
    # The exact_config from exact_solve_corridor already writes into full var_index slots
    full_exact_config_corrected = zeros(Int, length(var_index))
    for corr in CORRIDORS
        ex_cfg = corridor_results[corr].exact_config
        for v in 1:length(ex_cfg)
            ex_cfg[v] != 0 && (full_exact_config_corrected[v] = ex_cfg[v])
        end
    end

    decoupled_exact_welfare = sum(corridor_results[c].exact_welfare for c in CORRIDORS)
    decoupled_sa_best       = sum(corridor_results[c].sa_best for c in CORRIDORS)
    decoupled_sqa_best      = sum(corridor_results[c].sqa_best for c in CORRIDORS)

    # Fleet violation check (pre-repair)
    n_violations = 0
    # Build concatenated SA best config
    concat_sa_config = zeros(Int, length(var_index))
    for corr in CORRIDORS
        src = corridor_results[corr].sa_best_config
        for v in 1:length(src)
            src[v] != 0 && (concat_sa_config[v] = src[v])
        end
    end
    for f in FIRMS
        active_k = active_corridors_for_firm(f, participation)
        isempty(active_k) && continue
        freq_sum = sum(concat_sa_config[var_index[(f, k, :freq)]]
                       for k in active_k if haskey(var_index, (f, k, :freq)))
        freq_sum > fleet_capacity && (n_violations += 1)
    end
    fleet_violation_rate = n_violations / length(FIRMS)

    # Apply greedy repair to concat SA best config
    repaired_config = copy(concat_sa_config)
    greedy_fleet_repair!(repaired_config, var_index, participation, fleet_capacity,
                         demand_params, cost, fixed_cost)
    repaired_welfare = compute_welfare_direct(repaired_config, var_index, participation,
                                              demand_params, cost, fixed_cost,
                                              fleet_capacity)

    (
        corridor_results       = corridor_results,
        decoupled_exact_welfare = decoupled_exact_welfare,
        decoupled_sa_best       = decoupled_sa_best,
        decoupled_sqa_best      = decoupled_sqa_best,
        fleet_violation_rate    = fleet_violation_rate,
        repaired_welfare        = repaired_welfare,
        repaired_config         = repaired_config,
    )
end
