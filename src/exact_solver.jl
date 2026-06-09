"""
    exact_solve_corridor(corridor, participation, demand_params,
                         cost, fixed_cost, fleet_capacity)
    -> (best_config::Vector{Int}, best_welfare::Float64, n_optima::Int)

Enumerate all 2^(4*n_k) configurations for a single corridor.
Returns the best feasible configuration, its welfare, and count of tied optima.
Fleet constraint is applied within this corridor only (no cross-corridor coupling).
"""
function exact_solve_corridor(corridor,
                               participation=PARTICIPATION,
                               demand_params=DEMAND_PARAMS,
                               cost=COST,
                               fixed_cost=FIXED_COST,
                               fleet_capacity=FLEET_CAPACITY)

    firms_in = active_firms_for_corridor(corridor, participation)
    n_firms  = length(firms_in)
    n_vars   = n_firms * 4   # b_low, b_mid, b_hi, freq per firm

    # Build a local var index for this single corridor
    local_idx = Dict{Tuple{Symbol,Symbol,Symbol}, Int}()
    i = 1
    for f in firms_in
        local_idx[(f, corridor, :b_low)] = i;  i += 1
        local_idx[(f, corridor, :b_mid)] = i;  i += 1
        local_idx[(f, corridor, :b_hi)]  = i;  i += 1
        local_idx[(f, corridor, :freq)]  = i;  i += 1
    end

    best_welfare = -Inf
    best_config  = zeros(Int, n_vars)
    n_optima     = 0

    x = zeros(Int, n_vars)

    for bits in 0:(2^n_vars - 1)
        # decode bits into x
        for v in 1:n_vars
            x[v] = (bits >> (v - 1)) & 1
        end

        # check one-hot feasibility
        feasible = true
        for f in firms_in
            bl = x[local_idx[(f, corridor, :b_low)]]
            bm = x[local_idx[(f, corridor, :b_mid)]]
            bh = x[local_idx[(f, corridor, :b_hi)]]
            if bl + bm + bh != 1
                feasible = false
                break
            end
        end
        feasible || continue

        # check fleet feasibility (within corridor)
        for f in firms_in
            fr = x[local_idx[(f, corridor, :freq)]]
            if fr > fleet_capacity
                feasible = false
                break
            end
        end
        feasible || continue

        # compute welfare for this corridor
        rates = Dict{Tuple{Symbol,Symbol}, Float64}()
        freqs = Dict{Tuple{Symbol,Symbol}, Float64}()
        for f in firms_in
            bl = x[local_idx[(f, corridor, :b_low)]]
            bm = x[local_idx[(f, corridor, :b_mid)]]
            bh = x[local_idx[(f, corridor, :b_hi)]]
            fr = x[local_idx[(f, corridor, :freq)]]
            rates[(f, corridor)] = RATE_LOW * bl + RATE_MID * bm + RATE_HIGH * bh
            freqs[(f, corridor)] = Float64(fr)
        end

        welfare = 0.0
        for f in firms_in
            welfare += compute_profit(f, corridor, rates, freqs,
                                      demand_params, cost, fixed_cost, participation)
        end

        if welfare > best_welfare + 1e-10
            best_welfare = welfare
            best_config  = copy(x)
            n_optima     = 1
        elseif abs(welfare - best_welfare) <= 1e-10
            n_optima += 1
        end
    end

    # Expand best_config back to full var_index keys for compatibility
    full_var_index = build_var_index(participation)
    full_config    = zeros(Int, length(full_var_index))
    for f in firms_in
        for vt in [:b_low, :b_mid, :b_hi, :freq]
            full_config[full_var_index[(f, corridor, vt)]] =
                best_config[local_idx[(f, corridor, vt)]]
        end
    end

    full_config, best_welfare, n_optima
end
