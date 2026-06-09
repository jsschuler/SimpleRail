const DEMAND_PARAMS = (
    a = 20.0,
    b = 5.0,
    d = 2.0,
    e = 3.0,
    g = 1.0,
)

const COST       = 1.0
const FIXED_COST = 5.0

"""
    compute_demand(firm, corridor, rates, freqs, params, participation)

Compute demand for `firm` on `corridor` given rate and frequency dicts.
`rates[(f,k)]` = chosen rate level (Float64).
`freqs[(f,k)]` = 1 or 0.
Clips to [0, Inf).
"""
function compute_demand(firm, corridor, rates, freqs, params, participation=PARTICIPATION)
    competitors = [f for f in FIRMS
                   if f != firm && get(participation, (f, corridor), false)]
    r_i = rates[(firm, corridor)]
    f_i = freqs[(firm, corridor)]
    cross_rate = sum(rates[(j, corridor)] for j in competitors; init=0.0)
    cross_freq = sum(freqs[(j, corridor)] for j in competitors; init=0.0)
    d = params.a - params.b * r_i + params.d * cross_rate +
        params.e * f_i - params.g * cross_freq
    max(0.0, d)
end

"""
    compute_profit(firm, corridor, rates, freqs, params, cost, fixed_cost, participation)

Profit for `firm` on `corridor`.
"""
function compute_profit(firm, corridor, rates, freqs, params, cost, fixed_cost, participation=PARTICIPATION)
    r_i = rates[(firm, corridor)]
    f_i = freqs[(firm, corridor)]
    d = compute_demand(firm, corridor, rates, freqs, params, participation)
    (r_i - cost) * d - fixed_cost * f_i
end

"""
    compute_welfare_direct(x, var_index, participation, demand_params, cost, fixed_cost)

Compute total industry welfare directly from the profit formula.
Returns -Inf if one-hot or fleet constraints are violated.
"""
function compute_welfare_direct(x, var_index, participation=PARTICIPATION,
                                demand_params=DEMAND_PARAMS, cost=COST,
                                fixed_cost=FIXED_COST, fleet_capacity=FLEET_CAPACITY)
    # check one-hot feasibility and decode rates
    rates = Dict{Tuple{Symbol,Symbol}, Float64}()
    freqs = Dict{Tuple{Symbol,Symbol}, Float64}()
    for f in FIRMS, k in CORRIDORS
        get(participation, (f, k), false) || continue
        bl = x[var_index[(f, k, :b_low)]]
        bm = x[var_index[(f, k, :b_mid)]]
        bh = x[var_index[(f, k, :b_hi)]]
        fr = x[var_index[(f, k, :freq)]]
        bl + bm + bh == 1 || return -Inf
        rates[(f, k)] = RATE_LOW * bl + RATE_MID * bm + RATE_HIGH * bh
        freqs[(f, k)] = Float64(fr)
    end
    # check fleet feasibility
    for f in FIRMS
        active_k = active_corridors_for_firm(f, participation)
        isempty(active_k) && continue
        sum(freqs[(f, k)] for k in active_k) <= fleet_capacity || return -Inf
    end
    # compute welfare
    welfare = 0.0
    for f in FIRMS, k in CORRIDORS
        get(participation, (f, k), false) || continue
        welfare += compute_profit(f, k, rates, freqs, demand_params, cost, fixed_cost, participation)
    end
    welfare
end
