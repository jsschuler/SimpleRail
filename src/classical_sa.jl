"""
    random_feasible_config(var_index, participation, fleet_capacity, rng)
    -> Vector{Int}

Generate a uniformly random feasible configuration.
For each active (firm, corridor): choose one of {b_low, b_mid, b_hi} at random.
For each firm: choose min(n_active_corridors, fleet_capacity) corridors at random
to set freq=1; set freq=0 elsewhere.
"""
function random_feasible_config(var_index=build_var_index(),
                                participation=PARTICIPATION,
                                fleet_capacity=FLEET_CAPACITY,
                                rng=Random.default_rng())
    n = length(var_index)
    x = zeros(Int, n)

    # rate variables: one-hot per active pair
    for firm in FIRMS, corr in CORRIDORS
        get(participation, (firm, corr), false) || continue
        chosen = rand(rng, [:b_low, :b_mid, :b_hi])
        x[var_index[(firm, corr, chosen)]] = 1
    end

    # frequency and slack variables
    for firm in FIRMS
        active_k = active_corridors_for_firm(firm, participation)
        n_active = length(active_k)
        isempty(active_k) && continue

        if n_active <= fleet_capacity
            # Fleet constraint can never bind — uniformly sample any freq assignment.
            n_high = rand(rng, 0:n_active)
        else
            # Fleet-constrained: sample freq_sum uniformly in {0,...,fleet_capacity}.
            n_high = rand(rng, 0:fleet_capacity)
        end

        perm = randperm(rng, n_active)
        for pos in 1:n_high
            x[var_index[(firm, active_k[perm[pos]], :freq)]] = 1
        end

        # Set slack vars (if any) to make sum(f) + sum(s) = fleet_capacity.
        s_vars = slack_vars_for_firm(firm, var_index, fleet_capacity)
        slack_needed = fleet_capacity - n_high
        for (pos, iv) in enumerate(s_vars)
            x[iv] = (pos - 1) < slack_needed ? 1 : 0
        end
    end

    x
end

"""
    qubo_energy(x, Q)

Compute x'Qx for upper-triangular Q.
"""
function qubo_energy(x, Q)
    e = 0.0
    n = length(x)
    for i in 1:n
        x[i] == 0 && continue
        for j in i:n
            x[j] == 0 && continue
            e += (i == j) ? Q[i, i] : Q[i, j]
        end
    end
    e
end

# ---------------------------------------------------------------------------
# Proposal distribution helpers
# ---------------------------------------------------------------------------

"""
    propose_freq_flip!(x, var_index, participation, rng) -> (x_new, delta_desc)

Flip a single frequency variable chosen uniformly at random.
Returns modified copy and the variable index that was flipped.
"""
function propose_freq_flip!(x_new, var_index, participation, rng)
    freq_vars = [var_index[(f, k, :freq)]
                 for f in FIRMS, k in CORRIDORS
                 if get(participation, (f, k), false)]
    iv = rand(rng, vec(freq_vars))
    x_new[iv] = 1 - x_new[iv]
    iv
end

"""
    propose_rate_rotate!(x_new, var_index, participation, rng)

Rotate rate for a random (firm, corridor): b_low→b_mid→b_hi→b_low.
Preserves one-hot feasibility.
"""
function propose_rate_rotate!(x_new, var_index, participation, rng)
    pairs = active_pairs(participation)
    (firm, corr) = rand(rng, pairs)
    i_bl = var_index[(firm, corr, :b_low)]
    i_bm = var_index[(firm, corr, :b_mid)]
    i_bh = var_index[(firm, corr, :b_hi)]
    # find current active rate
    if x_new[i_bl] == 1
        x_new[i_bl] = 0; x_new[i_bm] = 1
    elseif x_new[i_bm] == 1
        x_new[i_bm] = 0; x_new[i_bh] = 1
    else
        x_new[i_bh] = 0; x_new[i_bl] = 1
    end
end

"""
    propose_joint_swap!(x_new, var_index, participation, rng)

Rotate rate AND flip frequency for a random (firm, corridor).
"""
function propose_joint_swap!(x_new, var_index, participation, rng)
    pairs = active_pairs(participation)
    (firm, corr) = rand(rng, pairs)
    # rotate rate
    i_bl = var_index[(firm, corr, :b_low)]
    i_bm = var_index[(firm, corr, :b_mid)]
    i_bh = var_index[(firm, corr, :b_hi)]
    if x_new[i_bl] == 1
        x_new[i_bl] = 0; x_new[i_bm] = 1
    elseif x_new[i_bm] == 1
        x_new[i_bm] = 0; x_new[i_bh] = 1
    else
        x_new[i_bh] = 0; x_new[i_bl] = 1
    end
    # flip freq
    i_f = var_index[(firm, corr, :freq)]
    x_new[i_f] = 1 - x_new[i_f]
end

"""
    propose_slack_flip!(x_new, var_index, participation, fleet_capacity, rng)

Flip one slack variable chosen uniformly at random from all firms that have slack vars.
"""
function propose_slack_flip!(x_new, var_index, participation, fleet_capacity, rng)
    all_slack = Int[]
    for f in FIRMS
        append!(all_slack, slack_vars_for_firm(f, var_index, fleet_capacity))
    end
    isempty(all_slack) && return
    iv = rand(rng, all_slack)
    x_new[iv] = 1 - x_new[iv]
end

"""
    run_sa(Q, offset, var_index, participation;
           T0, alpha, max_steps, fleet_capacity, rng)
    -> (best_config, best_welfare, final_config, final_welfare, feasible)

Run one trajectory of simulated annealing.
Returns best config/welfare seen during the run (not just final state).
"""
function run_sa(Q, offset, var_index=build_var_index(),
                participation=PARTICIPATION;
                T0=10.0, alpha=0.997, max_steps=50_000,
                fleet_capacity=FLEET_CAPACITY,
                rng=Random.default_rng())

    x       = random_feasible_config(var_index, participation, fleet_capacity, rng)
    e_cur   = qubo_energy(x, Q)
    best_e  = e_cur

    best_x  = copy(x)
    T       = T0
    x_new   = similar(x)

    for step in 1:max_steps
        copyto!(x_new, x)

        r = rand(rng)
        if r < 0.42
            propose_freq_flip!(x_new, var_index, participation, rng)
        elseif r < 0.70
            propose_rate_rotate!(x_new, var_index, participation, rng)
        elseif r < 0.88
            propose_joint_swap!(x_new, var_index, participation, rng)
        else
            propose_slack_flip!(x_new, var_index, participation, fleet_capacity, rng)
        end

        e_new = qubo_energy(x_new, Q)
        dE    = e_new - e_cur

        if dE < 0 || rand(rng) < exp(-dE / T)
            copyto!(x, x_new)
            e_cur = e_new

            if e_cur < best_e
                best_e = e_cur
                copyto!(best_x, x)
            end
        end

        T *= alpha
    end

    best_welfare  = welfare_from_config(best_x, Q, offset, var_index, participation, fleet_capacity)
    final_welfare = welfare_from_config(x, Q, offset, var_index, participation, fleet_capacity)
    feasible      = compute_welfare_direct(best_x, var_index, participation,
                                           DEMAND_PARAMS, COST, FIXED_COST,
                                           fleet_capacity) > -Inf
    best_x, best_welfare, copy(x), final_welfare, feasible
end

"""
    estimate_t0(var_index, participation, fleet_capacity, n_samples, rng)
    -> Float64

Estimate T0 as std of welfare across random feasible configurations.
"""
function estimate_t0(var_index=build_var_index(),
                     participation=PARTICIPATION,
                     fleet_capacity=FLEET_CAPACITY,
                     n_samples=500,
                     rng=Random.default_rng())
    Q, offset, _ = build_qubo(participation)
    welfares = [begin
        x = random_feasible_config(var_index, participation, fleet_capacity, rng)
        offset - qubo_energy(x, Q)
    end for _ in 1:n_samples]
    std(welfares)
end
