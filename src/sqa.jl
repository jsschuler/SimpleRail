"""
    run_sqa(Q, offset, var_index, participation;
            P, beta, gamma_0, max_steps, fleet_capacity, rng)
    -> (best_config, best_welfare, final_config, final_welfare, feasible)

Run one trajectory of simulated quantum annealing (path-integral / Trotter SQA).
P Trotter slices; proposal distribution matches SA (rate rotate, freq flip, joint swap).
One-hot violations are hard-rejected; fleet violations are handled by penalty energy.
Tracks best welfare across ALL slices at EVERY accepted step.
"""
function run_sqa(Q, offset, var_index=build_var_index(),
                 participation=PARTICIPATION;
                 P=20,
                 beta=15.0,
                 gamma_0=5.0,
                 max_steps=50_000,
                 fleet_capacity=FLEET_CAPACITY,
                 rng=Random.default_rng())

    n = length(var_index)

    # Initialize P replicas from independent random feasible configs
    replicas = [random_feasible_config(var_index, participation, fleet_capacity, rng)
                for _ in 1:P]
    energies = [qubo_energy(r, Q) for r in replicas]

    best_x       = copy(replicas[argmin(energies)])
    best_welfare = offset - minimum(energies)

    x_new = zeros(Int, n)

    for step in 1:max_steps
        # Linearly anneal transverse field
        gamma = gamma_0 * (1.0 - step / max_steps)
        # Trotter coupling: J_perp = -(P/2beta) * log(tanh(beta*gamma/P))
        # Clamp to avoid log(0) when gamma -> 0
        bg_over_P = beta * gamma / P
        J_perp = if bg_over_P < 1e-10
            Inf
        else
            -(P / (2 * beta)) * log(tanh(bg_over_P))
        end

        # Pick a random replica to propose to
        slice = rand(rng, 1:P)
        copyto!(x_new, replicas[slice])

        r = rand(rng)
        if r < 0.5
            propose_freq_flip!(x_new, var_index, participation, rng)
        elseif r < 0.8
            propose_rate_rotate!(x_new, var_index, participation, rng)
        else
            propose_joint_swap!(x_new, var_index, participation, rng)
        end

        # Hard-enforce one-hot: check if the proposed state has any violation
        one_hot_ok = true
        for firm in FIRMS, corr in CORRIDORS
            get(participation, (firm, corr), false) || continue
            bl = x_new[var_index[(firm, corr, :b_low)]]
            bm = x_new[var_index[(firm, corr, :b_mid)]]
            bh = x_new[var_index[(firm, corr, :b_hi)]]
            if bl + bm + bh != 1
                one_hot_ok = false
                break
            end
        end
        one_hot_ok || continue

        e_new = qubo_energy(x_new, Q)
        e_old = energies[slice]

        # Classical energy difference for this slice
        dE_classical = e_new - e_old

        # Trotter coupling to neighboring slices (periodic)
        slice_prev = (slice == 1) ? P : slice - 1
        slice_next = (slice == P) ? 1 : slice + 1

        dE_trotter = if isinf(J_perp)
            # At gamma=0 the transverse field is off; no Trotter coupling
            0.0
        else
            J_perp * sum(
                (x_new[v] - replicas[slice][v]) *
                (replicas[slice_prev][v] + replicas[slice_next][v])
                for v in 1:n
            )
        end

        dE = dE_classical + dE_trotter

        if dE < 0 || rand(rng) < exp(-beta * dE / P)
            copyto!(replicas[slice], x_new)
            energies[slice] = e_new

            # Track best across all slices
            for p in 1:P
                w = offset - energies[p]
                if w > best_welfare
                    best_welfare = w
                    copyto!(best_x, replicas[p])
                end
            end
        end
    end

    # Final: best slice by energy
    final_slice   = argmin(energies)
    final_welfare = offset - energies[final_slice]
    feasible      = compute_welfare_direct(best_x, var_index, participation,
                                           DEMAND_PARAMS, COST, FIXED_COST,
                                           fleet_capacity) > -Inf
    best_x, best_welfare, copy(replicas[final_slice]), final_welfare, feasible
end
