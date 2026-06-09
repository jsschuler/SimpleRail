const FLEET_CAPACITY  = 2
const FLEET_PENALTY   = 50.0
const ONEHOT_PENALTY  = 50.0

# ---------------------------------------------------------------------------
# QUBO expansion notes
#
# The profit function pi[i,k] = (r[i,k] - COST) * D[i,k] - FIXED_COST * f[i,k]
# where D[i,k] = a - b*r[i,k] + d*sum(r[j,k]) + e*f[i,k] - g*sum(f[j,k])
#
# Expanding (r[i,k] - COST)*D[i,k]:
#   = (r - c)(a - b*r + d*R_-i + e*f_i - g*F_-i)
# where r = r[i,k], R_-i = sum competitor rates, F_-i = sum competitor freqs.
#
# Terms involving r^2: r^2 = r_low^2*b_low + r_mid^2*b_mid + r_hi^2*b_hi
#   (exploiting one-hot: b_x*b_y = 0 for x != y)
#
# Terms involving r*f_i: b_x*f_i — product of two distinct binary vars, quadratic.
#
# Terms involving r*r[j,k]: sum over (x,y) in {low,mid,hi}^2 of
#   r_x * r_y * b_x[i,k] * b_y[j,k] — each is a product of two distinct binary vars.
#
# Terms involving r*f[j,k]: b_x[i,k]*f[j,k] — quadratic.
#
# All cross terms are at most quadratic. No auxiliary variables needed because
# the one-hot constraint ensures degree-3 terms like b_low*b_mid*f = 0 always.
# ---------------------------------------------------------------------------

"""
    build_qubo(participation, demand_params, cost, fixed_cost,
               fleet_capacity, fleet_penalty, onehot_penalty)
    -> (Q::Matrix{Float64}, offset::Float64, var_index::Dict)

Build the QUBO matrix for the railroad coordination problem.
Q is n_vars × n_vars upper-triangular (Q[i,j] for i<=j).
The QUBO energy to minimize is x'Qx; maximizing welfare means we negate it.
Returns Q, a constant offset (for recovering welfare from energy), and var_index.
"""
function build_qubo(participation=PARTICIPATION,
                    demand_params=DEMAND_PARAMS,
                    cost=COST,
                    fixed_cost=FIXED_COST,
                    fleet_capacity=FLEET_CAPACITY,
                    fleet_penalty=FLEET_PENALTY,
                    onehot_penalty=ONEHOT_PENALTY)

    var_index = build_var_index(participation)
    n = length(var_index)
    Q = zeros(Float64, n, n)
    offset = 0.0

    # Helper: add value to upper-triangular entry
    function add_Q!(i, j, val)
        if i == j
            Q[i, i] += val
        elseif i < j
            Q[i, j] += val
        else
            Q[j, i] += val
        end
    end

    # -----------------------------------------------------------------------
    # Welfare terms (negated, since QUBO minimizes)
    # -----------------------------------------------------------------------
    # For each active (firm i, corridor k):
    #   pi[i,k] = (r[i,k] - cost) * D[i,k] - fixed_cost * f[i,k]
    #
    # D[i,k] = a - b*r[i,k] + d*sum_{j!=i}(r[j,k]) + e*f[i,k] - g*sum_{j!=i}(f[j,k])
    #
    # We expand (r[i,k] - cost)*D[i,k] in binary variables.
    # r[i,k] = sum_x r_x * b_x[i,k]
    # f[i,k] = freq variable directly
    #
    # Terms by type:
    # A) constant * r[i,k]: linear in b_x[i,k] -> diagonal Q
    # B) r[i,k]^2: sum_x r_x^2 * b_x[i,k] -> diagonal Q (one-hot)
    # C) r[i,k] * r[j,k] for j!=i: off-diagonal
    # D) r[i,k] * f[i,k]: off-diagonal (b_x[i,k] * f[i,k])
    # E) r[i,k] * f[j,k] for j!=i: off-diagonal (b_x[i,k] * f[j,k])
    # F) constant * f[i,k]: diagonal
    # G) f[i,k]^2 = f[i,k]: diagonal (binary)
    # H) f[i,k] * f[j,k] for j!=i: off-diagonal

    for firm in FIRMS, corr in CORRIDORS
        get(participation, (firm, corr), false) || continue

        competitors = [f for f in FIRMS
                       if f != firm && get(participation, (f, corr), false)]

        i_bl = var_index[(firm, corr, :b_low)]
        i_bm = var_index[(firm, corr, :b_mid)]
        i_bh = var_index[(firm, corr, :b_hi)]
        i_f  = var_index[(firm, corr, :freq)]

        rate_vars = [(i_bl, RATE_LOW), (i_bm, RATE_MID), (i_bh, RATE_HIGH)]

        # pi[i,k] = (r - cost)*(a - b*r + d*R_- + e*f_i - g*F_-) - fixed_cost*f_i
        # = r*a - b*r^2 + d*r*R_- + e*r*f_i - g*r*F_-
        #   - cost*a + cost*b*r - cost*d*R_- - cost*e*f_i + cost*g*F_-
        #   - fixed_cost*f_i
        #
        # Negated for minimization:

        a = demand_params.a
        b = demand_params.b
        d = demand_params.d
        e = demand_params.e
        g = demand_params.g

        # -- terms linear in r[i,k] (diagonal in b_x[i,k]) --
        # +r*a: coeff = a
        # -b*r^2 = -b*sum_x r_x^2*b_x (diagonal, one-hot)
        # +cost*b*r: coeff = cost*b
        for (iv, rv) in rate_vars
            coeff = a * rv - b * rv^2 + cost * b * rv
            add_Q!(iv, iv, -coeff)   # negated
        end

        # -- constant -cost*a (goes to offset, contributes +cost*a to welfare) --
        offset += cost * a   # offset is added back when recovering welfare

        # -- r[i,k] * R_-i: d * r[i,k] * sum_j r[j,k] --
        # and - cost * d * R_-i (linear in r[j,k] vars)
        for j in competitors
            j_bl = var_index[(j, corr, :b_low)]
            j_bm = var_index[(j, corr, :b_mid)]
            j_bh = var_index[(j, corr, :b_hi)]
            comp_rate_vars = [(j_bl, RATE_LOW), (j_bm, RATE_MID), (j_bh, RATE_HIGH)]

            # d * r_i * r_j: off-diagonal b_x[i] * b_y[j]
            for (iv, rv) in rate_vars
                for (jv, rjv) in comp_rate_vars
                    add_Q!(iv, jv, -d * rv * rjv)
                end
            end

            # -cost*d * r[j,k]: linear in b_y[j]
            for (jv, rjv) in comp_rate_vars
                add_Q!(jv, jv, cost * d * rjv)   # negated negated = +
            end
        end

        # -- e * r[i,k] * f[i,k]: off-diagonal b_x[i] * f[i] --
        for (iv, rv) in rate_vars
            add_Q!(iv, i_f, -e * rv)
        end

        # -- -g * r[i,k] * F_-i: -g * r[i,k] * sum_j f[j,k] --
        for j in competitors
            j_f = var_index[(j, corr, :freq)]
            for (iv, rv) in rate_vars
                add_Q!(iv, j_f, g * rv)   # negated: -(-g*r*f_j) = +g*r*f_j
            end
        end

        # -- f[i,k] terms --
        # -cost*e * f[i,k]: diagonal
        add_Q!(i_f, i_f, cost * e)

        # +cost*g * F_-i: +cost*g * sum_j f[j,k], linear in f[j,k]
        for j in competitors
            j_f = var_index[(j, corr, :freq)]
            add_Q!(j_f, j_f, -cost * g)
        end

        # -fixed_cost * f[i,k]: diagonal
        add_Q!(i_f, i_f, fixed_cost)

        # -- f[i,k]^2 = f[i,k] terms --
        # e*f[i,k] from r-term already handled above; this is the stand-alone
        # f*f cross among competitors:
        # -(-g * f[i,k] * sum_j f[j,k]) wait — these come from a cross term:
        # cost*g * F_-i is already handled as linear; no f[i]*f[j] cross from it.
        # The only f*f cross comes from the demand interaction with f[i]:
        # The demand term has no f_i * f_j cross except through the frequency
        # competitor substitution, which we already captured linearly.
        # (There is no quadratic f[i]*f[j] term in the profit directly since
        #  f_i appears only in D[i,k] and the competitor effect is linear in f_j.)
    end

    # -----------------------------------------------------------------------
    # One-hot penalty: ONEHOT_PENALTY * (b_low + b_mid + b_hi - 1)^2
    # Expand: sum_x b_x^2 + 2*sum_{x<y} b_x*b_y - 2*sum_x b_x + 1
    # diagonal: (1 - 2)*b_x = -b_x for each x  (b_x^2 = b_x for binary)
    # off-diag: 2*b_x*b_y for x < y
    # constant: 1 (goes to offset)
    # -----------------------------------------------------------------------
    for firm in FIRMS, corr in CORRIDORS
        get(participation, (firm, corr), false) || continue
        i_bl = var_index[(firm, corr, :b_low)]
        i_bm = var_index[(firm, corr, :b_mid)]
        i_bh = var_index[(firm, corr, :b_hi)]

        # diagonal: (1 - 2) = -1 per variable
        add_Q!(i_bl, i_bl, onehot_penalty * (-1.0))
        add_Q!(i_bm, i_bm, onehot_penalty * (-1.0))
        add_Q!(i_bh, i_bh, onehot_penalty * (-1.0))

        # off-diagonal: 2 for each pair
        add_Q!(i_bl, i_bm, onehot_penalty * 2.0)
        add_Q!(i_bl, i_bh, onehot_penalty * 2.0)
        add_Q!(i_bm, i_bh, onehot_penalty * 2.0)

        # constant
        offset += onehot_penalty * 1.0
    end

    # -----------------------------------------------------------------------
    # Fleet penalty: FLEET_PENALTY * (sum_k f[i,k] - fleet_capacity)^2
    # Expand: sum_k f[i,k]^2 + 2*sum_{k<k'} f[i,k]*f[i,k']
    #         - 2*fleet_capacity*sum_k f[i,k] + fleet_capacity^2
    # diagonal: (1 - 2*fleet_capacity)*f[i,k]
    # off-diag: 2*f[i,k]*f[i,k'] for k < k'
    # constant: fleet_capacity^2
    # -----------------------------------------------------------------------
    for firm in FIRMS
        active_k = active_corridors_for_firm(firm, participation)
        isempty(active_k) && continue
        freq_vars = [var_index[(firm, k, :freq)] for k in active_k]

        # diagonal
        for iv in freq_vars
            add_Q!(iv, iv, fleet_penalty * (1.0 - 2.0 * fleet_capacity))
        end

        # off-diagonal pairs
        for a_idx in 1:length(freq_vars), b_idx in (a_idx+1):length(freq_vars)
            add_Q!(freq_vars[a_idx], freq_vars[b_idx], fleet_penalty * 2.0)
        end

        # constant
        offset += fleet_penalty * fleet_capacity^2
    end

    Q, offset, var_index
end

"""
    welfare_from_config(x, Q, offset, var_index, participation)

Recover total industry welfare from a configuration vector x and the QUBO matrix.

The QUBO energy E(x) = x'Qx + offset = -W(x) + fleet_pen(x) + OH_pen(x),
so W(x) = -(x'Qx + offset) + fleet_pen(x) + OH_pen(x).
The penalty terms must be added back explicitly because feasible configs
may still have non-zero fleet_pen when sum(f) < fleet_cap.
"""
function welfare_from_config(x, Q, offset, var_index, participation=PARTICIPATION,
                              fleet_capacity=FLEET_CAPACITY)
    energy = qubo_energy(x, Q)

    oh_pen = 0.0
    for f in FIRMS, k in CORRIDORS
        get(participation, (f, k), false) || continue
        s = x[var_index[(f, k, :b_low)]] + x[var_index[(f, k, :b_mid)]] + x[var_index[(f, k, :b_hi)]]
        oh_pen += ONEHOT_PENALTY * (s - 1)^2
    end

    fleet_pen = 0.0
    for f in FIRMS
        active_k = active_corridors_for_firm(f, participation)
        isempty(active_k) && continue
        s = sum(x[var_index[(f, k, :freq)]] for k in active_k)
        fleet_pen += FLEET_PENALTY * (s - fleet_capacity)^2
    end

    -(energy + offset) + fleet_pen + oh_pen
end
